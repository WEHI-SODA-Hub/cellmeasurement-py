"""Crash-safe on-disk store for extracted polygon geometries.

Polygon extraction over a million-cell mask takes long enough that a job can be
killed (wall-time, OOM) before it finishes.  This module persists geometries
incrementally as independent *shards* so a restart resumes from the last
completed batch instead of starting over.

Encoding
--------
Polygons are stored as flat ragged arrays rather than pickled Shapely objects
or GeoJSON text.  For ~1.2M cells the compressed shards total ~85 MB, against
~440 MB for a pickled ``dict[int, Polygon]`` and ~500 MB for JSONL, and the
bulk reload path (:func:`shapely.linearrings` / :func:`shapely.polygons`)
rebuilds every polygon in C rather than one Python call per cell.

Coordinates are ``float32``.  Marching-squares output lies on a half-pixel grid,
and float32 represents such values exactly well beyond any realistic image
width, so the round trip is lossless.

Each shard is one ``.npz`` holding four arrays:

``ids``
    ``uint32``, one label ID per polygon.
``ring_counts``
    ``int32``, vertex count per ring (exterior ring first, then any holes).
``poly_ring_counts``
    ``int32``, number of rings per polygon (``1`` when the polygon has no
    holes).  Holes are preserved because the raster mask export carves them
    out of a cell's footprint.
``coords``
    ``float32`` of shape ``(total_vertices, 2)`` as ``(x, y)``.

Shards are written to ``<name>.tmp`` and then :func:`os.replace`\\ d into place,
which is atomic on POSIX.  A process killed mid-write therefore leaves either a
complete shard or a stray temp file, never a half-readable one.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon

__all__ = ["GeometryShardStore", "arrays_to_polygons", "polygons_to_arrays"]

logger = logging.getLogger(__name__)

_SHARD_RE = re.compile(r"^geoms-(\d{6})\.npz$")


def polygons_to_arrays(
    items: list[tuple[int, Polygon]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten ``(label_id, Polygon)`` pairs into the ragged shard arrays.

    Args:
        items: Label ID / polygon pairs, in the order they should be stored.

    Returns:
        ``(ids, ring_counts, poly_ring_counts, coords)`` ready to hand to
        :meth:`GeometryShardStore.write_shard`.
    """
    ids: list[int] = []
    ring_counts: list[int] = []
    poly_ring_counts: list[int] = []
    chunks: list[np.ndarray] = []

    for label_id, poly in items:
        rings = [poly.exterior, *poly.interiors]
        n_rings = 0
        for ring in rings:
            xy = np.asarray(ring.coords, dtype=np.float32)
            # A ring needs 4 positions (3 distinct + closing point) to bound area.
            if xy.shape[0] < 4:
                continue
            chunks.append(xy)
            ring_counts.append(int(xy.shape[0]))
            n_rings += 1
        if n_rings == 0:
            # Exterior was degenerate; drop the polygon rather than emit a
            # shard entry that cannot be rebuilt.
            continue
        ids.append(int(label_id))
        poly_ring_counts.append(n_rings)

    if not ids:
        return (
            np.empty(0, np.uint32),
            np.empty(0, np.int32),
            np.empty(0, np.int32),
            np.empty((0, 2), np.float32),
        )

    return (
        np.asarray(ids, np.uint32),
        np.asarray(ring_counts, np.int32),
        np.asarray(poly_ring_counts, np.int32),
        np.concatenate(chunks).astype(np.float32, copy=False),
    )


def arrays_to_polygons(
    ids: np.ndarray,
    ring_counts: np.ndarray,
    poly_ring_counts: np.ndarray,
    coords: np.ndarray,
) -> dict[int, Polygon]:
    """Rebuild polygons from one shard's ragged arrays in two vectorised calls."""
    if ids.size == 0:
        return {}

    ring_index = np.repeat(np.arange(ring_counts.size), ring_counts)
    rings = shapely.linearrings(coords.astype(np.float64, copy=False), indices=ring_index)

    # The first ring of each group is the shell; the rest become holes.
    poly_index = np.repeat(np.arange(poly_ring_counts.size), poly_ring_counts)
    polys = shapely.polygons(rings, indices=poly_index)

    return {int(label_id): poly for label_id, poly in zip(ids, polys) if poly is not None}


class GeometryShardStore:
    """Append-only shard directory for extracted geometries.

    Args:
        directory: Directory holding the shards. Created if absent.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- fingerprint ------------------------------------------------------

    @property
    def _fingerprint_path(self) -> Path:
        return self.directory / "params.json"

    def reconcile_fingerprint(self, fingerprint: dict[str, object]) -> bool:
        """Decide whether existing shards may be reused, recording the fingerprint.

        A checkpoint directory outlives the run that created it, so a later run
        with a different mask, tolerance, or batch size could otherwise silently
        inherit geometries that do not correspond to its own inputs. When the
        fingerprint differs, every shard is discarded so extraction restarts
        cleanly.

        Args:
            fingerprint: Values that must match for shards to remain valid.

        Returns:
            ``True`` when existing shards are reusable, ``False`` when they were
            discarded.
        """
        if self._fingerprint_path.exists():
            try:
                stored = json.loads(self._fingerprint_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                stored = None
            if stored == fingerprint:
                return True
            logger.warning(
                "Geometry checkpoint in %s was built with different settings "
                "(stored=%s, current=%s); discarding %d stale shard(s).",
                self.directory,
                stored,
                fingerprint,
                len(self.completed_batches()),
            )
            self.clear()

        self._fingerprint_path.write_text(json.dumps(fingerprint, sort_keys=True), encoding="utf-8")
        return False

    def clear(self) -> None:
        """Remove every shard and the recorded fingerprint."""
        for path in self.directory.glob("geoms-*.npz"):
            path.unlink()
        for path in self.directory.glob("*.npz.tmp"):
            path.unlink()
        self._fingerprint_path.unlink(missing_ok=True)

    # -- writing ----------------------------------------------------------

    def write_shard(
        self,
        batch_index: int,
        ids: np.ndarray,
        ring_counts: np.ndarray,
        poly_ring_counts: np.ndarray,
        coords: np.ndarray,
    ) -> Path:
        """Atomically persist one batch's geometries.

        Args:
            batch_index: Batch ordinal; also names the shard.
            ids: ``uint32`` label IDs.
            ring_counts: ``int32`` vertex count per ring.
            poly_ring_counts: ``int32`` ring count per polygon.
            coords: ``float32`` ``(total_vertices, 2)`` vertex array.

        Returns:
            Path to the written shard.
        """
        final = self._shard_path(batch_index)
        tmp = final.with_suffix(".npz.tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(
                fh,
                ids=ids,
                ring_counts=ring_counts,
                poly_ring_counts=poly_ring_counts,
                coords=coords,
            )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, final)
        return final

    def _shard_path(self, batch_index: int) -> Path:
        return self.directory / f"geoms-{batch_index:06d}.npz"

    # -- resume -----------------------------------------------------------

    def completed_batches(self) -> set[int]:
        """Return the batch indices already persisted as complete shards.

        Stray ``.npz.tmp`` files from a killed run are removed, since a shard is
        only ever renamed into place once fully written and fsynced.
        """
        for stale in self.directory.glob("*.npz.tmp"):
            try:
                stale.unlink()
            except OSError:
                logger.warning("Could not remove stale shard temp file: %s", stale)

        done: set[int] = set()
        for path in self.directory.iterdir():
            match = _SHARD_RE.match(path.name)
            if match is not None:
                done.add(int(match.group(1)))
        return done

    # -- reading ----------------------------------------------------------

    def load(self) -> dict[int, Polygon]:
        """Rebuild every stored polygon, keyed by label ID."""
        geoms: dict[int, Polygon] = {}
        for path in sorted(self.directory.glob("geoms-*.npz")):
            with np.load(path) as data:
                geoms.update(
                    arrays_to_polygons(
                        data["ids"],
                        data["ring_counts"],
                        data["poly_ring_counts"],
                        data["coords"],
                    )
                )
        logger.info("Loaded %d geometries from %s", len(geoms), self.directory)
        return geoms
