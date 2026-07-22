"""Raster-to-polygon geometry utilities.

Pre-extracts simplified polygon geometries from raster label masks so that
downstream steps (measurement, GeoJSON export) work with vectors rather than
re-querying the zarr store per cell.

Extraction mirrors the measurement pipeline's execution model: the label array
is materialised once, every bounding box comes from a single
:func:`scipy.ndimage.find_objects` pass, and labels are polygonized in batches
across a process pool with the labels held in shared memory.

Typical pipeline position
--------------------------
1. Load masks (``SegmentationMask``)
2. **extract_label_geometries** — batched, process-parallel, resumable
3. ``match_rois`` — raster-based matching (unchanged)
4. Convert synthesised masks → polygons (where needed)
5. Measure / export using pre-simplified geometry dicts
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import scipy.ndimage as ndi
from shapely.geometry import Polygon
from shapely.ops import unary_union
from skimage.measure import find_contours

from ..shared_array import SharedArraySpec, create_shared_array, open_shared_array
from .geometry_store import GeometryShardStore, arrays_to_polygons, polygons_to_arrays

if TYPE_CHECKING:
    import dask.array as da
    import geopandas as gpd

__all__ = [
    "extract_label_geometries",
    "mask_to_geometry",
    "boundaries_to_geometries",
]

logger = logging.getLogger(__name__)

DEFAULT_GEOMETRY_BATCH_SIZE = 2000


# ---------------------------------------------------------------------------
# Core raster-to-polygon primitive
# ---------------------------------------------------------------------------


def mask_to_geometry(
    mask: np.ndarray,
    simplify: bool = True,
    tolerance: float = 0.5,
    row_offset: int = 0,
    col_offset: int = 0,
) -> Polygon | None:
    """Convert a binary 2-D mask to a Shapely Polygon.

    Uses marching-squares contours at the 0.5 iso-level, merges any disjoint
    pieces via ``unary_union``, and optionally simplifies with Douglas-Peucker.
    Returns the single largest polygon (QuPath convention: one polygon per
    cell detection).

    Args:
        mask: 2-D binary array (truthy = foreground).
        simplify: Apply Douglas-Peucker simplification.
        tolerance: Simplification tolerance in pixels.
        row_offset: Global row offset added to contour coordinates.
        col_offset: Global column offset added to contour coordinates.

    Returns:
        The cell polygon in global image coordinates, or ``None`` if the mask
        is empty or produces no valid geometry.
    """
    if not np.any(mask):
        return None

    # Pad with a 1-px zero border so marching-squares contours are always
    # closed (never clipped at the array edge) and the minimum array size
    # requirement of find_contours (2×2) is always met.
    padded = np.pad(mask.astype(np.uint8), pad_width=1)
    contours = find_contours(padded, level=0.5)
    if not contours:
        return None

    polys = []
    for c in contours:
        if len(c) < 3:
            continue
        # find_contours returns (row, col) in padded space; subtract 1 for the
        # pad offset, then add global offsets. GeoJSON/Shapely want (x=col, y=row).
        xy = [(float(col_offset + p[1] - 1), float(row_offset + p[0] - 1)) for p in c]
        poly = Polygon(xy)
        if poly.is_empty:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty:
            polys.append(poly)

    if not polys:
        return None

    g = unary_union(polys)

    # Discard non-polygon artefacts (LineStrings, Points) from boolean union.
    if g.geom_type == "GeometryCollection":
        keep = [
            p for p in g.geoms  # type: ignore[union-attr]
            if p.geom_type in ("Polygon", "MultiPolygon") and not p.is_empty
        ]
        if not keep:
            return None
        g = unary_union(keep)

    # Keep only the largest polygon (matches QuPath's single-polygon convention).
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda p: p.area)  # type: ignore[union-attr]

    if g.geom_type != "Polygon" or g.is_empty:
        return None

    if simplify and tolerance > 0:
        g = g.simplify(tolerance, preserve_topology=True)

    if not g.is_valid:
        g = g.buffer(0)

    return g if isinstance(g, Polygon) and not g.is_empty else None


# ---------------------------------------------------------------------------
# Chunk-level bbox helpers (no dependency on roi_matcher)
# ---------------------------------------------------------------------------


def _materialize_labels(label_arr: da.Array | np.ndarray) -> np.ndarray:
    """Bring a label array fully into memory exactly once.

    A single materialisation is cheap: a 21006x39138 ``uint32`` mask is only
    ~3.3 GB, and the measurement pipeline already materialises the same array
    for the same reason.
    """
    if isinstance(label_arr, np.ndarray):
        return label_arr
    compute_fn = getattr(label_arr, "compute", None)
    if callable(compute_fn):
        return cast(np.ndarray, compute_fn())
    return np.asarray(label_arr)


def _label_bboxes(labels: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    """Per-label bounding boxes via a single C-level pass.

    Returns ``{label_id: (r0, c0, r1, c1)}`` with **exclusive** row/col stops,
    so ``labels[r0:r1, c0:c1]`` is the label's bounding-box crop.

    ``find_objects`` allocates one list slot per value in ``1..labels.max()``,
    which suits the dense ``1..N`` labelling every supported segmenter emits.
    A mask whose IDs were sparse across a very wide range would make that list
    disproportionately large.
    """
    slices = ndi.find_objects(labels)
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for index, slc in enumerate(slices):
        if slc is None:
            continue
        row_slice, col_slice = slc
        boxes[index + 1] = (
            int(row_slice.start),
            int(col_slice.start),
            int(row_slice.stop),
            int(col_slice.stop),
        )
    return boxes


# ---------------------------------------------------------------------------
# Process-parallel batch polygonization
# ---------------------------------------------------------------------------

# Per-worker state, populated by the pool initializer.
_WORKER_LABELS: np.ndarray | None = None
_WORKER_SHM: SharedMemory | None = None


def _init_geometry_worker(spec: SharedArraySpec) -> None:
    """Attach the worker to the parent's shared label array."""
    global _WORKER_LABELS, _WORKER_SHM
    _WORKER_LABELS, _WORKER_SHM = open_shared_array(spec)


def _polygonize_batch(
    batch: tuple[int, tuple[tuple[int, int, int, int, int], ...], bool, float],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Polygonize one batch of labels and return flat ragged arrays.

    Returning arrays rather than Shapely objects keeps the inter-process
    payload small; the parent never rebuilds a polygon it is only going to
    write straight to disk.
    """
    batch_index, entries, simplify, tolerance = batch
    labels = _WORKER_LABELS
    if labels is None:
        raise RuntimeError("Geometry worker was not initialised with a label array")

    items: list[tuple[int, Polygon]] = []
    for label_id, r0, c0, r1, c1 in entries:
        crop = labels[r0:r1, c0:c1]
        poly = mask_to_geometry(crop == label_id, simplify, tolerance, r0, c0)
        if poly is not None:
            items.append((label_id, poly))

    ids, ring_counts, poly_ring_counts, coords = polygons_to_arrays(items)
    return batch_index, ids, ring_counts, poly_ring_counts, coords


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _build_batches(
    bboxes: dict[int, tuple[int, int, int, int]],
    batch_size: int,
) -> list[tuple[int, tuple[tuple[int, int, int, int, int], ...]]]:
    """Split labels into deterministically numbered batches.

    Batch numbering must be stable across runs for checkpoint resume to be able
    to skip completed work, so labels are always sorted and chunked identically.
    """
    label_ids = sorted(bboxes)
    batches: list[tuple[int, tuple[tuple[int, int, int, int, int], ...]]] = []
    for batch_index, start in enumerate(range(0, len(label_ids), batch_size)):
        entries = tuple(
            (label_id, *bboxes[label_id]) for label_id in label_ids[start : start + batch_size]
        )
        batches.append((batch_index, entries))
    return batches


def extract_label_geometries(
    label_arr: da.Array | np.ndarray,
    simplify: bool = True,
    tolerance: float = 0.5,
    workers: int = 1,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    batch_size: int = DEFAULT_GEOMETRY_BATCH_SIZE,
) -> dict[int, Polygon]:
    """Extract and optionally simplify polygon geometries for all labels.

    Materialises the label array once, derives every bounding box in a single
    :func:`scipy.ndimage.find_objects` pass, then polygonizes labels in batches
    across a process pool with the labels held in shared memory.

    When ``checkpoint_dir`` is set, each completed batch is persisted as a shard
    before the next is collected, so a job killed by wall-time or OOM can resume
    from the last completed batch instead of restarting.

    Args:
        label_arr: 2-D integer label array (dask or NumPy).
        simplify: Apply Douglas-Peucker simplification.
        tolerance: Simplification tolerance in pixels.
        workers: Worker processes. ``<= 1`` runs serially in-process.
        checkpoint_dir: Directory for resumable geometry shards. ``None``
            disables checkpointing.
        resume: Reuse shards already present in ``checkpoint_dir``.
        batch_size: Labels per batch. Also the checkpoint granularity.

    Returns:
        Mapping of label ID to Shapely Polygon in global image coordinates.
        Labels that produce no valid contour are omitted.
    """
    t_start = time.perf_counter()
    labels = _materialize_labels(label_arr)
    logger.info(
        "Materialized labels for geometry extraction in %.2fs: shape=%s, dtype=%s",
        time.perf_counter() - t_start,
        labels.shape,
        labels.dtype,
    )

    t_bbox = time.perf_counter()
    bboxes = _label_bboxes(labels)
    if not bboxes:
        return {}
    logger.info(
        "Collected %d label bounding boxes in %.2fs", len(bboxes), time.perf_counter() - t_bbox
    )

    batches = _build_batches(bboxes, batch_size)
    store = GeometryShardStore(checkpoint_dir) if checkpoint_dir is not None else None

    reusable = False
    if store is not None:
        if not resume:
            store.clear()
        # Guard against a checkpoint left behind by a run with different inputs
        # or settings, whose shards would otherwise be silently adopted.
        reusable = store.reconcile_fingerprint(
            {
                "simplify": bool(simplify),
                "tolerance": float(tolerance),
                "batch_size": int(batch_size),
                "n_labels": len(bboxes),
                "image_shape": [int(labels.shape[0]), int(labels.shape[1])],
            }
        )

    done: set[int] = set()
    if store is not None and resume and reusable:
        done = store.completed_batches()
        if done:
            logger.info(
                "Resuming geometry extraction: %d/%d batches already checkpointed in %s",
                len(done),
                len(batches),
                store.directory,
            )
    pending = [batch for batch in batches if batch[0] not in done]

    logger.info(
        "Geometry extraction start: labels=%d, batches=%d (pending=%d), "
        "batch_size=%d, workers=%d, checkpoint=%s",
        len(bboxes),
        len(batches),
        len(pending),
        batch_size,
        workers,
        store.directory if store is not None else "disabled",
    )

    geoms: dict[int, Polygon] = {}
    t_poly = time.perf_counter()
    completed = 0

    def _handle(result: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        nonlocal completed
        batch_index, ids, ring_counts, poly_ring_counts, coords = result
        if store is not None:
            store.write_shard(batch_index, ids, ring_counts, poly_ring_counts, coords)
        else:
            geoms.update(arrays_to_polygons(ids, ring_counts, poly_ring_counts, coords))
        completed += 1
        if completed % 25 == 0 or completed == len(pending):
            logger.info(
                "Geometry extraction progress: %d/%d batches (%.1f%%), elapsed=%.2fs",
                completed,
                len(pending),
                100.0 * completed / max(1, len(pending)),
                time.perf_counter() - t_poly,
            )

    if pending:
        payloads = [
            (batch_index, entries, simplify, tolerance) for batch_index, entries in pending
        ]
        if workers <= 1 or len(payloads) == 1:
            global _WORKER_LABELS
            _WORKER_LABELS = labels
            try:
                for payload in payloads:
                    _handle(_polygonize_batch(payload))
            finally:
                _WORKER_LABELS = None
        else:
            spec, shm = create_shared_array(labels)
            # Workers read the shared copy; drop the parent's private one so
            # peak RSS stays at one label array rather than two.
            del labels
            try:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_geometry_worker,
                    initargs=(spec,),
                ) as executor:
                    futures = [executor.submit(_polygonize_batch, p) for p in payloads]
                    for future in as_completed(futures):
                        _handle(future.result())
            finally:
                shm.close()
                shm.unlink()

    if store is not None:
        geoms = store.load()

    logger.info(
        "Extracted %d/%d label geometries in %.2fs",
        len(geoms),
        len(bboxes),
        time.perf_counter() - t_start,
    )
    return geoms


def boundaries_to_geometries(
    boundaries: gpd.GeoDataFrame,
    simplify: bool = True,
    tolerance: float = 0.5,
) -> dict[int, Polygon]:
    """Extract and optionally simplify polygon geometries from a boundaries GeoDataFrame.

    Preferred over :func:`extract_label_geometries` when boundaries are already
    available as a ``GeoDataFrame`` (e.g. loaded from a sopa parquet file),
    because it avoids the rasterize → chunk-scan → contour cycle.

    Args:
        boundaries: GeoDataFrame with Polygon/MultiPolygon geometry and a
            1-based integer index (as produced by
            :func:`~cellmeasurement.io.mask_io.load_mask`).
        simplify: Apply Douglas-Peucker simplification.
        tolerance: Simplification tolerance in pixels.

    Returns:
        Mapping of label ID (GeoDataFrame index value) to Shapely Polygon.
        Entries with empty or invalid geometry are omitted.
    """
    geoms: dict[int, Polygon] = {}
    for label_id, geom in boundaries.geometry.items():
        if geom is None or geom.is_empty:
            continue
        # Keep only the largest polygon for MultiPolygon (QuPath convention).
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda p: p.area)
        if geom.geom_type != "Polygon":
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        if simplify and tolerance > 0:
            geom = geom.simplify(tolerance, preserve_topology=True)
        if isinstance(geom, Polygon) and not geom.is_empty:
            geoms[cast(int, label_id)] = geom
    logger.debug("Extracted %d geometries from boundaries GeoDataFrame", len(geoms))
    return geoms
