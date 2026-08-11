"""Cell overlap-constraint geometry pass for GeoJSON features.

Implements QuPath-style overlap clipping:
- when two cells overlap, trim the larger by the smaller;
- keep only the largest polygon fragment after boolean operations;
- drop cells whose geometry becomes empty;
- clip nucleusGeometry to the final cell geometry (remove if empty).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import shapely
from shapely.geometry import Polygon, mapping, shape

log = logging.getLogger(__name__)

__all__ = ["constrain_cell_overlaps"]

# Wall-clock gap between trim-loop progress lines.
_PROGRESS_SECONDS = 60.0

# Iterations between clock reads, so the timer costs nothing in the hot loop.
_PROGRESS_STRIDE = 1024


def _ensure_largest_polygon(geom: Any) -> Polygon:
    """Return a single valid Polygon, keeping the largest polygonal piece."""
    if geom is None or geom.is_empty:
        return Polygon()

    if not geom.is_valid:
        geom = geom.buffer(0)

    if geom.geom_type == "Polygon":
        return geom

    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)

    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type == "Polygon" and not g.is_empty and g.area > 0]
        if not polys:
            return Polygon()
        return max(polys, key=lambda g: g.area)

    return Polygon()


def _geometry_array(geoms: list[Any]) -> np.ndarray:
    """Wrap geometries in an object array for shapely's vectorised functions."""
    arr = np.empty(len(geoms), dtype=object)
    arr[:] = geoms
    return arr


def _candidate_pairs(geoms: list[Any]) -> np.ndarray:
    """Return ``(i, j)`` index pairs, ``i < j``, whose geometries intersect.

    An STRtree resolves both the bounding-box filter and the ``intersects``
    predicate in C against prepared geometries. The area test stays in the trim
    loop: geometries are edited in place as the pass runs, so overlap has to be
    re-checked against the current geometry rather than settled up front.
    """
    empty = np.empty((0, 2), dtype=np.intp)
    if len(geoms) < 2:
        return empty

    t_start = time.perf_counter()
    geom_arr = _geometry_array(geoms)
    left, right = shapely.STRtree(geom_arr).query(geom_arr, predicate="intersects")

    # query() reports self-hits and both directions of every pair; keep one
    # ordered copy of each distinct pair.
    upper = left < right
    left, right = left[upper], right[upper]
    if left.size == 0:
        return empty

    order = np.lexsort((right, left))  # stable trim order across runs
    log.info(
        "Overlap constraint broad phase complete in %.2fs: %d intersecting pairs",
        time.perf_counter() - t_start,
        left.size,
    )
    return np.column_stack((left[order], right[order]))


def _has_meaningful_overlap(geom_a: Any, geom_b: Any) -> bool:
    """Return True when intersection area is non-trivial and geometry ops succeed."""
    if geom_a.is_empty or geom_b.is_empty:
        return False
    try:
        if not geom_a.intersects(geom_b):
            return False
        intersection = geom_a.intersection(geom_b)
        return not intersection.is_empty and intersection.area >= 1e-10
    except Exception:
        return False


def _trim_larger_cell(i: int, j: int, geoms: list[Any], areas: list[float]) -> bool:
    """Trim the larger of two overlapping cells and update geometry/area in place."""
    gi = geoms[i]
    gj = geoms[j]
    if not _has_meaningful_overlap(gi, gj):
        return False

    # Trim larger cell; if equal area, lower index i is trimmed.
    if areas[i] >= areas[j]:
        gi = _ensure_largest_polygon(gi.difference(gj))
        geoms[i] = gi
        areas[i] = gi.area if not gi.is_empty else 0.0
    else:
        gj = _ensure_largest_polygon(gj.difference(gi))
        geoms[j] = gj
        areas[j] = gj.area if not gj.is_empty else 0.0
    return True


def _clip_nucleus_geometry(feature: dict[str, Any], cell_geom: Polygon) -> None:
    """Clip nucleusGeometry to the final cell polygon or remove it when empty."""
    if "nucleusGeometry" not in feature:
        return
    try:
        ng = shape(feature["nucleusGeometry"]).intersection(cell_geom)
        ng = _ensure_largest_polygon(ng)
        if ng.is_empty:
            del feature["nucleusGeometry"]
        else:
            feature["nucleusGeometry"] = mapping(ng)
    except Exception:
        del feature["nucleusGeometry"]


def _finalize_features(features: list[dict[str, Any]], geoms: list[Any]) -> list[dict[str, Any]]:
    """Build final output features from clipped cell geometries."""
    out: list[dict[str, Any]] = []
    for f, g in zip(features, geoms):
        cell_geom = _ensure_largest_polygon(g)
        if cell_geom.is_empty:
            continue

        updated = dict(f)
        updated["geometry"] = mapping(cell_geom)
        _clip_nucleus_geometry(updated, cell_geom)
        out.append(updated)
    return out


def constrain_cell_overlaps(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clip overlapping cell geometries so output cells do not share area."""
    t_start = time.perf_counter()
    if not features:
        return features

    n = len(features)
    geoms = [shape(f["geometry"]) for f in features]
    areas = [g.area for g in geoms]

    pairs = _candidate_pairs(geoms)
    total = len(pairs)

    # Geometries are trimmed in place, so a pair can stop overlapping before its
    # turn comes round; _trim_larger_cell re-checks rather than trusting the
    # broad phase.
    clipped = 0
    next_report = time.perf_counter() + _PROGRESS_SECONDS
    for k in range(total):
        i, j = pairs[k]
        if _trim_larger_cell(int(i), int(j), geoms, areas):
            clipped += 1
        if k % _PROGRESS_STRIDE == 0:
            now = time.perf_counter()
            if now >= next_report:
                log.info(
                    "Overlap constraint progress: %d/%d pairs (%.1f%%), elapsed=%.2fs, clipped=%d",
                    k + 1,
                    total,
                    100.0 * (k + 1) / total,
                    now - t_start,
                    clipped,
                )
                next_report = now + _PROGRESS_SECONDS

    out = _finalize_features(features, geoms)

    log.info(
        "Overlap constraint complete in %.2fs: checked %d pairs, clipped %d, removed %d empty cells",
        time.perf_counter() - t_start,
        total,
        clipped,
        n - len(out),
    )
    return out
