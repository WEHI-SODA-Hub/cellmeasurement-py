"""Cell overlap-constraint geometry pass for GeoJSON features.

Implements QuPath-style overlap clipping:
- when two cells overlap, trim the larger by the smaller;
- keep only the largest polygon fragment after boolean operations;
- drop cells whose geometry becomes empty;
- clip nucleusGeometry to the final cell geometry (remove if empty).
"""

from __future__ import annotations

import logging
from typing import Any

from shapely.geometry import Polygon, mapping, shape

log = logging.getLogger(__name__)

__all__ = ["constrain_cell_overlaps"]


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


def _build_spatial_grid(geoms: list[Any]) -> dict[tuple[int, int], list[int]]:
    """Build a broad-phase spatial grid keyed by integer cell coordinates."""
    bounds = [g.bounds for g in geoms]
    all_minx = min(b[0] for b in bounds)
    all_miny = min(b[1] for b in bounds)
    all_maxx = max(b[2] for b in bounds)
    all_maxy = max(b[3] for b in bounds)
    span = max(all_maxx - all_minx, all_maxy - all_miny, 1.0)
    grid_size = max(span / max(int(len(geoms) ** 0.5), 1), 1.0)

    grid: dict[tuple[int, int], list[int]] = {}
    for i, (minx, miny, maxx, maxy) in enumerate(bounds):
        gx0 = int((minx - all_minx) / grid_size)
        gy0 = int((miny - all_miny) / grid_size)
        gx1 = int((maxx - all_minx) / grid_size)
        gy1 = int((maxy - all_miny) / grid_size)
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                grid.setdefault((gx, gy), []).append(i)
    return grid


def _pair_candidates(grid: dict[tuple[int, int], list[int]]) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    """Generate unique candidate geometry pairs from shared grid buckets."""
    checked: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    for cell_list in grid.values():
        for ii in range(len(cell_list)):
            i = cell_list[ii]
            for jj in range(ii + 1, len(cell_list)):
                j = cell_list[jj]
                pair = (min(i, j), max(i, j))
                if pair in checked:
                    continue
                checked.add(pair)
                pairs.append(pair)
    return checked, pairs


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
    if not features:
        return features

    n = len(features)
    geoms = [shape(f["geometry"]) for f in features]
    areas = [g.area for g in geoms]

    grid = _build_spatial_grid(geoms)
    checked, pairs = _pair_candidates(grid)
    clipped = 0
    for i, j in pairs:
        if _trim_larger_cell(i, j, geoms, areas):
            clipped += 1

    out = _finalize_features(features, geoms)

    log.info(
        "Overlap constraint: checked %d pairs, clipped %d, removed %d empty cells",
        len(checked),
        clipped,
        n - len(out),
    )
    return out
