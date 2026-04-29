"""Raster-to-polygon geometry utilities.

Pre-extracts simplified polygon geometries from raster label masks so that
downstream steps (measurement, GeoJSON export) work with vectors rather than
re-querying the zarr store per cell.

Typical pipeline position
--------------------------
1. Load masks (``SegmentationMask``)
2. **extract_label_geometries** — chunk-parallel scan → simplified polygons
3. ``match_rois`` — raster-based matching (unchanged)
4. Convert synthesised masks → polygons (where needed)
5. Measure / export using pre-simplified geometry dicts
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import numpy as np
from dask.delayed import delayed
from dask.base import compute
from shapely.geometry import Polygon
from shapely.ops import unary_union
from skimage.measure import find_contours

if TYPE_CHECKING:
    import dask.array as da

__all__ = [
    "extract_label_geometries",
    "mask_to_geometry",
    "boundaries_to_geometries",
]

logger = logging.getLogger(__name__)


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


def _bboxes_chunk(arr: np.ndarray, row_off: int, col_off: int) -> dict[int, tuple[int, int, int, int]]:
    """Return ``{label_id: (row_min, col_min, row_max, col_max)}`` for one chunk."""
    result: dict[int, tuple[int, int, int, int]] = {}
    rr, cc = np.nonzero(arr)
    if rr.size == 0:
        return result

    labels = arr[rr, cc]
    glo_rr = rr.astype(np.int64) + row_off
    glo_cc = cc.astype(np.int64) + col_off

    unique_labels, inverse = np.unique(labels, return_inverse=True)
    n = len(unique_labels)

    row_min = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    row_max = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
    col_min = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    col_max = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)

    np.minimum.at(row_min, inverse, glo_rr)
    np.maximum.at(row_max, inverse, glo_rr)
    np.minimum.at(col_min, inverse, glo_cc)
    np.maximum.at(col_max, inverse, glo_cc)

    for i, lab in enumerate(unique_labels):
        result[int(lab)] = (int(row_min[i]), int(col_min[i]), int(row_max[i]), int(col_max[i]))
    return result


def _merge_bboxes(
    parts: list[dict[int, tuple[int, int, int, int]]],
) -> dict[int, tuple[int, int, int, int]]:
    merged: dict[int, tuple[int, int, int, int]] = {}
    for part in parts:
        for label_id, (r0, c0, r1, c1) in part.items():
            if label_id not in merged:
                merged[label_id] = (r0, c0, r1, c1)
            else:
                mr0, mc0, mr1, mc1 = merged[label_id]
                merged[label_id] = (
                    min(mr0, r0), min(mc0, c0),
                    max(mr1, r1), max(mc1, c1),
                )
    return merged


def _collect_label_bboxes(label_arr: da.Array) -> dict[int, tuple[int, int, int, int]]:
    """Chunk-parallel per-label bounding boxes (row_min, col_min, row_max, col_max)."""
    n_row = len(label_arr.chunks[0])
    n_col = len(label_arr.chunks[1])
    row_offs = [0] + [int(x) for x in np.cumsum(label_arr.chunks[0])[:-1]]
    col_offs = [0] + [int(x) for x in np.cumsum(label_arr.chunks[1])[:-1]]
    delayed_arr = label_arr.to_delayed()  # type: ignore[arg-type]
    delayed_results = [
        delayed(_bboxes_chunk)(delayed_arr[i, j], row_offs[i], col_offs[j])
        for i in range(n_row)
        for j in range(n_col)
    ]
    return _merge_bboxes(list(compute(*delayed_results)))


# ---------------------------------------------------------------------------
# Delayed per-label polygonization
# ---------------------------------------------------------------------------


@delayed
def _polygonize_delayed(
    crop: np.ndarray,
    label_id: int,
    simplify: bool,
    tolerance: float,
    row_off: int,
    col_off: int,
) -> Polygon | None:
    """Materialise a crop array and extract the polygon for one label."""
    return mask_to_geometry(crop == label_id, simplify, tolerance, row_off, col_off)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def extract_label_geometries(
    label_arr: da.Array,
    simplify: bool = True,
    tolerance: float = 0.5,
) -> dict[int, Polygon]:
    """Extract and optionally simplify polygon geometries for all labels.

    Performs a single chunk-parallel scan to collect per-label bounding boxes,
    then submits one delayed task per label that materialises only its bbox crop
    and extracts the polygon. Dask computes these concurrently so individual
    crop arrays are never all resident in memory at the same time.

    Args:
        label_arr: 2-D dask integer label array.
        simplify: Apply Douglas-Peucker simplification.
        tolerance: Simplification tolerance in pixels.

    Returns:
        Mapping of label ID to Shapely Polygon in global image coordinates.
        Labels that produce no valid contour are omitted.
    """
    bboxes = _collect_label_bboxes(label_arr)
    if not bboxes:
        return {}

    H, W = int(label_arr.shape[0]), int(label_arr.shape[1])
    label_ids = sorted(bboxes.keys())

    delayed_geoms = []
    for label_id in label_ids:
        r0, c0, r1, c1 = bboxes[label_id]
        # 1-px padding so contours never coincide with the crop boundary.
        pr0, pc0 = max(0, r0 - 1), max(0, c0 - 1)
        pr1, pc1 = min(H, r1 + 2), min(W, c1 + 2)
        crop = label_arr[pr0:pr1, pc0:pc1]
        delayed_geoms.append(
            _polygonize_delayed(crop, label_id, simplify, tolerance, pr0, pc0)
        )

    computed: tuple[Polygon | None, ...] = compute(*delayed_geoms)

    geoms: dict[int, Polygon] = {}
    for label_id, poly in zip(label_ids, computed):
        if poly is not None:
            geoms[label_id] = poly

    logger.info("Extracted %d/%d label geometries", len(geoms), len(label_ids))
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
            :func:`~cellmeasurement.io.mask_reader.load_mask`).
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
