"""Overlap-first, chunk-parallel nuclear-to-whole-cell ROI matcher.

Algorithm overview
------------------
1. **Normalise** — ensure both label arrays are 2-D dask arrays aligned to the
   same chunk grid (rechunk the whole-cell array to match the nuclear array).
2. **Chunk-parallel overlap counting** — for each chunk pair, accumulate
   ``(nuc_id, wc_id) -> pixel_count`` overlaps and per-label spatial stats
   using a single ``np.nonzero`` pass (O(non-zero pixels), not O(labels *
   pixels)).
3. **Reduce** — merge chunk results into global overlap counts and label stats.
4. **Greedy 1-to-1 assignment** — sort candidates by
   ``(-overlap, nuc_id, wc_id)`` for determinism and assign each nucleus and
   whole-cell label at most once.  Note: greedy matching is optimal when
   overlap relationships are non-conflicting (the common case for sopa outputs)
   but may give suboptimal results when many nuclei compete for the same
   whole-cell region.
5. **Watershed synthesis** — unmatched nuclei are expanded into unclaimed
   territory via constrained watershed with a configurable growth radius.

Single-mask modes (nuclear-only or whole-cell-only) bypass the matching
pipeline entirely and produce ``CellMatch`` objects with the appropriate
``match_source`` tag.
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

import dask
import dask.array as da
import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from skimage.morphology import disk
from skimage.segmentation import watershed

from ..geometry import mask_to_geometry
from .cell import CellMatch
from .types import (
    CellId,
    ChunkResult,
    LabelId,
    LabelStats,
    LabelStatsById,
    MatchSource,
    OverlapCounts,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

__all__ = ["match_rois"]


def match_rois(
    nuc_labels: da.Array | None,
    wc_labels: da.Array | None,
    dist_threshold: float = 10.0,
    estimate_cell_boundary_dist: float = 3.0,
    downsample_factor: float = 1.0,
) -> tuple[list[CellMatch], dict[CellId, Polygon]]:
    """Match nuclear ROIs to whole-cell ROIs and return a list of cells.

    Depending on which arrays are provided the function operates in one of
    three modes:

    * **Paired mode** (both arrays provided): primary overlap-based 1-to-1
      matching followed by watershed synthesis for unmatched nuclei.
    * **Whole-cell-only mode** (``nuc_labels`` is ``None``): each whole-cell
      label becomes a separate cell with ``match_source="wc_only"``.
    * **Nuclear-only mode** (``wc_labels`` is ``None``): each nucleus becomes
      a cell with ``match_source="nuc_only"``.

    Args:
        nuc_labels: 2-D dask label array for the nuclear segmentation, or
            ``None`` for whole-cell-only mode.
        wc_labels: 2-D dask label array for the whole-cell segmentation, or
            ``None`` for nuclear-only mode.
        dist_threshold: Maximum centroid distance (pixels) used for secondary
            nearest-neighbour matching after overlap assignment. Interpreted
            in downsampled coordinate space if downsample_factor > 1.
        estimate_cell_boundary_dist: Radius in pixels for watershed expansion of unmatched
            nuclei.  Ignored in single-mask modes. Interpreted in downsampled
            coordinate space if downsample_factor > 1.
        downsample_factor: Optional downsampling factor (e.g. 2.0, 4.0) to reduce
            memory usage before matching. Values <= 1.0 or that round to step < 2
            result in no downsampling.

    Returns:
        A two-tuple of:

        * List of :class:`~cellmeasurement.segmentation.cell.CellMatch` objects,
          one per output cell, sorted by ``cell_id``.
        * Dict mapping ``cell_id`` to Shapely :class:`~shapely.geometry.Polygon`
          for every ``watershed_synth`` cell.

    Raises:
        ValueError: If both arrays are ``None``.
    """
    if nuc_labels is None and wc_labels is None:
        raise ValueError("At least one of nuc_labels or wc_labels must be provided.")
    if dist_threshold <= 0:
        raise ValueError("dist_threshold must be > 0")
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be > 0")

    # Handle single-mask modes before attempting downsampling (which requires both masks)
    if nuc_labels is None:
        assert wc_labels is not None
        return _wc_only_matches(wc_labels), {}
    if wc_labels is None:
        return _nuc_only_matches(nuc_labels), {}

    # Type narrowers: guarantee both are non-None past this point for paired mode
    assert nuc_labels is not None
    assert wc_labels is not None

    # Apply optional downsampling to both label arrays (paired mode only)
    if downsample_factor > 1.0:
        from ..io.mask_io import maybe_downsample
        import numpy as np_for_downsample

        step = int(round(downsample_factor))
        if step >= 2:
            # Convert dask arrays to numpy for downsampling (since they're typically not huge)
            nuc_np = np_for_downsample.asarray(nuc_labels)
            wc_np = np_for_downsample.asarray(wc_labels)

            # Create a dummy image array for maybe_downsample compatibility
            # (we only care about the mask downsampling part)
            dummy_img = np_for_downsample.zeros((1, wc_np.shape[0], wc_np.shape[1]), dtype=np_for_downsample.float32)

            _, nuc_ds, wc_ds = maybe_downsample(dummy_img, nuc_np, wc_np, downsample_factor)

            # Convert back to dask arrays with sensible chunks
            nuc_labels = da.from_array(nuc_ds, chunks="auto")
            wc_labels = da.from_array(wc_ds, chunks="auto")
            logger.info("Applied downsampling factor %.1f (step=%d) to label arrays", downsample_factor, step)

    # Paired mode ------------------------------------------------------------
    H, W = int(nuc_labels.shape[0]), int(nuc_labels.shape[1])

    # Align chunk grids so we can pair chunks by index.
    wc_labels = wc_labels.rechunk(nuc_labels.chunks)

    # --- Chunk-parallel overlap counting + per-label stats ---
    n_row = len(nuc_labels.chunks[0])
    n_col = len(nuc_labels.chunks[1])
    row_offs = [0] + [int(x) for x in np.cumsum(nuc_labels.chunks[0])[:-1]]
    col_offs = [0] + [int(x) for x in np.cumsum(nuc_labels.chunks[1])[:-1]]

    nuc_del = nuc_labels.to_delayed()
    wc_del = wc_labels.to_delayed()

    delayed_results = [
        dask.delayed(_count_overlaps_chunk)(
            nuc_del[i, j], wc_del[i, j], row_offs[i], col_offs[j]
        )
        for i in range(n_row)
        for j in range(n_col)
    ]

    logger.info("Computing overlaps across %d chunks...", len(delayed_results))
    chunk_results = dask.compute(*delayed_results)

    overlap_counts = _merge_overlap_counts([r[0] for r in chunk_results])
    nuc_stats = _merge_stats([r[1] for r in chunk_results])
    wc_stats = _merge_stats([r[2] for r in chunk_results])

    logger.info(
        "Labels: %d nuclei, %d whole-cells, %d overlap pairs",
        len(nuc_stats),
        len(wc_stats),
        len(overlap_counts),
    )

    # --- Greedy 1-to-1 assignment ---
    matches, matched_nuc, matched_wc, next_id = _resolve_one_to_one(
        overlap_counts, nuc_stats, wc_stats, next_id=1
    )
    logger.info(
        "Matched %d pairs; %d nuclei unmatched, %d whole-cells unmatched",
        len(matches),
        len(nuc_stats) - len(matched_nuc),
        len(wc_stats) - len(matched_wc),
    )

    dist_matches, matched_nuc, matched_wc, next_id = _resolve_distance_threshold(
        nuc_stats=nuc_stats,
        wc_stats=wc_stats,
        matched_nuc=matched_nuc,
        matched_wc=matched_wc,
        next_id=next_id,
        dist_threshold=dist_threshold,
    )
    if dist_matches:
        matches.extend(dist_matches)
        logger.info(
            "Distance-threshold matched %d additional pairs (<= %.2f px)",
            len(dist_matches),
            dist_threshold,
        )

    # --- Synthesis for unmatched nuclei ---
    unmatched_nuc = set(nuc_stats.keys()) - matched_nuc
    synth, synth_geoms, _, dropped_synth_cells = _synthesis_pass(
        unmatched_nuc,
        matched_wc,
        nuc_labels,
        wc_labels,
        nuc_stats,
        estimate_cell_boundary_dist,
        H,
        W,
        next_id,
    )
    logger.info(
        "Synthesised boundaries for %d unmatched nuclei (dropped=%d)",
        len(synth),
        dropped_synth_cells,
    )

    return matches + synth, synth_geoms


# ---------------------------------------------------------------------------
# Single-mask modes
# ---------------------------------------------------------------------------


def _wc_only_matches(wc_labels: da.Array) -> list[CellMatch]:
    stats = _collect_label_stats(wc_labels)
    return [
        _stats_to_cell(cell_id, None, wc_id, stats[wc_id], stats[wc_id], 0, 0.0, "wc_only")
        for cell_id, wc_id in enumerate(sorted(stats), start=1)
    ]


def _nuc_only_matches(nuc_labels: da.Array) -> list[CellMatch]:
    stats = _collect_label_stats(nuc_labels)
    return [
        _stats_to_cell(cell_id, nuc_id, None, stats[nuc_id], stats[nuc_id], 0, 0.0, "nuc_only")
        for cell_id, nuc_id in enumerate(sorted(stats), start=1)
    ]


def _stats_to_cell(
    cell_id: int,
    nuc_label: LabelId | None,
    wc_label: LabelId | None,
    nuc_s: LabelStats,
    cell_s: LabelStats,
    overlap_px: int,
    overlap_frac: float,
    source: MatchSource,
) -> CellMatch:
    """Build a CellMatch from pre-merged stats dicts."""
    row_min = min(nuc_s["row_min"], cell_s["row_min"])
    col_min = min(nuc_s["col_min"], cell_s["col_min"])
    row_max_excl = max(nuc_s["row_max"], cell_s["row_max"]) + 1
    col_max_excl = max(nuc_s["col_max"], cell_s["col_max"]) + 1
    centroid = (cell_s["row_sum"] / cell_s["area"], cell_s["col_sum"] / cell_s["area"])
    return CellMatch(
        cell_id=cell_id,
        nucleus_label=nuc_label,
        whole_cell_label=wc_label,
        bbox=(row_min, col_min, row_max_excl, col_max_excl),
        centroid=centroid,
        nucleus_area_px=nuc_s["area"],
        cell_area_px=cell_s["area"],
        overlap_px=overlap_px,
        overlap_fraction=overlap_frac,
        match_source=source,
    )


# ---------------------------------------------------------------------------
# Chunk-level computation helpers (called via dask.delayed)
# ---------------------------------------------------------------------------


def _label_stats_chunk(
    arr: npt.NDArray[np.int_], row_off: int, col_off: int
) -> LabelStatsById:
    """Compute per-label spatial statistics for one label chunk.

    Uses a single ``np.nonzero`` pass with vectorised ``np.minimum.at`` /
    ``np.maximum.at`` / ``np.add.at`` accumulation — O(non-zero pixels).
    """
    stats: LabelStatsById = {}
    rr, cc = np.nonzero(arr)
    if rr.size == 0:
        return stats

    labels = arr[rr, cc]
    glo_rr = rr.astype(np.int64) + row_off
    glo_cc = cc.astype(np.int64) + col_off

    unique_labels, inverse, label_counts = np.unique(labels, return_inverse=True, return_counts=True)
    n = len(unique_labels)

    row_min = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    row_max = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
    col_min = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    col_max = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
    row_sum = np.zeros(n, dtype=np.float64)
    col_sum = np.zeros(n, dtype=np.float64)

    np.minimum.at(row_min, inverse, glo_rr)
    np.maximum.at(row_max, inverse, glo_rr)
    np.minimum.at(col_min, inverse, glo_cc)
    np.maximum.at(col_max, inverse, glo_cc)
    np.add.at(row_sum, inverse, glo_rr.astype(np.float64))
    np.add.at(col_sum, inverse, glo_cc.astype(np.float64))

    for i, lab in enumerate(unique_labels):
        stats[int(lab)] = {
            "area": int(label_counts[i]),
            "row_min": int(row_min[i]),
            "row_max": int(row_max[i]),
            "col_min": int(col_min[i]),
            "col_max": int(col_max[i]),
            "row_sum": float(row_sum[i]),
            "col_sum": float(col_sum[i]),
        }
    return stats


def _count_overlaps_chunk(
    nuc_arr: npt.NDArray[np.int_],
    wc_arr: npt.NDArray[np.int_],
    row_off: int,
    col_off: int,
) -> ChunkResult:
    """Count pixel overlaps and collect stats for one aligned chunk pair.

    Returns:
        A three-tuple of:
        * ``overlap_counts`` — ``{(nuc_id, wc_id): pixel_count}``
        * ``nuc_stats`` — per-label stats for the nuclear array
        * ``wc_stats`` — per-label stats for the whole-cell array
    """
    overlap_counts: OverlapCounts = {}

    nz = (nuc_arr > 0) & (wc_arr > 0)
    if nz.any():
        nuc_nz = nuc_arr.ravel()[nz.ravel()]
        wc_nz = wc_arr.ravel()[nz.ravel()]
        pairs = np.stack([nuc_nz, wc_nz], axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        overlap_counts = {
            (int(p[0]), int(p[1])): int(c) for p, c in zip(unique_pairs, counts)
        }

    return (
        overlap_counts,
        _label_stats_chunk(nuc_arr, row_off, col_off),
        _label_stats_chunk(wc_arr, row_off, col_off),
    )


# ---------------------------------------------------------------------------
# Reduction helpers
# ---------------------------------------------------------------------------


def _merge_overlap_counts(parts: list[OverlapCounts]) -> OverlapCounts:
    merged: OverlapCounts = {}
    for part in parts:
        for pair, count in part.items():
            merged[pair] = merged.get(pair, 0) + count
    return merged


def _merge_stats(parts: list[LabelStatsById]) -> LabelStatsById:
    merged: LabelStatsById = {}
    for part in parts:
        for label_id, s in part.items():
            if label_id not in merged:
                merged[label_id] = {
                    "area": s["area"],
                    "row_min": s["row_min"],
                    "row_max": s["row_max"],
                    "col_min": s["col_min"],
                    "col_max": s["col_max"],
                    "row_sum": s["row_sum"],
                    "col_sum": s["col_sum"],
                }
            else:
                m = merged[label_id]
                m["area"] += s["area"]
                m["row_min"] = min(m["row_min"], s["row_min"])
                m["row_max"] = max(m["row_max"], s["row_max"])
                m["col_min"] = min(m["col_min"], s["col_min"])
                m["col_max"] = max(m["col_max"], s["col_max"])
                m["row_sum"] += s["row_sum"]
                m["col_sum"] += s["col_sum"]
    return merged


def _collect_label_stats(labels: da.Array) -> LabelStatsById:
    """Chunk-parallel per-label stats for a single label array."""
    n_row = len(labels.chunks[0])
    n_col = len(labels.chunks[1])
    row_offs = [0] + [int(x) for x in np.cumsum(labels.chunks[0])[:-1]]
    col_offs = [0] + [int(x) for x in np.cumsum(labels.chunks[1])[:-1]]
    delayed_arr = labels.to_delayed()
    delayed_results = [
        dask.delayed(_label_stats_chunk)(delayed_arr[i, j], row_offs[i], col_offs[j])
        for i in range(n_row)
        for j in range(n_col)
    ]
    return _merge_stats(list(dask.compute(*delayed_results)))


# ---------------------------------------------------------------------------
# 1-to-1 greedy assignment
# ---------------------------------------------------------------------------


def _resolve_one_to_one(
    overlap_counts: OverlapCounts,
    nuc_stats: LabelStatsById,
    wc_stats: LabelStatsById,
    next_id: int,
) -> tuple[list[CellMatch], set[LabelId], set[LabelId], int]:
    """Greedy 1-to-1 assignment by descending overlap.

    Candidates are sorted by ``(-overlap_px, nuc_id, wc_id)`` so that
    tie-breaking is deterministic regardless of dict insertion order.

    Note: greedy assignment is optimal when each nucleus overlaps at most one
    whole-cell region (common for well-separated sopa outputs).  When multiple
    nuclei compete for a single whole-cell label, the one with the highest
    pixel overlap wins; remaining competing nuclei fall through to watershed
    synthesis.

    Returns:
        * List of matched :class:`CellMatch` objects.
        * Set of matched nucleus label IDs.
        * Set of matched whole-cell label IDs.
        * Updated ``next_id`` counter.
    """
    candidates = sorted(
        [(n, w, c) for (n, w), c in overlap_counts.items()],
        key=lambda x: (-x[2], x[0], x[1]),
    )

    matched_nuc: set[LabelId] = set()
    matched_wc: set[LabelId] = set()
    matches: list[CellMatch] = []

    for nuc_id, wc_id, overlap_px in candidates:
        if nuc_id in matched_nuc or wc_id in matched_wc:
            continue

        matched_nuc.add(nuc_id)
        matched_wc.add(wc_id)

        ns = nuc_stats[nuc_id]
        ws = wc_stats[wc_id]
        frac = overlap_px / ns["area"] if ns["area"] > 0 else 0.0

        row_min = min(ns["row_min"], ws["row_min"])
        col_min = min(ns["col_min"], ws["col_min"])
        row_max_excl = max(ns["row_max"], ws["row_max"]) + 1
        col_max_excl = max(ns["col_max"], ws["col_max"]) + 1
        centroid = (ws["row_sum"] / ws["area"], ws["col_sum"] / ws["area"])

        matches.append(
            CellMatch(
                cell_id=next_id,
                nucleus_label=nuc_id,
                whole_cell_label=wc_id,
                bbox=(row_min, col_min, row_max_excl, col_max_excl),
                centroid=centroid,
                nucleus_area_px=ns["area"],
                cell_area_px=ws["area"],
                overlap_px=overlap_px,
                overlap_fraction=frac,
                match_source="overlap_1to1",
            )
        )
        next_id += 1

    return matches, matched_nuc, matched_wc, next_id


def _centroid_from_stats(stats: LabelStats) -> tuple[float, float]:
    """Compute centroid (row, col) from aggregated label stats."""
    return (stats["row_sum"] / stats["area"], stats["col_sum"] / stats["area"])


def _resolve_distance_threshold(
    nuc_stats: LabelStatsById,
    wc_stats: LabelStatsById,
    matched_nuc: set[LabelId],
    matched_wc: set[LabelId],
    next_id: int,
    dist_threshold: float,
) -> tuple[list[CellMatch], set[LabelId], set[LabelId], int]:
    """Match unmatched nuclei to nearest unmatched whole-cell centroids within threshold."""
    unmatched_nuc = sorted(set(nuc_stats.keys()) - matched_nuc)
    candidate_wc = sorted(set(wc_stats.keys()) - matched_wc)
    if not unmatched_nuc or not candidate_wc:
        return [], matched_nuc, matched_wc, next_id

    wc_points = np.array([_centroid_from_stats(wc_stats[wid]) for wid in candidate_wc], dtype=np.float64)
    tree = cKDTree(wc_points)

    matches: list[CellMatch] = []
    for nuc_id in unmatched_nuc:
        ns = nuc_stats[nuc_id]
        nuc_point = np.array(_centroid_from_stats(ns), dtype=np.float64)
        idxs = tree.query_ball_point(nuc_point, r=dist_threshold)
        if not idxs:
            continue

        best_wc: LabelId | None = None
        best_dist_sq = float("inf")
        for idx in idxs:
            wc_id = candidate_wc[int(idx)]
            if wc_id in matched_wc:
                continue
            ws = wc_stats[wc_id]
            wr, wc = _centroid_from_stats(ws)
            dr = nuc_point[0] - wr
            dc = nuc_point[1] - wc
            dist_sq = float(dr * dr + dc * dc)
            if dist_sq < best_dist_sq or (dist_sq == best_dist_sq and (best_wc is None or wc_id < best_wc)):
                best_dist_sq = dist_sq
                best_wc = wc_id

        if best_wc is None:
            continue

        ws = wc_stats[best_wc]
        matched_nuc.add(nuc_id)
        matched_wc.add(best_wc)

        row_min = min(ns["row_min"], ws["row_min"])
        col_min = min(ns["col_min"], ws["col_min"])
        row_max_excl = max(ns["row_max"], ws["row_max"]) + 1
        col_max_excl = max(ns["col_max"], ws["col_max"]) + 1
        centroid = _centroid_from_stats(ws)

        matches.append(
            CellMatch(
                cell_id=next_id,
                nucleus_label=nuc_id,
                whole_cell_label=best_wc,
                bbox=(row_min, col_min, row_max_excl, col_max_excl),
                centroid=centroid,
                nucleus_area_px=ns["area"],
                cell_area_px=ws["area"],
                overlap_px=0,
                overlap_fraction=0.0,
                match_source="overlap_1to1",
            )
        )
        next_id += 1

    return matches, matched_nuc, matched_wc, next_id


# ---------------------------------------------------------------------------
# Watershed synthesis for unmatched nuclei
# ---------------------------------------------------------------------------


def _synthesis_pass(
    unmatched_nuc_ids: set[LabelId],
    matched_wc_ids: set[LabelId],
    nuc_labels: da.Array,
    wc_labels: da.Array,
    nuc_stats: LabelStatsById,
    estimate_cell_boundary_dist: float,
    H: int,
    W: int,
    next_id: int,
) -> tuple[list[CellMatch], dict[CellId, Polygon], int, int]:
    """Synthesise whole-cell boundaries for unmatched nuclei via watershed.

    Nuclei are expanded into pixels not already claimed by a matched whole-cell
    label, up to *estimate_cell_boundary_dist* pixels from the nucleus border.  Polygon
    geometry is extracted immediately per cell and the watershed arrays are
    explicitly freed afterwards to minimise peak RSS.

    Returns:
        * List of synthesised :class:`CellMatch` objects.
        * Dict mapping ``cell_id`` to Shapely Polygon in global image
          coordinates for every synthesised cell.
        * Updated ``next_id`` counter.
        * Number of unmatched nuclei dropped because watershed yielded no pixels.
    """
    if not unmatched_nuc_ids:
        return [], {}, next_id, 0

    pad = max(1, int(np.ceil(estimate_cell_boundary_dist)))
    all_row_min = min(nuc_stats[n]["row_min"] for n in unmatched_nuc_ids)
    all_row_max = max(nuc_stats[n]["row_max"] for n in unmatched_nuc_ids)
    all_col_min = min(nuc_stats[n]["col_min"] for n in unmatched_nuc_ids)
    all_col_max = max(nuc_stats[n]["col_max"] for n in unmatched_nuc_ids)

    r0 = max(0, all_row_min - pad)
    r1 = min(H, all_row_max + pad + 1)
    c0 = max(0, all_col_min - pad)
    c1 = min(W, all_col_max + pad + 1)

    nuc_region: npt.NDArray[np.int_] = nuc_labels[r0:r1, c0:c1].compute()
    wc_region: npt.NDArray[np.int_] = wc_labels[r0:r1, c0:c1].compute()

    # Use local 1..N seed labels so find_objects can index directly by label.
    sorted_nuc_ids = sorted(unmatched_nuc_ids)
    # local_id (1-based) -> (nuc_id, global cell_id)
    local_map: list[tuple[LabelId, CellId]] = [
        (nuc_id, next_id + i) for i, nuc_id in enumerate(sorted_nuc_ids)
    ]
    next_id += len(sorted_nuc_ids)

    seeds = np.zeros_like(nuc_region, dtype=np.int32)
    for local_id, (nuc_id, _) in enumerate(local_map, start=1):
        seeds[nuc_region == nuc_id] = local_id

    # Growth zone: within estimate_cell_boundary_dist of any seed, excluding matched wc pixels.
    growth_zone = ndi.binary_dilation(
        seeds > 0, structure=disk(max(1, int(estimate_cell_boundary_dist)))
    )
    assert growth_zone.dtype == bool
    if matched_wc_ids:
        claimed: npt.NDArray[np.bool_] = np.isin(wc_region, list(matched_wc_ids))
        growth_zone = growth_zone & ~claimed

    # Watershed: EDT drives expansion from seeds into growth_zone.
    dist_map = ndi.distance_transform_edt(seeds == 0)
    ws_result = watershed(dist_map, markers=seeds, mask=growth_zone)

    # Release large intermediate arrays before polygon extraction.
    del dist_map, wc_region
    gc.collect()

    # find_objects returns one slice per label (index = local_id - 1).
    obj_slices = ndi.find_objects(ws_result)

    synth: list[CellMatch] = []
    synth_geoms: dict[CellId, Polygon] = {}
    dropped_synth_cells = 0

    for local_id, (nuc_id, cid) in enumerate(local_map, start=1):
        idx = local_id - 1
        sl = obj_slices[idx] if idx < len(obj_slices) else None

        if sl is not None:
            sub_mask: npt.NDArray[np.bool_] = (ws_result[sl] == local_id) & growth_zone[sl]
            row_off_local: int = sl[0].start
            col_off_local: int = sl[1].start
        else:
            sub_mask = np.zeros((1, 1), dtype=np.bool_)
            row_off_local = 0
            col_off_local = 0

        if not sub_mask.any():
            dropped_synth_cells += 1
            continue

        poly = mask_to_geometry(
            sub_mask,
            simplify=True,
            tolerance=0.5,
            row_offset=r0 + row_off_local,
            col_offset=c0 + col_off_local,
        )
        if poly is None:
            continue

        rows_loc, cols_loc = np.nonzero(sub_mask)
        ns = nuc_stats[nuc_id]
        bbox = (
            r0 + row_off_local + int(rows_loc.min()),
            c0 + col_off_local + int(cols_loc.min()),
            r0 + row_off_local + int(rows_loc.max()) + 1,
            c0 + col_off_local + int(cols_loc.max()) + 1,
        )
        centroid = (
            r0 + row_off_local + float(rows_loc.mean()),
            c0 + col_off_local + float(cols_loc.mean()),
        )

        synth.append(
            CellMatch(
                cell_id=cid,
                nucleus_label=nuc_id,
                whole_cell_label=None,
                bbox=bbox,
                centroid=centroid,
                nucleus_area_px=ns["area"],
                cell_area_px=int(sub_mask.sum()),
                overlap_px=0,
                overlap_fraction=0.0,
                match_source="watershed_synth",
            )
        )
        synth_geoms[cid] = poly

    # Release remaining synthesis workspace.
    del nuc_region, seeds, ws_result, growth_zone
    gc.collect()

    return synth, synth_geoms, next_id, dropped_synth_cells
