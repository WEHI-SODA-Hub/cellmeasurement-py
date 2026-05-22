from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..segmentation.cell import CellMatch

_COMPARTMENT_LABEL = {
    "CELL": "Cell",
    "NUCLEUS": "Nucleus",
    "CYTOPLASM": "Cytoplasm",
    "MEMBRANE": "Membrane",
}


@dataclass(frozen=True)
class TileLabelIndexCache:
    """Tile-local reusable label-to-flat-index lookup tables."""

    wc_indices_by_label: Mapping[int, np.ndarray]
    nuc_indices_by_label: Mapping[int, np.ndarray]


def _build_index_map(mask_2d: np.ndarray | None, labels: set[int]) -> dict[int, np.ndarray]:
    if mask_2d is None or not labels:
        return {}
    mask_flat = np.asarray(mask_2d).ravel()
    label_arr = np.array(sorted(labels), dtype=mask_flat.dtype)
    hit_mask = np.isin(mask_flat, label_arr)
    hit_idx = np.flatnonzero(hit_mask)
    if hit_idx.size == 0:
        return {}

    hit_labels = mask_flat[hit_idx]
    order = np.argsort(hit_labels, kind="stable")
    sorted_idx = hit_idx[order]
    sorted_labels = hit_labels[order]

    split_points = np.flatnonzero(np.diff(sorted_labels)) + 1
    idx_groups = np.split(sorted_idx, split_points)
    label_groups = np.split(sorted_labels, split_points)
    out: dict[int, np.ndarray] = {}
    for label_values, idx_values in zip(label_groups, idx_groups):
        if label_values.size == 0:
            continue
        out[int(label_values[0])] = np.asarray(idx_values, dtype=np.int64)
    return out


def build_tile_label_index_cache(
    cells: Sequence[CellMatch],
    wc_tile: np.ndarray | None,
    nuc_tile: np.ndarray | None,
) -> TileLabelIndexCache:
    """Build tile-level label membership caches used by intensity families."""
    wc_labels = {int(cell.whole_cell_label) for cell in cells if cell.whole_cell_label is not None}
    nuc_labels = {int(cell.nucleus_label) for cell in cells if cell.nucleus_label is not None}
    return TileLabelIndexCache(
        wc_indices_by_label=_build_index_map(wc_tile, wc_labels),
        nuc_indices_by_label=_build_index_map(nuc_tile, nuc_labels),
    )


def indexed_compartment_indices(
    cell: CellMatch,
    *,
    cache: TileLabelIndexCache,
) -> dict[str, np.ndarray] | None:
    """Return tile-local flat indices for CELL/NUCLEUS/CYTOPLASM when available."""
    if cell.match_source == "watershed_synth":
        return None

    cell_idx: np.ndarray | None = None
    if cell.whole_cell_label is not None:
        cell_idx = cache.wc_indices_by_label.get(int(cell.whole_cell_label))
    if cell_idx is None and cell.nucleus_label is not None:
        cell_idx = cache.nuc_indices_by_label.get(int(cell.nucleus_label))
    if cell_idx is None or cell_idx.size == 0:
        return None

    nuc_idx = np.empty(0, dtype=np.int64)
    if cell.nucleus_label is not None:
        raw_nuc_idx = cache.nuc_indices_by_label.get(int(cell.nucleus_label))
        if raw_nuc_idx is not None and raw_nuc_idx.size > 0:
            nuc_idx = np.intersect1d(cell_idx, raw_nuc_idx, assume_unique=False)
    cyto_idx = np.setdiff1d(cell_idx, nuc_idx, assume_unique=False)
    return {"CELL": cell_idx, "NUCLEUS": nuc_idx, "CYTOPLASM": cyto_idx}


def _summary_stats(vals: np.ndarray, *, include_median: bool) -> dict[str, float]:
    if vals.size == 0:
        return {}
    out = {
        "Mean": float(np.mean(vals)),
        "Min": float(np.min(vals)),
        "Max": float(np.max(vals)),
        "Std.Dev.": float(np.std(vals)),
    }
    if include_median:
        out["Median"] = float(np.median(vals))
    return out


def add_indexed_summary_stats(
    props: dict[str, float],
    image_cyx: np.ndarray,
    ch_names: Sequence[str],
    compartment_indices: Mapping[str, np.ndarray],
    *,
    include_median: bool,
) -> None:
    """Populate baseline summary stats from precomputed flat index arrays."""
    if not compartment_indices:
        return
    channel_flat = [np.asarray(image_cyx[ci]).ravel() for ci in range(len(ch_names))]
    for ci, ch in enumerate(ch_names):
        flat = channel_flat[ci]
        for comp, idx in compartment_indices.items():
            if idx.size == 0:
                continue
            vals = flat[idx]
            for stat_key, value in _summary_stats(vals, include_median=include_median).items():
                props[f"{ch}: {_COMPARTMENT_LABEL[comp]}: {stat_key}"] = value


def add_indexed_percentiles(
    props: dict[str, float],
    image_cyx: np.ndarray,
    ch_names: Sequence[str],
    compartment_indices: Mapping[str, np.ndarray],
    percentiles: Sequence[float],
) -> None:
    """Populate percentile stats from precomputed flat index arrays."""
    if not percentiles or not compartment_indices:
        return
    channel_flat = [np.asarray(image_cyx[ci]).ravel() for ci in range(len(ch_names))]
    for ci, ch in enumerate(ch_names):
        flat = channel_flat[ci]
        for comp, idx in compartment_indices.items():
            if idx.size == 0:
                continue
            vals = flat[idx]
            for p in percentiles:
                props[f"{ch}: {_COMPARTMENT_LABEL[comp]}: Percentile: {p}"] = float(np.percentile(vals, p))


def add_masked_summary_stats(
    props: dict[str, float],
    image_cyx: np.ndarray,
    ch_names: Sequence[str],
    comp_masks: Mapping[str, np.ndarray],
    *,
    compartments: Sequence[str],
    include_median: bool,
) -> None:
    """Populate baseline summary stats for selected compartments from masks."""
    for ci, ch in enumerate(ch_names):
        ch_img = image_cyx[ci]
        for comp in compartments:
            mask = comp_masks[comp]
            vals = ch_img[mask]
            if vals.size == 0:
                continue
            for stat_key, value in _summary_stats(vals, include_median=include_median).items():
                props[f"{ch}: {_COMPARTMENT_LABEL[comp]}: {stat_key}"] = value


def add_masked_percentiles(
    props: dict[str, float],
    image_cyx: np.ndarray,
    ch_names: Sequence[str],
    comp_masks: Mapping[str, np.ndarray],
    percentiles: Sequence[float],
    *,
    compartments: Sequence[str],
) -> None:
    """Populate percentile stats for selected compartments from masks."""
    if not percentiles:
        return
    for ci, ch in enumerate(ch_names):
        ch_img = image_cyx[ci]
        for comp in compartments:
            mask = comp_masks[comp]
            vals = ch_img[mask]
            if vals.size == 0:
                continue
            for p in percentiles:
                props[f"{ch}: {_COMPARTMENT_LABEL[comp]}: Percentile: {p}"] = float(np.percentile(vals, p))
