from __future__ import annotations

import numpy as np

from cellmeasurement.measurement.compartment_metrics import compartment_masks
from cellmeasurement.measurement.tile_intensity import (
    add_indexed_summary_stats,
    add_masked_summary_stats,
    build_tile_label_index_cache,
    indexed_compartment_indices,
)
from cellmeasurement.segmentation.cell import CellMatch


def _make_cell() -> CellMatch:
    return CellMatch(
        cell_id=1,
        nucleus_label=1,
        whole_cell_label=1,
        bbox=(1, 1, 4, 4),
        centroid=(2.0, 2.0),
        nucleus_area_px=1,
        cell_area_px=9,
        overlap_px=1,
        overlap_fraction=1.0,
        match_source="overlap_1to1",
    )


def test_build_tile_label_index_cache_maps_label_pixels() -> None:
    wc = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint32,
    )
    nuc = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint32,
    )

    cache = build_tile_label_index_cache([_make_cell()], wc_tile=wc, nuc_tile=nuc)
    assert cache.wc_indices_by_label[1].size == 4
    assert cache.nuc_indices_by_label[1].size == 2


def test_indexed_summary_matches_masked_summary_for_fast_compartments() -> None:
    image = np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4)
    wc = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint32,
    )
    nuc = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint32,
    )
    cell = _make_cell()
    cache = build_tile_label_index_cache([cell], wc_tile=wc, nuc_tile=nuc)
    indexed = indexed_compartment_indices(cell, cache=cache)
    assert indexed is not None

    props_indexed: dict[str, float] = {}
    add_indexed_summary_stats(
        props_indexed,
        image,
        ["ch1"],
        indexed,
        include_median=True,
    )

    cell_mask = wc.astype(bool)
    nuc_mask = nuc.astype(bool) & cell_mask
    comp_masks = compartment_masks(cell_mask, nuc_mask)
    props_masked: dict[str, float] = {}
    add_masked_summary_stats(
        props_masked,
        image,
        ["ch1"],
        comp_masks,
        compartments=("CELL", "NUCLEUS", "CYTOPLASM"),
        include_median=True,
    )

    for key, value in props_masked.items():
        assert key in props_indexed
        assert abs(props_indexed[key] - value) < 1e-8
