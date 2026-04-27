"""Unit tests for the ROI matcher and supporting utilities."""

from __future__ import annotations

import numpy as np
import pytest
import dask.array as da

from cellmeasurement.segmentation.roi_matcher import (
    _count_overlaps_chunk,
    _label_stats_chunk,
    _merge_overlap_counts,
    _merge_stats,
    _resolve_one_to_one,
    match_rois,
)
from cellmeasurement.io.mask_reader import validate_grid_compatibility


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_da(arr: np.ndarray, chunks: tuple[int, int] = (4, 4)) -> da.Array:
    """Wrap a small numpy array in a single-chunk dask array."""
    return da.from_array(arr, chunks=chunks)


# Two distinct nuclei and whole-cells with clean 1-to-1 overlap.
#
#   nuc layout (4 × 4):      wc layout (4 × 4):
#   0  1  0  0                0  10  0   0
#   0  1  0  0                0  10  0   0
#   0  0  2  0                0   0  20  0
#   0  0  2  0                0   0  20  0
NUC_2x2 = np.array(
    [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 2, 0]], dtype=np.int32
)
WC_2x2 = np.array(
    [[0, 10, 0, 0], [0, 10, 0, 0], [0, 0, 20, 0], [0, 0, 20, 0]], dtype=np.int32
)

# Conflict fixture: two nuclei overlap the SAME whole-cell with equal overlap.
#   nuc:  [[1, 1, 2, 2], zeros ...]
#   wc:   [[10,10,10,10], zeros ...]
NUC_CONFLICT = np.array(
    [[1, 1, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32
)
WC_CONFLICT = np.array(
    [[10, 10, 10, 10], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32
)


# ---------------------------------------------------------------------------
# _label_stats_chunk
# ---------------------------------------------------------------------------


class TestLabelStatsChunk:
    def test_empty_returns_empty(self):
        arr = np.zeros((4, 4), dtype=np.int32)
        assert _label_stats_chunk(arr, 0, 0) == {}

    def test_basic_stats(self):
        stats = _label_stats_chunk(NUC_2x2, 0, 0)
        assert set(stats.keys()) == {1, 2}

        s1 = stats[1]
        assert s1["area"] == 2
        assert s1["row_min"] == 0
        assert s1["row_max"] == 1
        assert s1["col_min"] == 1
        assert s1["col_max"] == 1
        assert s1["row_sum"] == pytest.approx(0 + 1)  # rows 0 and 1
        assert s1["col_sum"] == pytest.approx(1 + 1)  # both col 1

        s2 = stats[2]
        assert s2["area"] == 2
        assert s2["row_min"] == 2
        assert s2["row_max"] == 3

    def test_offset_applied(self):
        # Shift by (10, 20) should translate all global coordinates.
        stats = _label_stats_chunk(NUC_2x2, row_off=10, col_off=20)
        s1 = stats[1]
        assert s1["row_min"] == 10
        assert s1["col_min"] == 21
        assert s1["row_sum"] == pytest.approx(10 + 11)
        assert s1["col_sum"] == pytest.approx(21 + 21)


# ---------------------------------------------------------------------------
# _count_overlaps_chunk
# ---------------------------------------------------------------------------


class TestCountOverlapsChunk:
    def test_clean_overlap(self):
        ov, ns, ws = _count_overlaps_chunk(NUC_2x2, WC_2x2, 0, 0)
        assert ov == {(1, 10): 2, (2, 20): 2}
        assert set(ns.keys()) == {1, 2}
        assert set(ws.keys()) == {10, 20}

    def test_no_overlap(self):
        # Shift wc so nothing overlaps.
        wc_shifted = np.zeros_like(WC_2x2)
        wc_shifted[0, :] = 99  # top row — nuc is in col 1 rows 0-1; no coincidence
        # Actually let's use a fully non-overlapping layout.
        nuc = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
        wc = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 10, 0]], dtype=np.int32)
        ov, ns, ws = _count_overlaps_chunk(nuc, wc, 0, 0)
        assert ov == {}
        assert 1 in ns
        assert 10 in ws

    def test_background_ignored(self):
        # Background (0) must never appear in stats or overlaps.
        nuc = np.array([[0, 1], [0, 0]], dtype=np.int32)
        wc = np.array([[0, 10], [0, 0]], dtype=np.int32)
        ov, ns, ws = _count_overlaps_chunk(nuc, wc, 0, 0)
        assert 0 not in ns
        assert 0 not in ws
        assert all(k[0] != 0 and k[1] != 0 for k in ov)


# ---------------------------------------------------------------------------
# _merge_overlap_counts and _merge_stats
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merge_overlap_counts_sums(self):
        a = {(1, 10): 3, (2, 20): 1}
        b = {(1, 10): 2, (3, 30): 5}
        merged = _merge_overlap_counts([a, b])
        assert merged == {(1, 10): 5, (2, 20): 1, (3, 30): 5}

    def test_merge_stats_accumulates(self):
        s1 = {1: {"area": 2, "row_min": 0, "row_max": 1, "col_min": 1, "col_max": 1,
                  "row_sum": 1.0, "col_sum": 2.0}}
        s2 = {1: {"area": 3, "row_min": 2, "row_max": 4, "col_min": 0, "col_max": 2,
                  "row_sum": 9.0, "col_sum": 3.0}}
        merged = _merge_stats([s1, s2])
        m = merged[1]
        assert m["area"] == 5
        assert m["row_min"] == 0
        assert m["row_max"] == 4
        assert m["col_min"] == 0
        assert m["col_max"] == 2
        assert m["row_sum"] == pytest.approx(10.0)
        assert m["col_sum"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _resolve_one_to_one
# ---------------------------------------------------------------------------


class TestResolveOneToOne:
    def _make_stats(self, label_id: int, area: int = 4) -> dict:
        r = float(label_id)
        return {"area": area, "row_min": 0, "row_max": 1, "col_min": 0, "col_max": 1,
                "row_sum": r * area, "col_sum": r * area}

    def test_trivial_no_conflict(self):
        overlap = {(1, 10): 4, (2, 20): 4}
        ns = {1: self._make_stats(1), 2: self._make_stats(2)}
        ws = {10: self._make_stats(10), 20: self._make_stats(20)}
        matches, mn, mw, nid = _resolve_one_to_one(overlap, ns, ws, next_id=1)
        assert len(matches) == 2
        nuc_labels = {m.nucleus_label for m in matches}
        wc_labels = {m.whole_cell_label for m in matches}
        assert nuc_labels == {1, 2}
        assert wc_labels == {10, 20}
        assert mn == {1, 2}
        assert mw == {10, 20}

    def test_conflict_higher_overlap_wins(self):
        # Nucleus 1 has 4px overlap with wc 10; nucleus 2 has 2px.  Nuc 1 wins.
        overlap = {(1, 10): 4, (2, 10): 2}
        ns = {1: self._make_stats(1), 2: self._make_stats(2)}
        ws = {10: self._make_stats(10)}
        matches, mn, mw, _ = _resolve_one_to_one(overlap, ns, ws, next_id=1)
        assert len(matches) == 1
        assert matches[0].nucleus_label == 1
        assert 2 not in mn

    def test_conflict_tie_break_lower_nuc_id(self):
        # Equal overlap; lower nuc_id (1 < 2) should win.
        overlap = {(1, 10): 4, (2, 10): 4}
        ns = {1: self._make_stats(1), 2: self._make_stats(2)}
        ws = {10: self._make_stats(10)}
        matches, mn, _, _ = _resolve_one_to_one(overlap, ns, ws, next_id=1)
        assert len(matches) == 1
        assert matches[0].nucleus_label == 1
        assert 2 not in mn

    def test_cell_ids_are_sequential(self):
        overlap = {(1, 10): 4, (2, 20): 4}
        ns = {1: self._make_stats(1), 2: self._make_stats(2)}
        ws = {10: self._make_stats(10), 20: self._make_stats(20)}
        matches, _, _, nid = _resolve_one_to_one(overlap, ns, ws, next_id=5)
        ids = {m.cell_id for m in matches}
        assert ids == {5, 6}
        assert nid == 7

    def test_overlap_fraction_computed(self):
        overlap = {(1, 10): 3}
        ns = {1: self._make_stats(1, area=4)}
        ws = {10: self._make_stats(10, area=6)}
        matches, _, _, _ = _resolve_one_to_one(overlap, ns, ws, next_id=1)
        assert matches[0].overlap_fraction == pytest.approx(3 / 4)
        assert matches[0].overlap_px == 3

    def test_bbox_is_union_exclusive_max(self):
        nuc_s = {"area": 2, "row_min": 1, "row_max": 2, "col_min": 3, "col_max": 4,
                 "row_sum": 3.0, "col_sum": 7.0}
        wc_s = {"area": 4, "row_min": 0, "row_max": 3, "col_min": 2, "col_max": 5,
                "row_sum": 6.0, "col_sum": 14.0}
        matches, _, _, _ = _resolve_one_to_one({(1, 10): 2}, {1: nuc_s}, {10: wc_s}, next_id=1)
        r_min, c_min, r_max_excl, c_max_excl = matches[0].bbox
        assert r_min == 0
        assert c_min == 2
        assert r_max_excl == 4    # row_max=3, exclusive → 4
        assert c_max_excl == 6    # col_max=5, exclusive → 6

    def test_match_source_label(self):
        overlap = {(1, 10): 4}
        ns = {1: self._make_stats(1)}
        ws = {10: self._make_stats(10)}
        matches, _, _, _ = _resolve_one_to_one(overlap, ns, ws, next_id=1)
        assert matches[0].match_source == "overlap_1to1"


# ---------------------------------------------------------------------------
# match_rois — end-to-end integration tests
# ---------------------------------------------------------------------------


class TestMatchRois:
    def test_raises_if_both_none(self):
        with pytest.raises(ValueError, match="At least one"):
            match_rois(None, None)

    def test_wc_only_mode(self):
        wc = _make_da(WC_2x2)
        cells, synth_geoms = match_rois(None, wc)
        assert all(c.match_source == "wc_only" for c in cells)
        assert all(c.nucleus_label is None for c in cells)
        assert {c.whole_cell_label for c in cells} == {10, 20}
        assert synth_geoms == {}

    def test_nuc_only_mode(self):
        nuc = _make_da(NUC_2x2)
        cells, synth_geoms = match_rois(nuc, None)
        assert all(c.match_source == "nuc_only" for c in cells)
        assert all(c.whole_cell_label is None for c in cells)
        assert {c.nucleus_label for c in cells} == {1, 2}
        assert synth_geoms == {}

    def test_paired_mode_clean_match(self):
        nuc = _make_da(NUC_2x2)
        wc = _make_da(WC_2x2)
        cells, synth_geoms = match_rois(nuc, wc, synthesis_dist=1.0)
        overlap_cells = [c for c in cells if c.match_source == "overlap_1to1"]
        assert len(overlap_cells) == 2
        pairs = {(c.nucleus_label, c.whole_cell_label) for c in overlap_cells}
        assert pairs == {(1, 10), (2, 20)}
        assert synth_geoms == {}

    def test_paired_mode_unmatched_nucleus_gets_synthesis(self):
        # Nucleus 2 has no overlapping wc; it should receive a synthesised boundary.
        nuc = np.array(
            [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 2, 0]], dtype=np.int32
        )
        wc = np.array(
            [[0, 10, 0, 0], [0, 10, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32
        )
        cells, synth_geoms = match_rois(_make_da(nuc), _make_da(wc), synthesis_dist=1.5)
        sources = {c.match_source for c in cells}
        assert "overlap_1to1" in sources
        assert "watershed_synth" in sources
        synth = [c for c in cells if c.match_source == "watershed_synth"]
        assert len(synth) == 1
        assert synth[0].nucleus_label == 2
        assert synth[0].whole_cell_label is None
        # synth_geoms must have a polygon for the synthesised cell.
        assert synth[0].cell_id in synth_geoms
        from shapely.geometry import Polygon
        assert isinstance(synth_geoms[synth[0].cell_id], Polygon)

    def test_conflict_only_one_match_produced(self):
        nuc = _make_da(NUC_CONFLICT)
        wc = _make_da(WC_CONFLICT)
        cells, _ = match_rois(nuc, wc, synthesis_dist=1.0)
        overlap = [c for c in cells if c.match_source == "overlap_1to1"]
        assert len(overlap) == 1

    def test_cell_ids_start_at_one(self):
        nuc = _make_da(NUC_2x2)
        wc = _make_da(WC_2x2)
        cells, _ = match_rois(nuc, wc)
        ids = sorted(c.cell_id for c in cells)
        assert ids[0] == 1

    def test_bbox_exclusive_upper_bound(self):
        """Every bbox max bound must be strictly greater than the min bound."""
        nuc = _make_da(NUC_2x2)
        wc = _make_da(WC_2x2)
        cells, _ = match_rois(nuc, wc)
        for cell in cells:
            r0, c0, r1, c1 = cell.bbox
            assert r1 > r0, f"row max_excl ({r1}) must exceed row_min ({r0})"
            assert c1 > c0, f"col max_excl ({c1}) must exceed col_min ({c0})"

    def test_multi_chunk_gives_same_result(self):
        """Splitting the array into multiple chunks must yield identical output."""
        nuc_single = _make_da(NUC_2x2, chunks=(4, 4))
        wc_single = _make_da(WC_2x2, chunks=(4, 4))
        nuc_multi = _make_da(NUC_2x2, chunks=(2, 2))
        wc_multi = _make_da(WC_2x2, chunks=(2, 2))

        cells_single_raw, _ = match_rois(nuc_single, wc_single)
        cells_single = sorted(cells_single_raw, key=lambda c: c.nucleus_label or 0)
        cells_multi_raw, _ = match_rois(nuc_multi, wc_multi)
        cells_multi = sorted(cells_multi_raw, key=lambda c: c.nucleus_label or 0)

        assert len(cells_single) == len(cells_multi)
        for cs, cm in zip(cells_single, cells_multi):
            assert cs.nucleus_label == cm.nucleus_label
            assert cs.whole_cell_label == cm.whole_cell_label
            assert cs.overlap_px == cm.overlap_px
            assert cs.bbox == cm.bbox


# ---------------------------------------------------------------------------
# validate_grid_compatibility
# ---------------------------------------------------------------------------


class TestValidateGridCompatibility:
    def test_compatible_shapes_pass(self):
        class FakeMask:
            def __init__(self, shape):
                self.shape = shape
                self.method = "test"

        validate_grid_compatibility(FakeMask((100, 100)), FakeMask((100, 100)))  # type: ignore[arg-type]

    def test_incompatible_shapes_raise(self):
        class FakeMask:
            def __init__(self, shape):
                self.shape = shape
                self.method = "test"

        with pytest.raises(ValueError, match="incompatible"):
            validate_grid_compatibility(FakeMask((100, 100)), FakeMask((200, 100)))  # type: ignore[arg-type]
