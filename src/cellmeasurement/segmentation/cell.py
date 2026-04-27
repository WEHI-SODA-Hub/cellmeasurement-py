from __future__ import annotations

from dataclasses import dataclass

from .types import BBox, Centroid, MatchSource


@dataclass
class CellMatch:
    """Result of matching a nuclear ROI to a whole-cell ROI.

    Represents one cell in the unified output, linking the sequential output
    ``cell_id`` to the original label values in the input masks, together with
    spatial metadata and match provenance.

    Attributes:
        cell_id: Sequential output cell identifier starting at 1.
        nucleus_label: Original label in the nuclear mask, or ``None`` if this
            cell has no nucleus (whole-cell-only mode or unresolvable).
        whole_cell_label: Original label in the whole-cell mask, or ``None`` if
            the cell boundary was synthesised via watershed expansion.
        bbox: Bounding box as ``(row_min, col_min, row_max_excl, col_max_excl)``
            in full-image pixel coordinates using Python slice convention
            (exclusive upper bound).
        centroid: ``(row, col)`` centroid in full-image pixel coordinates.
        nucleus_area_px: Number of pixels in the nuclear label.  Zero for
            whole-cell-only mode.
        cell_area_px: Number of pixels in the cell (whole-cell or synthesised).
            Zero for nuclear-only mode.
        overlap_px: Number of pixels where the nuclear and whole-cell labels
            coincide.  Zero for synthesised cells and single-mask modes.
        overlap_fraction: ``overlap_px / nucleus_area_px``.  Zero when
            ``nucleus_area_px`` is zero.
        match_source: Provenance tag describing how this cell was created.

            * ``"overlap_1to1"`` – matched by pixel overlap (primary path).
            * ``"watershed_synth"`` – nucleus had no matching whole-cell label;
              boundary was synthesised by watershed expansion.
            * ``"wc_only"`` – whole-cell-only mode; no nuclear mask was
              provided.
            * ``"nuc_only"`` – nuclear-only mode; no whole-cell mask was
              provided.
    """

    cell_id: int
    nucleus_label: int | None
    whole_cell_label: int | None
    # (row_min, col_min, row_max_excl, col_max_excl) — exclusive upper bounds
    bbox: BBox
    centroid: Centroid
    nucleus_area_px: int
    cell_area_px: int
    overlap_px: int
    overlap_fraction: float
    match_source: MatchSource
