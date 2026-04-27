from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

LabelId: TypeAlias = int
CellId: TypeAlias = int
MatchSource: TypeAlias = Literal["overlap_1to1", "watershed_synth", "wc_only", "nuc_only"]
BBox: TypeAlias = tuple[int, int, int, int]
Centroid: TypeAlias = tuple[float, float]
OverlapKey: TypeAlias = tuple[LabelId, LabelId]
OverlapCounts: TypeAlias = dict[OverlapKey, int]


class LabelStats(TypedDict):
    area: int
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    row_sum: float
    col_sum: float


LabelStatsById: TypeAlias = dict[LabelId, LabelStats]
ChunkResult: TypeAlias = tuple[OverlapCounts, LabelStatsById, LabelStatsById]
