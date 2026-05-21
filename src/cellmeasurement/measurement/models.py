from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PerformanceMode = Literal["exact", "fast", "fast_small_cells"]


@dataclass(frozen=True)
class MeasurementConfig:
    """Execution and feature settings shared across all tile tasks."""

    image_shape: tuple[int, int]
    tile_size: int
    tile_overlap: int
    workers: int
    percentiles: tuple[float, ...]
    erosion_enabled: bool
    expansion_enabled: bool
    environment_expansion_enabled: bool
    neighbours: int
    pixel_size_microns: float
    performance_mode: PerformanceMode = "exact"


@dataclass(frozen=True)
class TileBounds:
    """Inclusive/exclusive tile bounds in image coordinates."""

    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.r1 - self.r0, self.c1 - self.c0)

    def contains(self, other: "TileBounds") -> bool:
        return self.r0 <= other.r0 and self.c0 <= other.c0 and self.r1 >= other.r1 and self.c1 >= other.c1

    @classmethod
    def from_bbox(cls, bbox: tuple[int, int, int, int]) -> "TileBounds":
        return cls(r0=int(bbox[0]), c0=int(bbox[1]), r1=int(bbox[2]), c1=int(bbox[3]))


@dataclass(frozen=True)
class TileTask:
    """Serializable unit of tiled measurement work."""

    tile_key: tuple[int, int]
    cell_ids: tuple[int, ...]
    bounds: TileBounds


@dataclass
class TileResult:
    """Per-tile measurement payload and instrumentation summary."""

    measurements_by_cell: dict[int, dict[str, float]]
    fallback_reads: int
    baseline_seconds: float = 0.0
    percentile_seconds: float = 0.0
    morphology_seconds: float = 0.0
