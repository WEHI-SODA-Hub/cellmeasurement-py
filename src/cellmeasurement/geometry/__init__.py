"""Geometry-domain utilities for polygon extraction and overlap clipping."""

from .geometry import boundaries_to_geometries, extract_label_geometries, mask_to_geometry
from .overlap_constraint import constrain_cell_overlaps

__all__ = [
    "mask_to_geometry",
    "extract_label_geometries",
    "boundaries_to_geometries",
    "constrain_cell_overlaps",
]
