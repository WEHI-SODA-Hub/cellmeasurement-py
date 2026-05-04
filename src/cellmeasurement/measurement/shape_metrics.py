from __future__ import annotations

import math

import numpy as np
from skimage.measure import regionprops

_MIN_CIRCULARITY_AREA_PX = 5.0


def _largest_region(mask: np.ndarray):
    """Return the largest connected component regionprops record for a mask."""
    regs = regionprops(mask.astype(np.uint8))
    if not regs:
        return None
    return max(regs, key=lambda r: r.area)


def _axis_major_length(region: object) -> float:
    """Return major-axis length across skimage API versions."""
    if hasattr(region, "axis_major_length"):
        return float(getattr(region, "axis_major_length"))
    return float(getattr(region, "major_axis_length"))


def _axis_minor_length(region: object) -> float:
    """Return minor-axis length across skimage API versions."""
    if hasattr(region, "axis_minor_length"):
        return float(getattr(region, "axis_minor_length"))
    return float(getattr(region, "minor_axis_length"))


def _clipped_circularity(area_px: float, perimeter_px: float, min_area_px: float = _MIN_CIRCULARITY_AREA_PX) -> float:
    """Compute circularity with clipping and tiny-object filtering."""
    if area_px < min_area_px or perimeter_px <= 0:
        return 0.0
    raw = float(4 * math.pi * area_px / (perimeter_px**2))
    return float(np.clip(raw, 0.0, 1.0))


def _basic_shape_metrics(
    cell_mask: np.ndarray,
    nuc_mask: np.ndarray,
    pixel_size_microns: float,
) -> dict[str, float]:
    """Compute cell/nucleus shape metrics from binary masks."""
    r = _largest_region(cell_mask)
    if r is None:
        return {}

    px_to_um = float(pixel_size_microns)
    area_scale = px_to_um**2
    perimeter = float(r.perimeter) if r.perimeter > 0 else 0.0
    cell_area_px = float(r.area)
    circularity = _clipped_circularity(cell_area_px, perimeter)
    major_px = _axis_major_length(r)
    minor_px = _axis_minor_length(r)
    out: dict[str, float] = {
        "Cell: Area µm^2": cell_area_px * area_scale,
        "Cell: Circularity": circularity,
        "Cell: Length µm": perimeter * px_to_um,
        "Cell: Max diameter µm": major_px * px_to_um,
        "Cell: Min diameter µm": minor_px * px_to_um,
        "Cell: Solidity": float(r.solidity) if r.solidity is not None else 0.0,
    }

    nr = _largest_region(nuc_mask)
    if nr is not None:
        n_area_px = float(nr.area)
        n_perimeter_px = float(nr.perimeter) if nr.perimeter > 0 else 0.0
        n_major_px = _axis_major_length(nr)
        n_minor_px = _axis_minor_length(nr)
        out["Nucleus: Area µm^2"] = n_area_px * area_scale
        out["Nucleus: Circularity"] = _clipped_circularity(n_area_px, n_perimeter_px)
        out["Nucleus: Length µm"] = n_perimeter_px * px_to_um
        out["Nucleus: Max diameter µm"] = n_major_px * px_to_um
        out["Nucleus: Min diameter µm"] = n_minor_px * px_to_um
        out["Nucleus: Solidity"] = float(nr.solidity) if nr.solidity is not None else 0.0
        out["Nucleus/Cell area ratio"] = n_area_px / cell_area_px if cell_area_px > 0 else 0.0

    return out
