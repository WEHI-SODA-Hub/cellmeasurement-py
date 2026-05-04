from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk

# Pre-computed 3x3 disk structuring element used throughout for single-pixel
# morphological erosion/dilation operations. Frozen to prevent accidental mutation.
_DISK_1 = disk(1).astype(bool)  # type: ignore[assignment]
_DISK_1.flags.writeable = False


def _compartment_masks(cell_mask: np.ndarray, nuc_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Derive CELL/NUCLEUS/CYTOPLASM/MEMBRANE boolean masks."""
    cm = cell_mask.astype(bool)
    nm = nuc_mask.astype(bool) & cm
    cyto = cm & ~nm
    mem = cm & ~ndi.binary_erosion(cm, structure=_DISK_1, iterations=1, border_value=0)  # type: ignore[arg-type]
    return {"CELL": cm, "NUCLEUS": nm, "CYTOPLASM": cyto, "MEMBRANE": mem}


def _stat_values(vals: np.ndarray) -> dict[str, float]:
    """Compute summary intensity statistics for a 1-D pixel array."""
    if vals.size == 0:
        return {}
    return {
        "Mean": float(np.mean(vals)),
        "Median": float(np.median(vals)),
        "Min": float(np.min(vals)),
        "Max": float(np.max(vals)),
        "Std.Dev.": float(np.std(vals)),
    }


def _add_intensity_measurements(
    props: dict[str, float],
    image_cyx: np.ndarray,
    ch_names: Sequence[str],
    comp_masks: dict[str, np.ndarray],
) -> None:
    """Populate baseline intensity stats for each channel/compartment pair.

    This is the core "QuPath-style" measurement block used by downstream tools:
    for each channel and each compartment mask we compute Mean/Median/Min/Max/
    Std.Dev. and store keys in the form:

    ``"<channel>: <Compartment>: <Stat>"``.

    Notes
    -----
    - ``comp_masks`` is expected to come from :func:`_compartment_masks`, so
      masks are already aligned to ``image_cyx`` spatial coordinates.
    - Empty compartments are skipped to avoid adding misleading zero-valued
      summary statistics for absent regions.
    """
    labels = {"CELL": "Cell", "NUCLEUS": "Nucleus", "CYTOPLASM": "Cytoplasm", "MEMBRANE": "Membrane"}
    for ci, ch in enumerate(ch_names):
        ch_img = image_cyx[ci]
        for comp, mask in comp_masks.items():
            # Boolean indexing flattens selected pixels to 1-D values for stats.
            vals = ch_img[mask]
            if vals.size == 0:
                continue
            for key, value in _stat_values(vals).items():
                props[f"{ch}: {labels[comp]}: {key}"] = value


def _add_percentiles(
    props: dict[str, float],
    image_cyx: np.ndarray,
    ch_names: Sequence[str],
    comp_masks: dict[str, np.ndarray],
    percentiles: Sequence[float],
) -> None:
    """Populate user-requested percentiles for each channel/compartment pair.

    Percentiles complement the fixed summary stats from
    :func:`_add_intensity_measurements` by exposing distribution shape (e.g.
    tails and skew). Keys are emitted as:

    ``"<channel>: <Compartment>: Percentile: <p>"``.

    ``percentiles`` is expected to be pre-parsed/validated at CLI boundary.
    """
    if not percentiles:
        return
    labels = {"CELL": "Cell", "NUCLEUS": "Nucleus", "CYTOPLASM": "Cytoplasm", "MEMBRANE": "Membrane"}
    for ci, ch in enumerate(ch_names):
        ch_img = image_cyx[ci]
        for comp, mask in comp_masks.items():
            vals = ch_img[mask]
            if vals.size == 0:
                continue
            for p in percentiles:
                props[f"{ch}: {labels[comp]}: Percentile: {p}"] = float(np.percentile(vals, p))


def _erosion_bins_for_mask(mask: np.ndarray, n_bins: int = 5) -> list[tuple[np.ndarray, int]]:
    """Compute cumulative equal-area erosion boundaries for one mask."""
    total = int(np.count_nonzero(mask))
    if total == 0:
        return []

    target_fractions = [(b / n_bins) for b in range(1, n_bins + 1)]
    bins: list[tuple[np.ndarray, int]] = []

    current = mask.astype(bool)
    depth = 0
    for target_frac in target_fractions:
        target_remaining = int(total * (1.0 - target_frac))
        while True:
            area = int(np.count_nonzero(current))
            if area <= target_remaining or area == 0:
                break

            # Avoid infinite loop if erosion does not change mask
            new_current = ndi.binary_erosion(current, structure=_DISK_1, iterations=1, border_value=0)
            if np.array_equal(new_current, current):
                break

            current = new_current
            depth += 1
        bins.append((current.copy(), depth))
        if area == 0:
            while len(bins) < n_bins:
                bins.append((current.copy(), depth))
            break

    return bins


def _add_erosion_measurements(
    props: dict[str, float],
    image_cyx: np.ndarray,
    ch_names: Sequence[str],
    comp_masks: dict[str, np.ndarray],
    n_bins: int = 5,
) -> None:
    """Populate equal-area inward erosion-bin measurements for cell and nucleus.

    The compartment is split into ``n_bins`` concentric shells from outside in.
    Bin boundaries are adaptive (equal-area targets), not fixed pixel depths.

    For each bin we emit:
    - geometric descriptors (``Area_px``, ``Area_Fraction``, ``Depth_px``)
    - per-channel mean/median intensity inside that bin shell

    ``Depth_px`` is cumulative erosion depth at the *inner* edge of the bin,
    mirroring the historical llm_rewrite semantics.
    """
    for comp in ("CELL", "NUCLEUS"):
        base = comp_masks[comp]
        base_area = int(np.count_nonzero(base))
        if base_area == 0:
            continue

        comp_name = comp.capitalize()
        bin_boundaries = _erosion_bins_for_mask(base, n_bins=n_bins)
        # Boundaries are cumulative masks; convert to mutually exclusive rings.
        prev_mask = base.astype(bool)
        for bin_idx, (eroded_mask, depth_px) in enumerate(bin_boundaries, start=1):
            ring = prev_mask & ~eroded_mask
            ring_area = int(np.count_nonzero(ring))

            props[f"{comp_name}: ErosionBin_{bin_idx}: Area_px"] = float(ring_area)
            props[f"{comp_name}: ErosionBin_{bin_idx}: Area_Fraction"] = float(ring_area / base_area)
            props[f"{comp_name}: ErosionBin_{bin_idx}: Depth_px"] = float(depth_px)

            if ring_area > 0:
                for ci, ch in enumerate(ch_names):
                    vals = image_cyx[ci][ring]
                    if vals.size > 0:
                        props[f"{ch}: {comp_name}: ErosionBin_{bin_idx}: Mean"] = float(np.mean(vals))
                        props[f"{ch}: {comp_name}: ErosionBin_{bin_idx}: Median"] = float(np.median(vals))

            prev_mask = eroded_mask


def _expansion_bins_for_mask(
    cell_mask: np.ndarray,
    total_expansion_px: int,
    n_bins: int = 5,
) -> list[tuple[np.ndarray, int]]:
    """Compute cumulative expansion boundaries splitting the 20 µm zone into equal-area bins."""
    cm = cell_mask.astype(bool)
    if not np.any(cm):
        return []

    full_dilated = ndi.binary_dilation(cm, structure=_DISK_1, iterations=total_expansion_px)
    zone = full_dilated & ~cm
    total_zone_area = int(np.count_nonzero(zone))
    if total_zone_area == 0:
        return []

    target_fractions = [(b / n_bins) for b in range(1, n_bins + 1)]
    bins: list[tuple[np.ndarray, int]] = []

    current = cm.copy()
    depth = 0
    for target_frac in target_fractions:
        target_area = int(total_zone_area * target_frac)
        while depth < total_expansion_px:
            current_ring_area = int(np.count_nonzero(current & ~cm))
            if current_ring_area >= target_area:
                break
            current = ndi.binary_dilation(current, structure=_DISK_1, iterations=1)
            depth += 1
        bins.append((current.copy(), depth))
        if depth >= total_expansion_px:
            while len(bins) < n_bins:
                bins.append((current.copy(), depth))
            break

    return bins


def _add_expansion_measurements(
    props: dict[str, float],
    image_cyx: np.ndarray,
    ch_names: Sequence[str],
    cell_mask: np.ndarray,
    pixel_size_microns: float,
    n_bins: int = 5,
) -> None:
    """Populate equal-area outward expansion-bin measurements for cell body.

    A fixed physical radius (20 µm) is converted to pixels using
    ``pixel_size_microns`` (already effective/scaled by any downsampling), then
    partitioned into ``n_bins`` approximately equal-area annular shells.

    Emitted keys mirror erosion naming but use ``ExpansionBin_<N>``.
    """
    expansion_um = 20.0
    total_expansion_px = max(1, int(round(expansion_um / pixel_size_microns)))

    cm = cell_mask.astype(bool)
    base_area = int(np.count_nonzero(cm))
    if base_area == 0:
        return

    bin_boundaries = _expansion_bins_for_mask(cm, total_expansion_px, n_bins=n_bins)
    if not bin_boundaries:
        return

    # As with erosion: cumulative boundaries -> disjoint annular ring bins.
    prev_mask = cm.copy()
    for bin_idx, (dilated_mask, depth_px) in enumerate(bin_boundaries, start=1):
        ring = dilated_mask & ~prev_mask
        ring_area = int(np.count_nonzero(ring))

        props[f"Cell: ExpansionBin_{bin_idx}: Area_px"] = float(ring_area)
        props[f"Cell: ExpansionBin_{bin_idx}: Area_Fraction"] = float(ring_area / base_area)
        props[f"Cell: ExpansionBin_{bin_idx}: Depth_px"] = float(depth_px)

        if ring_area > 0:
            for ci, ch in enumerate(ch_names):
                vals = image_cyx[ci][ring]
                if vals.size > 0:
                    props[f"{ch}: Cell: ExpansionBin_{bin_idx}: Mean"] = float(np.mean(vals))
                    props[f"{ch}: Cell: ExpansionBin_{bin_idx}: Median"] = float(np.median(vals))

        prev_mask = dilated_mask


def _add_environment_measurements(
    props: dict[str, float],
    image_cyx: np.ndarray,
    ch_names: Sequence[str],
    cell_mask: np.ndarray,
    pixel_size_microns: float,
) -> None:
    """Populate single-zone 20 µm pericellular environment measurements.

    Unlike :func:`_add_expansion_measurements` (which yields multiple bins),
    this computes one aggregate environment compartment covering the full
    20 µm ring outside the cell boundary.
    """
    environment_um = 20.0
    expansion_px = max(1, int(round(environment_um / pixel_size_microns)))
    cm = cell_mask.astype(bool)
    if not np.any(cm):
        return

    dilated = ndi.binary_dilation(cm, structure=_DISK_1, iterations=expansion_px)
    env_mask = dilated & ~cm
    env_area = int(np.count_nonzero(env_mask))
    if env_area == 0:
        return

    base_area = int(np.count_nonzero(cm))
    props["Cell: Environment_20um: Pixel_Count"] = float(env_area)
    props["Cell: Environment_20um: Area_Fraction"] = float(env_area / base_area) if base_area > 0 else 0.0
    for ci, ch in enumerate(ch_names):
        # Keep the same stat family as base compartment metrics for consistency.
        vals = image_cyx[ci][env_mask]
        if vals.size == 0:
            continue
        props[f"{ch}: Cell: Environment_20um: Mean"] = float(np.mean(vals))
        props[f"{ch}: Cell: Environment_20um: Median"] = float(np.median(vals))
        props[f"{ch}: Cell: Environment_20um: Min"] = float(np.min(vals))
        props[f"{ch}: Cell: Environment_20um: Max"] = float(np.max(vals))
        props[f"{ch}: Cell: Environment_20um: Std.Dev."] = float(np.std(vals))
