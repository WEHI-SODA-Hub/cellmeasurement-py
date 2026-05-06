from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from cellmeasurement.measurement.compartment_metrics import (
    _DISK_1,
    _add_environment_measurements,
    _add_erosion_measurements,
    _add_expansion_measurements,
    _compartment_masks,
    _erosion_bins_for_mask,
    _expansion_bins_for_mask,
)


def _circular_mask(radius: int, size: int = 0) -> np.ndarray:
    if size == 0:
        size = 2 * radius + 11
    centre = size // 2
    yy, xx = np.ogrid[:size, :size]
    return ((xx - centre) ** 2 + (yy - centre) ** 2 <= radius**2).astype(bool)


def _default_comp_masks(cell_mask: np.ndarray, nuc_mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if nuc_mask is None:
        nuc_mask = np.zeros_like(cell_mask, dtype=bool)
    return _compartment_masks(cell_mask, nuc_mask)


def test_erosion_bins_returns_requested_count() -> None:
    mask = _circular_mask(20)
    for n_bins in (3, 5, 7):
        bins = _erosion_bins_for_mask(mask, n_bins=n_bins)
        assert len(bins) == n_bins


def test_erosion_bins_empty_mask_returns_empty() -> None:
    mask = np.zeros((30, 30), dtype=bool)
    assert _erosion_bins_for_mask(mask) == []


def test_erosion_bins_small_mask_pads_to_n_bins() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 4:6] = True
    bins = _erosion_bins_for_mask(mask, n_bins=5)
    assert len(bins) == 5


def test_erosion_bin_depths_non_decreasing() -> None:
    bins = _erosion_bins_for_mask(_circular_mask(25), n_bins=5)
    depths = [depth for _, depth in bins]
    assert depths == sorted(depths)


def test_erosion_bin_areas_monotonic_non_increasing() -> None:
    bins = _erosion_bins_for_mask(_circular_mask(25), n_bins=5)
    areas = [int(np.count_nonzero(mask)) for mask, _ in bins]
    assert all(curr <= prev for prev, curr in zip(areas, areas[1:]))


def test_add_erosion_measurements_ring_areas_cover_cell_mask() -> None:
    mask = _circular_mask(25)
    image = np.ones((1, *mask.shape), dtype=np.float32) * 42.0
    props: dict[str, float] = {}
    _add_erosion_measurements(props, image, ["ch1"], _default_comp_masks(mask), n_bins=5)

    total_area = int(np.count_nonzero(mask))
    ring_areas = [props[f"Cell: ErosionBin_{i}: Area_px"] for i in range(1, 6)]
    assert sum(ring_areas) == total_area


def test_add_erosion_measurements_fractions_sum_to_one() -> None:
    mask = _circular_mask(25)
    image = np.ones((1, *mask.shape), dtype=np.float32)
    props: dict[str, float] = {}
    _add_erosion_measurements(props, image, ["ch1"], _default_comp_masks(mask), n_bins=5)

    fracs = [props[f"Cell: ErosionBin_{i}: Area_Fraction"] for i in range(1, 6)]
    assert abs(sum(fracs) - 1.0) < 1e-9


def test_add_erosion_measurements_uniform_intensity_per_bin() -> None:
    mask = _circular_mask(20)
    image = np.full((1, *mask.shape), 7.0, dtype=np.float32)
    props: dict[str, float] = {}
    _add_erosion_measurements(props, image, ["ch1"], _default_comp_masks(mask), n_bins=5)

    for i in range(1, 6):
        area = props[f"Cell: ErosionBin_{i}: Area_px"]
        key = f"ch1: Cell: ErosionBin_{i}: Mean"
        if area > 0:
            assert key in props
            assert abs(props[key] - 7.0) < 1e-6
        else:
            assert key not in props


def test_add_erosion_measurements_radial_gradient_decreases_inward() -> None:
    size = 61
    mask = _circular_mask(25, size=size)
    centre = size // 2
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2).astype(np.float32)
    image = dist[np.newaxis, :, :]

    props: dict[str, float] = {}
    _add_erosion_measurements(props, image, ["ch1"], _default_comp_masks(mask), n_bins=5)
    means = [props[f"ch1: Cell: ErosionBin_{i}: Mean"] for i in range(1, 6) if f"ch1: Cell: ErosionBin_{i}: Mean" in props]

    assert all(curr <= prev for prev, curr in zip(means, means[1:]))


def test_add_erosion_measurements_produces_nucleus_bins() -> None:
    cell = _circular_mask(25)
    nucleus = _circular_mask(12, size=cell.shape[0])
    image = np.ones((1, *cell.shape), dtype=np.float32)
    props: dict[str, float] = {}

    _add_erosion_measurements(props, image, ["ch1"], _default_comp_masks(cell, nucleus), n_bins=5)

    for i in range(1, 6):
        assert f"Nucleus: ErosionBin_{i}: Area_px" in props


def test_add_erosion_measurements_nucleus_ring_areas_sum_to_nucleus_area() -> None:
    cell = _circular_mask(25)
    nucleus = _circular_mask(12, size=cell.shape[0])
    image = np.ones((1, *cell.shape), dtype=np.float32)
    props: dict[str, float] = {}

    _add_erosion_measurements(props, image, ["ch1"], _default_comp_masks(cell, nucleus), n_bins=5)
    ring_sum = sum(props[f"Nucleus: ErosionBin_{i}: Area_px"] for i in range(1, 6))
    assert ring_sum == int(np.count_nonzero(nucleus))


def test_expansion_bins_returns_requested_count() -> None:
    mask = _circular_mask(15, size=200)
    for n_bins in (3, 5, 7):
        bins = _expansion_bins_for_mask(mask, total_expansion_px=40, n_bins=n_bins)
        assert len(bins) == n_bins


def test_expansion_bins_empty_mask_returns_empty() -> None:
    assert _expansion_bins_for_mask(np.zeros((30, 30), dtype=bool), total_expansion_px=10) == []


def test_expansion_bin_depths_non_decreasing() -> None:
    bins = _expansion_bins_for_mask(_circular_mask(15, size=200), total_expansion_px=40, n_bins=5)
    depths = [depth for _, depth in bins]
    assert depths == sorted(depths)


def test_expansion_bin_areas_monotonic_non_decreasing() -> None:
    bins = _expansion_bins_for_mask(_circular_mask(15, size=200), total_expansion_px=40, n_bins=5)
    areas = [int(np.count_nonzero(mask)) for mask, _ in bins]
    assert all(curr >= prev for prev, curr in zip(areas, areas[1:]))


def test_add_expansion_measurements_uniform_intensity_per_bin() -> None:
    mask = _circular_mask(15, size=200)
    image = np.full((1, *mask.shape), 9.0, dtype=np.float32)
    props: dict[str, float] = {}
    _add_expansion_measurements(props, image, ["ch1"], mask, pixel_size_microns=0.5, n_bins=5)

    for i in range(1, 6):
        area = props[f"Cell: ExpansionBin_{i}: Area_px"]
        key = f"ch1: Cell: ExpansionBin_{i}: Mean"
        if area > 0:
            assert key in props
            assert abs(props[key] - 9.0) < 1e-6


def test_add_expansion_measurements_radial_gradient_increases_outward() -> None:
    size = 200
    mask = _circular_mask(15, size=size)
    centre = size // 2
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2).astype(np.float32)
    image = dist[np.newaxis, :, :]
    props: dict[str, float] = {}
    _add_expansion_measurements(props, image, ["ch1"], mask, pixel_size_microns=0.5, n_bins=5)

    means = [props[f"ch1: Cell: ExpansionBin_{i}: Mean"] for i in range(1, 6) if f"ch1: Cell: ExpansionBin_{i}: Mean" in props]
    assert all(curr >= prev for prev, curr in zip(means, means[1:]))


def test_add_expansion_measurements_pixel_size_controls_total_expansion_area() -> None:
    mask = _circular_mask(15, size=300)
    image = np.ones((1, *mask.shape), dtype=np.float32)

    coarse_props: dict[str, float] = {}
    _add_expansion_measurements(coarse_props, image, ["ch1"], mask, pixel_size_microns=1.0, n_bins=5)
    fine_props: dict[str, float] = {}
    _add_expansion_measurements(fine_props, image, ["ch1"], mask, pixel_size_microns=0.5, n_bins=5)

    coarse_total = sum(coarse_props.get(f"Cell: ExpansionBin_{i}: Area_px", 0.0) for i in range(1, 6))
    fine_total = sum(fine_props.get(f"Cell: ExpansionBin_{i}: Area_px", 0.0) for i in range(1, 6))
    assert fine_total > coarse_total


def test_expansion_bin1_contains_immediate_adjacent_ring() -> None:
    mask = _circular_mask(15, size=200)
    dilated_once = ndi.binary_dilation(mask, structure=_DISK_1, iterations=1)
    adjacent_ring = dilated_once & ~mask

    bins = _expansion_bins_for_mask(mask, total_expansion_px=max(1, int(round(20.0 / 0.5))), n_bins=5)
    bin1_ring = bins[0][0] & ~mask
    assert np.all(bin1_ring[adjacent_ring])


def test_erosion_bin1_contains_outer_boundary_ring() -> None:
    mask = _circular_mask(25)
    eroded_once = ndi.binary_erosion(mask, structure=_DISK_1, iterations=1)
    outer_ring = mask & ~eroded_once

    bins = _erosion_bins_for_mask(mask, n_bins=5)
    bin1_ring = mask & ~bins[0][0]
    assert np.all(bin1_ring[outer_ring])


def test_add_environment_measurements_emits_keys_and_nonzero_area() -> None:
    mask = _circular_mask(15)
    image = np.ones((2, *mask.shape), dtype=np.float32)
    props: dict[str, float] = {}

    _add_environment_measurements(props, image, ["ch1", "ch2"], mask, pixel_size_microns=0.5)

    assert props["Cell: Environment_20um: Pixel_Count"] > 0
    assert "Cell: Environment_20um: Area_Fraction" in props
    assert "ch1: Cell: Environment_20um: Mean" in props
    assert "ch2: Cell: Environment_20um: Mean" in props


def test_add_environment_measurements_pixel_size_controls_environment_area() -> None:
    mask = _circular_mask(15, size=200)
    image = np.ones((1, *mask.shape), dtype=np.float32)
    coarse_props: dict[str, float] = {}
    fine_props: dict[str, float] = {}

    _add_environment_measurements(coarse_props, image, ["ch1"], mask, pixel_size_microns=1.0)
    _add_environment_measurements(fine_props, image, ["ch1"], mask, pixel_size_microns=0.5)

    assert fine_props["Cell: Environment_20um: Pixel_Count"] > coarse_props["Cell: Environment_20um: Pixel_Count"]


def test_add_environment_measurements_empty_mask_is_noop() -> None:
    mask = np.zeros((30, 30), dtype=bool)
    image = np.ones((1, *mask.shape), dtype=np.float32)
    props: dict[str, float] = {}

    _add_environment_measurements(props, image, ["ch1"], mask, pixel_size_microns=0.5)
    assert props == {}
