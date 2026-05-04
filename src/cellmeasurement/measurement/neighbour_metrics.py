from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from ..segmentation.cell import CellMatch


def _ordered_measurement_cell_ids(
    measurements_by_cell: dict[int, dict[str, float]],
    cells: Sequence[CellMatch],
) -> tuple[list[int], dict[int, tuple[float, float]]]:
    """Return measurement cell IDs that have centroids, in deterministic order."""
    centroid_by_cell: dict[int, tuple[float, float]] = {cell.cell_id: cell.centroid for cell in cells}
    ordered_ids = [cid for cid in sorted(measurements_by_cell) if cid in centroid_by_cell]
    return ordered_ids, centroid_by_cell


def _collect_numeric_measurement_keys(
    measurements_by_cell: dict[int, dict[str, float]],
    ordered_ids: Sequence[int],
) -> set[str]:
    """Return base numeric measurement keys eligible for neighbour aggregation."""
    numeric_keys: set[str] = set()
    for cell_id in ordered_ids:
        for key, value in measurements_by_cell[cell_id].items():
            if key.startswith("Neighbours: "):
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric_keys.add(key)
    return numeric_keys


def _build_measurement_key_vectors(
    measurements_by_cell: dict[int, dict[str, float]],
    ordered_ids: Sequence[int],
    numeric_keys: set[str],
) -> dict[str, np.ndarray]:
    """Build dense vectors (cell-aligned) for each numeric measurement key."""
    key_vectors: dict[str, np.ndarray] = {}
    for key in numeric_keys:
        arr = np.full(len(ordered_ids), np.nan, dtype=np.float64)
        for i, cell_id in enumerate(ordered_ids):
            value = measurements_by_cell[cell_id].get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                arr[i] = float(value)
        key_vectors[key] = arr
    return key_vectors


def _query_neighbour_indices(
    ordered_ids: Sequence[int],
    centroid_by_cell: dict[int, tuple[float, float]],
    neighbours: int,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Query k-nearest neighbours in centroid space."""
    if len(ordered_ids) < 2:
        return None

    centroids = np.array([centroid_by_cell[cid] for cid in ordered_ids], dtype=np.float64)
    tree = cKDTree(centroids)
    actual_k = min(neighbours + 1, len(ordered_ids))
    if actual_k <= 1:
        return None

    distances, indices = tree.query(centroids, k=actual_k)
    return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64), actual_k


def _add_neighbour_measurements(
    measurements_by_cell: dict[int, dict[str, float]],
    cells: Sequence[CellMatch],
    neighbours: int,
    pixel_size_microns: float,
) -> None:
    """Aggregate each cell's numeric metrics over k nearest neighbours.

    Neighbours are selected in centroid space with a hard 20 µm distance cap
    (converted to pixels via ``pixel_size_microns``), then aggregated per key.
    Current behaviour intentionally writes mean-only neighbour summaries:

    ``Neighbours: Mean: <original measurement key>``.
    """
    if neighbours <= 0 or len(measurements_by_cell) < 2:
        return

    ordered_ids, centroid_by_cell = _ordered_measurement_cell_ids(measurements_by_cell, cells)
    query = _query_neighbour_indices(ordered_ids, centroid_by_cell, neighbours)
    if query is None:
        return
    distances, indices, actual_k = query
    max_distance_px = 20.0 / pixel_size_microns
    numeric_keys = _collect_numeric_measurement_keys(measurements_by_cell, ordered_ids)
    if not numeric_keys:
        return

    # Build one dense vector per measurement key so each cell can aggregate
    # neighbours by fast index selection instead of repeated dict lookups.
    key_vectors = _build_measurement_key_vectors(measurements_by_cell, ordered_ids, numeric_keys)

    for i, cell_id in enumerate(ordered_ids):
        neighbour_idx = np.asarray(indices[i, 1:actual_k], dtype=np.int64)
        neighbour_dist = np.asarray(distances[i, 1:actual_k], dtype=np.float64)
        within = neighbour_dist <= max_distance_px
        neighbour_idx = neighbour_idx[within]
        if neighbour_idx.size == 0:
            continue

        cell_measurements = measurements_by_cell[cell_id]
        for key, arr in key_vectors.items():
            vals = arr[neighbour_idx]
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                cell_measurements[f"Neighbours: Mean: {key}"] = float(np.mean(vals))
