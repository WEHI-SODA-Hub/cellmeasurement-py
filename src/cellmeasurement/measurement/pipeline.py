from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Iterator, Sequence, TextIO

import dask.array as da
import numpy as np
from shapely.geometry import Polygon
from skimage.draw import polygon as draw_polygon

from ..segmentation.cell import CellMatch
from .compartment_metrics import (
    add_environment_measurements,
    add_erosion_measurements,
    add_expansion_measurements,
    compartment_masks,
)
from .data_access import (
    InMemoryTileDataAccess,
    SharedArraySpec,
    TiffDataAccessSpec,
    TiffTileDataAccess,
    TileDataAccess,
    create_shared_array,
    open_shared_array,
)
from .image_io import _load_tiff_image, inspect_tiff_image
from .models import MeasurementConfig, PerformanceMode, TileBounds, TileResult, TileTask
from .neighbour_metrics import _add_neighbour_measurements
from .shape_metrics import _basic_shape_metrics
from .tile_intensity import (
    add_indexed_percentiles,
    add_indexed_summary_stats,
    add_masked_percentiles,
    add_masked_summary_stats,
    build_tile_label_index_cache,
    indexed_compartment_indices,
)

logger = logging.getLogger(__name__)

_MEMBRANE_ONLY = ("MEMBRANE",)


@dataclass
class _WorkerContext:
    config: MeasurementConfig
    ch_names: tuple[str, ...]
    synth_geoms: dict[int, Polygon]
    cells_by_id: dict[int, CellMatch]
    data_access: TileDataAccess


_WORKER_CONTEXT: _WorkerContext | None = None
_WORKER_SHARED_HANDLES: list[SharedMemory] = []


def _active_worker_context() -> _WorkerContext:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Measurement worker context has not been initialized.")
    return _WORKER_CONTEXT


def _polygon_to_local_mask(poly: Polygon, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Rasterize a global polygon into a local bbox mask."""
    r0, c0, r1, c1 = bbox
    h = max(0, r1 - r0)
    w = max(0, c1 - c0)
    mask = np.zeros((h, w), dtype=bool)
    if h == 0 or w == 0 or poly.is_empty:
        return mask

    ext = np.array(poly.exterior.coords)
    rr, cc = draw_polygon(ext[:, 1] - r0, ext[:, 0] - c0, shape=mask.shape)
    mask[rr, cc] = True
    for interior in poly.interiors:
        hole = np.array(interior.coords)
        hr, hc = draw_polygon(hole[:, 1] - r0, hole[:, 0] - c0, shape=mask.shape)
        mask[hr, hc] = False
    return mask


def _materialize_label_array(
    arr: da.Array | np.ndarray | None,
    *,
    name: str,
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    """Materialize one label array exactly once and validate shape."""
    if arr is None:
        return None
    t0 = time.perf_counter()
    source = "dask" if isinstance(arr, da.Array) else "numpy"
    materialized = np.asarray(arr.compute()) if isinstance(arr, da.Array) else np.asarray(arr)
    if materialized.ndim != 2:
        raise ValueError(f"{name} labels must be 2D, got shape={materialized.shape}")
    if tuple(materialized.shape) != expected_shape:
        raise ValueError(
            f"{name} labels shape {tuple(materialized.shape)} does not match measurement image shape {expected_shape}."
        )
    logger.info(
        "Materialized %s labels once in %.2fs: source=%s, shape=%s, dtype=%s",
        name,
        time.perf_counter() - t0,
        source,
        materialized.shape,
        materialized.dtype,
    )
    return materialized


def _materialize_labels_once(
    nuc_labels: da.Array | np.ndarray | None,
    wc_labels: da.Array | np.ndarray | None,
    *,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Materialize nuclear and whole-cell labels once for tiled measurement."""
    nuc_np = _materialize_label_array(nuc_labels, name="nuclear", expected_shape=expected_shape)
    wc_np = _materialize_label_array(wc_labels, name="whole-cell", expected_shape=expected_shape)
    return nuc_np, wc_np


def _validate_measure_cells_inputs(neighbours: int, downsample_factor: float, performance_mode: PerformanceMode) -> None:
    """Validate public input arguments for tiled measurement."""
    if neighbours < 0:
        raise ValueError("neighbours must be >= 0")
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be > 0")
    if performance_mode not in {"exact", "fast", "fast_small_cells"}:
        raise ValueError("performance_mode must be one of: exact, fast, fast_small_cells")


def _prepare_measurement_image_and_masks(
    tiff_file: Path,
    image_shape: tuple[int, int],
    nuc_labels: da.Array | np.ndarray | None,
    wc_labels: da.Array | np.ndarray | None,
    pixel_size_microns: float,
    downsample_factor: float,
) -> tuple[np.ndarray, list[str], da.Array | np.ndarray | None, da.Array | np.ndarray | None, float]:
    """Load TIFF image and apply optional downsampling to image/masks."""
    loaded_image_cyx, ch_names = _load_tiff_image(tiff_file)
    if loaded_image_cyx.shape[1:] != image_shape:
        raise ValueError(
            f"TIFF image shape {tuple(loaded_image_cyx.shape[1:])} does not match segmentation shape {image_shape}."
        )

    effective_pixel_size = pixel_size_microns
    if downsample_factor > 1.0:
        from ..io.mask_io import maybe_downsample

        step = int(round(downsample_factor))
        if step >= 2:
            nuc_np = None if nuc_labels is None else np.asarray(nuc_labels)
            wc_np = np.asarray(wc_labels)
            loaded_image_cyx, nuc_ds, wc_ds = maybe_downsample(loaded_image_cyx, nuc_np, wc_np, downsample_factor)
            nuc_labels = nuc_ds
            wc_labels = wc_ds
            effective_pixel_size = pixel_size_microns * step
            logger.info(
                "Applied downsampling factor %.1f to image and masks; effective pixel size now %.2f µm",
                downsample_factor,
                effective_pixel_size,
            )

    return loaded_image_cyx, ch_names, nuc_labels, wc_labels, effective_pixel_size


def _group_cells_by_tile_ids(cells: Sequence[CellMatch], tile_size: int) -> dict[tuple[int, int], list[int]]:
    """Group cell IDs by centroid-owned tile coordinates."""
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cell in cells:
        row, col = cell.centroid
        tile_row = int(row // tile_size)
        tile_col = int(col // tile_size)
        groups[(tile_row, tile_col)].append(cell.cell_id)
    return groups


def _tile_bounds_for_key(tile_key: tuple[int, int], *, config: MeasurementConfig) -> TileBounds:
    H, W = config.image_shape
    tile_row, tile_col = tile_key
    r0 = max(0, tile_row * config.tile_size - config.tile_overlap)
    c0 = max(0, tile_col * config.tile_size - config.tile_overlap)
    r1 = min(H, (tile_row + 1) * config.tile_size + config.tile_overlap)
    c1 = min(W, (tile_col + 1) * config.tile_size + config.tile_overlap)
    return TileBounds(r0=r0, c0=c0, r1=r1, c1=c1)


def _build_tile_tasks(
    tile_groups: dict[tuple[int, int], list[int]],
    *,
    config: MeasurementConfig,
) -> list[TileTask]:
    tasks: list[TileTask] = []
    for tile_key in sorted(tile_groups):
        tasks.append(
            TileTask(
                tile_key=tile_key,
                cell_ids=tuple(tile_groups[tile_key]),
                bounds=_tile_bounds_for_key(tile_key, config=config),
            )
        )
    return tasks


def _local_bbox_slices(bounds: TileBounds, bbox: tuple[int, int, int, int]) -> tuple[slice, slice]:
    br0, bc0, br1, bc1 = bbox
    return slice(br0 - bounds.r0, br1 - bounds.r0), slice(bc0 - bounds.c0, bc1 - bounds.c0)


def _cell_masks_from_crops(
    cell: CellMatch,
    nuc_crop: np.ndarray | None,
    wc_crop: np.ndarray | None,
    synth_geoms: dict[int, Polygon],
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build local cell and nucleus masks for one cell crop."""
    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]
    nuc_mask = np.zeros((h, w), dtype=bool)
    if nuc_crop is not None and cell.nucleus_label is not None:
        nuc_mask = nuc_crop == int(cell.nucleus_label)

    if cell.match_source == "watershed_synth":
        synth_poly = synth_geoms.get(cell.cell_id)
        cell_mask = _polygon_to_local_mask(synth_poly, bbox) if synth_poly is not None else nuc_mask.copy()
    elif wc_crop is not None and cell.whole_cell_label is not None:
        cell_mask = wc_crop == int(cell.whole_cell_label)
    elif nuc_crop is not None and cell.nucleus_label is not None:
        cell_mask = nuc_crop == int(cell.nucleus_label)
    else:
        cell_mask = np.zeros((h, w), dtype=bool)

    if not np.any(cell_mask) and np.any(nuc_mask):
        cell_mask = nuc_mask.copy()
    nuc_mask = nuc_mask & cell_mask
    return cell_mask, nuc_mask


def _include_median(mode: PerformanceMode) -> bool:
    return mode == "exact"


def _should_skip_percentiles(mode: PerformanceMode, *, cell_mask: np.ndarray) -> bool:
    if mode != "fast_small_cells":
        return False
    return int(np.count_nonzero(cell_mask)) < 25


def _has_whole_cell_boundary(cell: CellMatch) -> bool:
    return cell.whole_cell_label is not None or cell.match_source == "watershed_synth"


def _selected_intensity_compartments(
    cell: CellMatch,
    *,
    cell_mask: np.ndarray,
    nuc_mask: np.ndarray,
) -> tuple[str, ...]:
    compartments: list[str] = ["CELL"]
    has_nucleus = cell.nucleus_label is not None and np.any(nuc_mask)
    if has_nucleus:
        compartments.append("NUCLEUS")

    has_membrane = _has_whole_cell_boundary(cell)
    if has_membrane and has_nucleus and np.any(cell_mask & ~nuc_mask):
        compartments.append("CYTOPLASM")
    if has_membrane:
        compartments.append("MEMBRANE")
    return tuple(compartments)


def _measure_baseline_family(
    measurements: dict[str, float],
    *,
    cell: CellMatch,
    cell_mask: np.ndarray,
    nuc_mask: np.ndarray,
    image_crop: np.ndarray,
    image_tile: np.ndarray | None,
    indexed_comps: dict[str, np.ndarray] | None,
    ch_names: Sequence[str],
    pixel_size_microns: float,
    performance_mode: PerformanceMode,
) -> dict[str, np.ndarray]:
    measurements.update(_basic_shape_metrics(cell_mask, nuc_mask, pixel_size_microns=pixel_size_microns))
    comp_masks = compartment_masks(cell_mask, nuc_mask)
    selected_compartments = _selected_intensity_compartments(cell, cell_mask=cell_mask, nuc_mask=nuc_mask)
    selected_set = set(selected_compartments)
    include_median = _include_median(performance_mode)
    if image_tile is not None and indexed_comps is not None:
        indexed_selected = {comp: idx for comp, idx in indexed_comps.items() if comp in selected_set}
        add_indexed_summary_stats(
            measurements,
            image_tile,
            ch_names,
            indexed_selected,
            include_median=include_median,
        )
        if "MEMBRANE" in selected_set:
            add_masked_summary_stats(
                measurements,
                image_crop,
                ch_names,
                comp_masks,
                compartments=_MEMBRANE_ONLY,
                include_median=include_median,
            )
    else:
        add_masked_summary_stats(
            measurements,
            image_crop,
            ch_names,
            comp_masks,
            compartments=selected_compartments,
            include_median=include_median,
        )
    return comp_masks


def _measure_percentile_family(
    measurements: dict[str, float],
    *,
    cell: CellMatch,
    image_crop: np.ndarray,
    image_tile: np.ndarray | None,
    ch_names: Sequence[str],
    comp_masks: dict[str, np.ndarray],
    indexed_comps: dict[str, np.ndarray] | None,
    percentiles: Sequence[float],
    performance_mode: PerformanceMode,
    cell_mask: np.ndarray,
) -> None:
    if not percentiles or _should_skip_percentiles(performance_mode, cell_mask=cell_mask):
        return
    selected_compartments = _selected_intensity_compartments(
        cell, cell_mask=cell_mask, nuc_mask=comp_masks["NUCLEUS"]
    )
    selected_set = set(selected_compartments)
    if image_tile is not None and indexed_comps is not None:
        indexed_selected = {comp: idx for comp, idx in indexed_comps.items() if comp in selected_set}
        add_indexed_percentiles(measurements, image_tile, ch_names, indexed_selected, percentiles)
        if "MEMBRANE" in selected_set:
            add_masked_percentiles(
                measurements,
                image_crop,
                ch_names,
                comp_masks,
                percentiles,
                compartments=_MEMBRANE_ONLY,
            )
    else:
        add_masked_percentiles(
            measurements,
            image_crop,
            ch_names,
            comp_masks,
            percentiles,
            compartments=selected_compartments,
        )


def _measure_morphology_family(
    measurements: dict[str, float],
    *,
    image_crop: np.ndarray,
    ch_names: Sequence[str],
    comp_masks: dict[str, np.ndarray],
    expansion_image_crop: np.ndarray | None,
    expansion_cell_mask: np.ndarray | None,
    config: MeasurementConfig,
) -> None:
    if config.erosion_enabled:
        add_erosion_measurements(measurements, image_crop, ch_names, comp_masks, n_bins=5)
    if config.expansion_enabled and expansion_image_crop is not None and expansion_cell_mask is not None:
        add_expansion_measurements(
            measurements,
            expansion_image_crop,
            ch_names,
            expansion_cell_mask,
            pixel_size_microns=config.pixel_size_microns,
            n_bins=5,
        )
    if config.environment_expansion_enabled and expansion_image_crop is not None and expansion_cell_mask is not None:
        add_environment_measurements(
            measurements,
            expansion_image_crop,
            ch_names,
            expansion_cell_mask,
            pixel_size_microns=config.pixel_size_microns,
        )


def _expansion_bounds_for_cell(cell: CellMatch, *, image_shape: tuple[int, int], pixel_size_microns: float) -> TileBounds:
    H, W = image_shape
    pad_px = max(1, int(round(20.0 / pixel_size_microns)))
    br0, bc0, br1, bc1 = cell.bbox
    return TileBounds(
        r0=max(0, br0 - pad_px),
        c0=max(0, bc0 - pad_px),
        r1=min(H, br1 + pad_px),
        c1=min(W, bc1 + pad_px),
    )


def _measure_tile_with_context(task: TileTask, context: _WorkerContext) -> TileResult:
    config = context.config
    image_tile = context.data_access.read_tile_image(task.bounds)
    nuc_tile = context.data_access.read_tile_mask("nucleus", task.bounds)
    wc_tile = context.data_access.read_tile_mask("whole_cell", task.bounds)
    tile_cells = [context.cells_by_id[cell_id] for cell_id in task.cell_ids]
    index_cache = build_tile_label_index_cache(tile_cells, wc_tile=wc_tile, nuc_tile=nuc_tile)

    measurements_by_cell: dict[int, dict[str, float]] = {}
    fallback_reads = 0
    baseline_seconds = 0.0
    percentile_seconds = 0.0
    morphology_seconds = 0.0

    for cell in tile_cells:
        cell_bounds = TileBounds.from_bbox(cell.bbox)
        outside_tile = not task.bounds.contains(cell_bounds)

        if outside_tile:
            fallback_reads += 1
            image_crop = context.data_access.read_bbox_image(cell.bbox)
            nuc_crop = context.data_access.read_bbox_mask("nucleus", cell.bbox)
            wc_crop = context.data_access.read_bbox_mask("whole_cell", cell.bbox)
        else:
            row_slice, col_slice = _local_bbox_slices(task.bounds, cell.bbox)
            image_crop = image_tile[:, row_slice, col_slice]
            nuc_crop = nuc_tile[row_slice, col_slice] if nuc_tile is not None else None
            wc_crop = wc_tile[row_slice, col_slice] if wc_tile is not None else None

        cell_mask, nuc_mask = _cell_masks_from_crops(
            cell,
            nuc_crop=nuc_crop,
            wc_crop=wc_crop,
            synth_geoms=context.synth_geoms,
            bbox=cell.bbox,
        )
        if not np.any(cell_mask):
            measurements_by_cell[cell.cell_id] = {}
            continue

        expansion_image_crop: np.ndarray | None = None
        expansion_cell_mask: np.ndarray | None = None
        if config.expansion_enabled or config.environment_expansion_enabled:
            expansion_bounds = _expansion_bounds_for_cell(
                cell,
                image_shape=config.image_shape,
                pixel_size_microns=config.pixel_size_microns,
            )
            if task.bounds.contains(expansion_bounds):
                row_slice, col_slice = _local_bbox_slices(
                    task.bounds,
                    (expansion_bounds.r0, expansion_bounds.c0, expansion_bounds.r1, expansion_bounds.c1),
                )
                expansion_image_crop = image_tile[:, row_slice, col_slice]
                expansion_nuc_crop = nuc_tile[row_slice, col_slice] if nuc_tile is not None else None
                expansion_wc_crop = wc_tile[row_slice, col_slice] if wc_tile is not None else None
            else:
                fallback_reads += 1
                expansion_bbox = (expansion_bounds.r0, expansion_bounds.c0, expansion_bounds.r1, expansion_bounds.c1)
                expansion_image_crop = context.data_access.read_bbox_image(expansion_bbox)
                expansion_nuc_crop = context.data_access.read_bbox_mask("nucleus", expansion_bbox)
                expansion_wc_crop = context.data_access.read_bbox_mask("whole_cell", expansion_bbox)
            expansion_cell_mask, _ = _cell_masks_from_crops(
                cell,
                nuc_crop=expansion_nuc_crop,
                wc_crop=expansion_wc_crop,
                synth_geoms=context.synth_geoms,
                bbox=(expansion_bounds.r0, expansion_bounds.c0, expansion_bounds.r1, expansion_bounds.c1),
            )

        indexed_comps: dict[str, np.ndarray] | None = None
        if not outside_tile:
            indexed_comps = indexed_compartment_indices(cell, cache=index_cache)

        measurements: dict[str, float] = {}
        t_baseline = time.perf_counter()
        comp_masks = _measure_baseline_family(
            measurements,
            cell=cell,
            cell_mask=cell_mask,
            nuc_mask=nuc_mask,
            image_crop=image_crop,
            image_tile=image_tile if indexed_comps is not None else None,
            indexed_comps=indexed_comps,
            ch_names=context.ch_names,
            pixel_size_microns=config.pixel_size_microns,
            performance_mode=config.performance_mode,
        )
        baseline_seconds += time.perf_counter() - t_baseline

        t_percentiles = time.perf_counter()
        _measure_percentile_family(
            measurements,
            cell=cell,
            image_crop=image_crop,
            image_tile=image_tile if indexed_comps is not None else None,
            ch_names=context.ch_names,
            comp_masks=comp_masks,
            indexed_comps=indexed_comps,
            percentiles=config.percentiles,
            performance_mode=config.performance_mode,
            cell_mask=cell_mask,
        )
        percentile_seconds += time.perf_counter() - t_percentiles

        t_morphology = time.perf_counter()
        _measure_morphology_family(
            measurements,
            image_crop=image_crop,
            ch_names=context.ch_names,
            comp_masks=comp_masks,
            expansion_image_crop=expansion_image_crop,
            expansion_cell_mask=expansion_cell_mask,
            config=config,
        )
        morphology_seconds += time.perf_counter() - t_morphology
        measurements_by_cell[cell.cell_id] = measurements

    return TileResult(
        measurements_by_cell=measurements_by_cell,
        fallback_reads=fallback_reads,
        baseline_seconds=baseline_seconds,
        percentile_seconds=percentile_seconds,
        morphology_seconds=morphology_seconds,
    )


def _measure_tile_worker(task: TileTask) -> TileResult:
    context = _active_worker_context()
    return _measure_tile_with_context(task, context)


def _init_process_worker_in_memory(
    image_spec: SharedArraySpec,
    nuc_spec: SharedArraySpec | None,
    wc_spec: SharedArraySpec | None,
    config: MeasurementConfig,
    ch_names: tuple[str, ...],
    synth_geoms: dict[int, Polygon],
    cells_by_id: dict[int, CellMatch],
) -> None:
    global _WORKER_CONTEXT
    global _WORKER_SHARED_HANDLES

    image_arr, image_shm = open_shared_array(image_spec)
    handles: list[SharedMemory] = [image_shm]

    nuc_arr: np.ndarray | None = None
    if nuc_spec is not None:
        nuc_arr, nuc_shm = open_shared_array(nuc_spec)
        handles.append(nuc_shm)

    wc_arr: np.ndarray | None = None
    if wc_spec is not None:
        wc_arr, wc_shm = open_shared_array(wc_spec)
        handles.append(wc_shm)

    _WORKER_SHARED_HANDLES = handles
    _WORKER_CONTEXT = _WorkerContext(
        config=config,
        ch_names=ch_names,
        synth_geoms=synth_geoms,
        cells_by_id=cells_by_id,
        data_access=InMemoryTileDataAccess(image_cyx=image_arr, nuc_labels=nuc_arr, wc_labels=wc_arr),
    )


def _init_process_worker_tiff(
    tiff_spec: TiffDataAccessSpec,
    nuc_spec: SharedArraySpec | None,
    wc_spec: SharedArraySpec | None,
    config: MeasurementConfig,
    ch_names: tuple[str, ...],
    synth_geoms: dict[int, Polygon],
    cells_by_id: dict[int, CellMatch],
) -> None:
    global _WORKER_CONTEXT
    global _WORKER_SHARED_HANDLES

    nuc_arr: np.ndarray | None = None
    handles: list[SharedMemory] = []
    if nuc_spec is not None:
        nuc_arr, nuc_shm = open_shared_array(nuc_spec)
        handles.append(nuc_shm)

    wc_arr: np.ndarray | None = None
    if wc_spec is not None:
        wc_arr, wc_shm = open_shared_array(wc_spec)
        handles.append(wc_shm)

    _WORKER_SHARED_HANDLES = handles
    _WORKER_CONTEXT = _WorkerContext(
        config=config,
        ch_names=ch_names,
        synth_geoms=synth_geoms,
        cells_by_id=cells_by_id,
        data_access=TiffTileDataAccess(tiff_spec=tiff_spec, nuc_labels=nuc_arr, wc_labels=wc_arr),
    )


def _create_shared_image_and_mask_specs(
    image_cyx: np.ndarray,
    nuc_labels: np.ndarray | None,
    wc_labels: np.ndarray | None,
) -> tuple[SharedArraySpec, SharedArraySpec | None, SharedArraySpec | None, list[SharedMemory]]:
    image_spec, image_shm = create_shared_array(image_cyx)
    handles: list[SharedMemory] = [image_shm]

    nuc_spec: SharedArraySpec | None = None
    if nuc_labels is not None:
        nuc_spec, nuc_shm = create_shared_array(nuc_labels)
        handles.append(nuc_shm)

    wc_spec: SharedArraySpec | None = None
    if wc_labels is not None:
        wc_spec, wc_shm = create_shared_array(wc_labels)
        handles.append(wc_shm)

    return image_spec, nuc_spec, wc_spec, handles


def _create_shared_mask_specs(
    nuc_labels: np.ndarray | None,
    wc_labels: np.ndarray | None,
) -> tuple[SharedArraySpec | None, SharedArraySpec | None, list[SharedMemory]]:
    handles: list[SharedMemory] = []

    nuc_spec: SharedArraySpec | None = None
    if nuc_labels is not None:
        nuc_spec, nuc_shm = create_shared_array(nuc_labels)
        handles.append(nuc_shm)

    wc_spec: SharedArraySpec | None = None
    if wc_labels is not None:
        wc_spec, wc_shm = create_shared_array(wc_labels)
        handles.append(wc_shm)

    return nuc_spec, wc_spec, handles


def _close_and_unlink_shared(handles: Sequence[SharedMemory]) -> None:
    for shm in handles:
        try:
            shm.close()
        finally:
            shm.unlink()


def _iter_tile_measurements(
    tasks: Sequence[TileTask],
    *,
    context: _WorkerContext,
    image_cyx: np.ndarray | None,
    nuc_labels: np.ndarray | None,
    wc_labels: np.ndarray | None,
    tiff_spec: TiffDataAccessSpec | None,
) -> Iterator[TileResult]:
    """Yield measured tile results in serial or process-parallel mode."""
    if context.config.workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            yield _measure_tile_with_context(task, context)
        return

    if tiff_spec is None:
        if image_cyx is None:
            raise ValueError("image_cyx must be provided for in-memory process execution")
        image_spec, nuc_spec, wc_spec, shared_handles = _create_shared_image_and_mask_specs(image_cyx, nuc_labels, wc_labels)
        try:
            with ProcessPoolExecutor(
                max_workers=context.config.workers,
                initializer=_init_process_worker_in_memory,
                initargs=(
                    image_spec,
                    nuc_spec,
                    wc_spec,
                    context.config,
                    context.ch_names,
                    context.synth_geoms,
                    context.cells_by_id,
                ),
            ) as executor:
                futures = [executor.submit(_measure_tile_worker, task) for task in tasks]
                for future in as_completed(futures):
                    yield future.result()
        finally:
            _close_and_unlink_shared(shared_handles)
        return

    nuc_spec, wc_spec, shared_handles = _create_shared_mask_specs(nuc_labels, wc_labels)
    try:
        with ProcessPoolExecutor(
            max_workers=context.config.workers,
            initializer=_init_process_worker_tiff,
            initargs=(
                tiff_spec,
                nuc_spec,
                wc_spec,
                context.config,
                context.ch_names,
                context.synth_geoms,
                context.cells_by_id,
            ),
        ) as executor:
            futures = [executor.submit(_measure_tile_worker, task) for task in tasks]
            for future in as_completed(futures):
                yield future.result()
    finally:
        _close_and_unlink_shared(shared_handles)


def _write_measurement_jsonl_row(
    fh: TextIO,
    cell_id: int,
    measurements: dict[str, float],
) -> None:
    """Write one ``{cell_id, measurements}`` record as JSONL."""
    json.dump({"cell_id": cell_id, "measurements": measurements}, fh, separators=(",", ":"))
    fh.write("\n")


def _flush_stream_rows(
    stream_fh: TextIO | None,
    stream_pending: dict[int, dict[str, float]],
    next_stream_id: int,
    tile_result: dict[int, dict[str, float]],
) -> int:
    """Flush in-order JSONL rows from accumulated tile results."""
    if stream_fh is None:
        return next_stream_id
    stream_pending.update(tile_result)
    while next_stream_id in stream_pending:
        _write_measurement_jsonl_row(stream_fh, next_stream_id, stream_pending.pop(next_stream_id))
        next_stream_id += 1
    return next_stream_id


def _finalize_stream(
    stream_fh: TextIO | None,
    *,
    needs_neighbour_aggregation: bool,
    cells: Sequence[CellMatch],
    results: dict[int, dict[str, float]],
    stream_pending: dict[int, dict[str, float]],
) -> None:
    """Write final pending rows and close JSONL stream if open."""
    if stream_fh is None:
        return
    if needs_neighbour_aggregation:
        for cell in sorted(cells, key=lambda c: c.cell_id):
            _write_measurement_jsonl_row(stream_fh, cell.cell_id, results.get(cell.cell_id, {}))
    else:
        for cell_id in sorted(stream_pending):
            _write_measurement_jsonl_row(stream_fh, cell_id, stream_pending[cell_id])
    stream_fh.close()


def measure_cells_tiled(
    cells: Sequence[CellMatch],
    nuc_labels: da.Array | np.ndarray | None,
    wc_labels: da.Array | np.ndarray | None,
    synth_geoms: dict[int, Polygon],
    tiff_file: Path,
    image_shape: tuple[int, int],
    percentiles: Sequence[float] = (),
    tile_size: int = 2048,
    tile_overlap: int = 200,
    threads: int = 1,
    erosion_enabled: bool = True,
    expansion_enabled: bool = True,
    environment_expansion_enabled: bool = False,
    neighbours: int = 0,
    pixel_size_microns: float = 0.5,
    downsample_factor: float = 1.0,
    jsonl_path: Path | None = None,
    return_results: bool = True,
    performance_mode: PerformanceMode = "exact",
) -> dict[int, dict[str, float]]:
    """Compute cell measurements from a TIFF image using tile-owned batching."""
    if not cells:
        return {}
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if tile_overlap < 0:
        raise ValueError("tile_overlap must be >= 0")
    if threads <= 0:
        raise ValueError("threads must be > 0")

    _validate_measure_cells_inputs(neighbours, downsample_factor, performance_mode)
    t_start = time.perf_counter()
    logger.info(
        "Measurement start: cells=%d, workers=%d, tile_size=%d, tile_overlap=%d, neighbours=%d, "
        "downsample_factor=%.3f, mode=%s",
        len(cells),
        threads,
        tile_size,
        tile_overlap,
        neighbours,
        downsample_factor,
        performance_mode,
    )

    image_cyx: np.ndarray | None = None
    tiff_spec: TiffDataAccessSpec | None = None
    use_worker_local_tiff_reads = threads > 1 and downsample_factor <= 1.0
    t_image_load_start = time.perf_counter()
    if use_worker_local_tiff_reads:
        cyx_shape, axes, source_shape, ch_names = inspect_tiff_image(tiff_file)
        if tuple(cyx_shape[1:]) != image_shape:
            raise ValueError(
                f"TIFF image shape {tuple(cyx_shape[1:])} does not match segmentation shape {image_shape}."
            )
        effective_pixel_size = pixel_size_microns
        tiff_spec = TiffDataAccessSpec(
            path=str(tiff_file),
            axes=axes,
            source_shape=source_shape,
            cyx_shape=cyx_shape,
        )
        logger.info(
            "Using worker-local TIFF window reads for process workers (no full image array materialization)."
        )
        logger.info(
            "Measurement image metadata preparation complete in %.2fs: source_shape=%s, normalized_shape=%s, "
            "channels=%d, effective_pixel_size=%.4f",
            time.perf_counter() - t_image_load_start,
            source_shape,
            cyx_shape,
            len(ch_names),
            effective_pixel_size,
        )
    else:
        image_cyx, ch_names, nuc_labels, wc_labels, effective_pixel_size = _prepare_measurement_image_and_masks(
            tiff_file=tiff_file,
            image_shape=image_shape,
            nuc_labels=nuc_labels,
            wc_labels=wc_labels,
            pixel_size_microns=pixel_size_microns,
            downsample_factor=downsample_factor,
        )
        logger.info(
            "Measurement image/mask preparation complete in %.2fs: image_shape=%s, channels=%d, "
            "effective_pixel_size=%.4f",
            time.perf_counter() - t_image_load_start,
            image_cyx.shape,
            len(ch_names),
            effective_pixel_size,
        )

    nuc_labels_np, wc_labels_np = _materialize_labels_once(
        nuc_labels,
        wc_labels,
        expected_shape=image_shape,
    )
    config = MeasurementConfig(
        image_shape=image_shape,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        workers=threads,
        percentiles=tuple(float(p) for p in percentiles),
        erosion_enabled=erosion_enabled,
        expansion_enabled=expansion_enabled,
        environment_expansion_enabled=environment_expansion_enabled,
        neighbours=neighbours,
        pixel_size_microns=effective_pixel_size,
        performance_mode=performance_mode,
    )
    cells_by_id = {cell.cell_id: cell for cell in cells}
    if tiff_spec is not None:
        data_access: TileDataAccess = TiffTileDataAccess(
            tiff_spec=tiff_spec,
            nuc_labels=nuc_labels_np,
            wc_labels=wc_labels_np,
        )
    else:
        if image_cyx is None:
            raise ValueError("image_cyx must be available for in-memory data access")
        data_access = InMemoryTileDataAccess(
            image_cyx=image_cyx,
            nuc_labels=nuc_labels_np,
            wc_labels=wc_labels_np,
        )
    context = _WorkerContext(
        config=config,
        ch_names=tuple(ch_names),
        synth_geoms=synth_geoms,
        cells_by_id=cells_by_id,
        data_access=data_access,
    )

    t_group_start = time.perf_counter()
    tile_groups = _group_cells_by_tile_ids(cells, tile_size=tile_size)
    tasks = _build_tile_tasks(tile_groups, config=config)
    logger.info(
        "Grouped cells into %d tile tasks in %.2fs",
        len(tasks),
        time.perf_counter() - t_group_start,
    )

    needs_neighbour_aggregation = neighbours > 0
    collect_results = return_results or needs_neighbour_aggregation
    results: dict[int, dict[str, float]] = {}
    stream_pending: dict[int, dict[str, float]] = {}
    next_stream_id = min(cell.cell_id for cell in cells)
    stream_fh: TextIO | None = None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        stream_fh = jsonl_path.open("w", encoding="utf-8")

    fallback_reads = 0
    baseline_seconds = 0.0
    percentile_seconds = 0.0
    morphology_seconds = 0.0
    total_tiles = len(tasks)
    progress_interval = max(1, total_tiles // 10) if total_tiles > 0 else 1
    processed_tiles = 0
    t_dispatch_start = time.perf_counter()

    try:
        for tile_result in _iter_tile_measurements(
            tasks,
            context=context,
            image_cyx=image_cyx,
            nuc_labels=nuc_labels_np,
            wc_labels=wc_labels_np,
            tiff_spec=tiff_spec,
        ):
            processed_tiles += 1
            fallback_reads += tile_result.fallback_reads
            baseline_seconds += tile_result.baseline_seconds
            percentile_seconds += tile_result.percentile_seconds
            morphology_seconds += tile_result.morphology_seconds

            if collect_results:
                results.update(tile_result.measurements_by_cell)
            if not needs_neighbour_aggregation:
                next_stream_id = _flush_stream_rows(
                    stream_fh,
                    stream_pending,
                    next_stream_id,
                    tile_result.measurements_by_cell,
                )

            if processed_tiles == 1 or processed_tiles == total_tiles or processed_tiles % progress_interval == 0:
                logger.info(
                    "Measurement tile progress: %d/%d tiles (%.1f%%), elapsed=%.2fs, fallback_reads=%d",
                    processed_tiles,
                    total_tiles,
                    (processed_tiles / total_tiles) * 100.0 if total_tiles else 100.0,
                    time.perf_counter() - t_dispatch_start,
                    fallback_reads,
                )
        logger.info("Measurement tile dispatch complete in %.2fs", time.perf_counter() - t_dispatch_start)

        if needs_neighbour_aggregation:
            t_neighbour_start = time.perf_counter()
            _add_neighbour_measurements(
                measurements_by_cell=results,
                cells=cells,
                neighbours=neighbours,
                pixel_size_microns=effective_pixel_size,
            )
            logger.info("Neighbour aggregation complete in %.2fs", time.perf_counter() - t_neighbour_start)
    finally:
        context.data_access.close()
        t_finalize_start = time.perf_counter()
        _finalize_stream(
            stream_fh,
            needs_neighbour_aggregation=needs_neighbour_aggregation,
            cells=cells,
            results=results,
            stream_pending=stream_pending,
        )
        logger.info("Measurement stream finalization complete in %.2fs", time.perf_counter() - t_finalize_start)

    if fallback_reads > 0:
        logger.info("Tile measurement fallback direct bbox reads: %d", fallback_reads)
    logger.info(
        "Measurement family timings: baseline=%.2fs, percentiles=%.2fs, morphology=%.2fs",
        baseline_seconds,
        percentile_seconds,
        morphology_seconds,
    )
    logger.info(
        "Measurement complete in %.2fs: measured_cells=%d, return_results=%s",
        time.perf_counter() - t_start,
        len(results) if collect_results else len(cells),
        return_results,
    )
    return results if return_results else {}
