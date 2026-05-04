from __future__ import annotations

import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Sequence, TextIO

import dask.array as da
import numpy as np
from shapely.geometry import Polygon
from skimage.draw import polygon as draw_polygon

from ..segmentation.cell import CellMatch
from .compartment_metrics import (
    _add_environment_measurements,
    _add_erosion_measurements,
    _add_expansion_measurements,
    _add_intensity_measurements,
    _add_percentiles,
    _compartment_masks,
)
from .image_io import _load_tiff_image
from .neighbour_metrics import _add_neighbour_measurements
from .shape_metrics import _basic_shape_metrics

logger = logging.getLogger(__name__)

# Remaining functions in this module orchestrate tiled execution and streaming;
# lower-level shape/compartment/neighbour metric kernels are split into
# dedicated sibling modules for maintainability.


def _polygon_to_local_mask(poly: Polygon, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Rasterize a global polygon into a local bbox mask."""
    r0, c0, r1, c1 = bbox
    h = max(0, r1 - r0)
    w = max(0, c1 - c0)
    mask = np.zeros((h, w), dtype=bool)
    if h == 0 or w == 0:
        return mask
    if poly.is_empty:
        return mask

    ext = np.array(poly.exterior.coords)
    rr, cc = draw_polygon(ext[:, 1] - r0, ext[:, 0] - c0, shape=mask.shape)
    mask[rr, cc] = True
    for interior in poly.interiors:
        hole = np.array(interior.coords)
        hr, hc = draw_polygon(hole[:, 1] - r0, hole[:, 0] - c0, shape=mask.shape)
        mask[hr, hc] = False
    return mask


def _group_cells_by_tile(cells: Sequence[CellMatch], tile_size: int) -> dict[tuple[int, int], list[CellMatch]]:
    """Group cells by centroid-owned tile coordinates."""
    groups: dict[tuple[int, int], list[CellMatch]] = defaultdict(list)
    for cell in cells:
        row, col = cell.centroid
        tile_row = int(row // tile_size)
        tile_col = int(col // tile_size)
        groups[(tile_row, tile_col)].append(cell)
    return groups


def _slice_or_compute(
    arr: da.Array | None,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> np.ndarray | None:
    """Materialize a dask label slice as a NumPy array."""
    if arr is None:
        return None
    return np.asarray(arr[r0:r1, c0:c1].compute())


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


def _measure_single_cell(
    cell: CellMatch,
    image_crop: np.ndarray,
    nuc_crop: np.ndarray | None,
    wc_crop: np.ndarray | None,
    bbox: tuple[int, int, int, int],
    expansion_image_crop: np.ndarray | None,
    expansion_nuc_crop: np.ndarray | None,
    expansion_wc_crop: np.ndarray | None,
    expansion_bbox: tuple[int, int, int, int] | None,
    synth_geoms: dict[int, Polygon],
    percentiles: Sequence[float],
    ch_names: Sequence[str],
    erosion_enabled: bool,
    expansion_enabled: bool,
    environment_expansion_enabled: bool,
    pixel_size_microns: float,
) -> dict[str, float]:
    """Compute all measurement families for a single cell crop."""
    cell_mask, nuc_mask = _cell_masks_from_crops(cell, nuc_crop, wc_crop, synth_geoms, bbox)
    if not np.any(cell_mask):
        return {}

    measurements: dict[str, float] = {}
    measurements.update(_basic_shape_metrics(cell_mask, nuc_mask, pixel_size_microns=pixel_size_microns))
    comps = _compartment_masks(cell_mask, nuc_mask)
    _add_intensity_measurements(measurements, image_crop, ch_names, comps)
    _add_percentiles(measurements, image_crop, ch_names, comps, percentiles)
    if erosion_enabled:
        _add_erosion_measurements(measurements, image_crop, ch_names, comps, n_bins=5)
    if (expansion_enabled or environment_expansion_enabled) and expansion_image_crop is not None and expansion_bbox is not None:
        expansion_cell_mask, _ = _cell_masks_from_crops(
            cell,
            expansion_nuc_crop,
            expansion_wc_crop,
            synth_geoms,
            expansion_bbox,
        )
        if expansion_enabled:
            _add_expansion_measurements(
                measurements,
                expansion_image_crop,
                ch_names,
                expansion_cell_mask,
                pixel_size_microns=pixel_size_microns,
                n_bins=5,
            )
        if environment_expansion_enabled:
            _add_environment_measurements(
                measurements,
                expansion_image_crop,
                ch_names,
                expansion_cell_mask,
                pixel_size_microns=pixel_size_microns,
            )
    return measurements


def _measure_tile(
    tile_key: tuple[int, int],
    tile_cells: Sequence[CellMatch],
    image_cyx: np.ndarray,
    ch_names: Sequence[str],
    nuc_labels: da.Array | None,
    wc_labels: da.Array | None,
    image_shape: tuple[int, int],
    tile_size: int,
    tile_overlap: int,
    synth_geoms: dict[int, Polygon],
    percentiles: Sequence[float],
    erosion_enabled: bool,
    expansion_enabled: bool,
    environment_expansion_enabled: bool,
    pixel_size_microns: float,
) -> tuple[dict[int, dict[str, float]], int]:
    """Measure all cells owned by a single tile and count fallback reads."""
    H, W = image_shape
    tile_row, tile_col = tile_key
    r0 = max(0, tile_row * tile_size - tile_overlap)
    c0 = max(0, tile_col * tile_size - tile_overlap)
    r1 = min(H, (tile_row + 1) * tile_size + tile_overlap)
    c1 = min(W, (tile_col + 1) * tile_size + tile_overlap)

    image_tile = image_cyx[:, r0:r1, c0:c1]
    nuc_tile = _slice_or_compute(nuc_labels, r0, r1, c0, c1)
    wc_tile = _slice_or_compute(wc_labels, r0, r1, c0, c1)

    results: dict[int, dict[str, float]] = {}
    fallback_reads = 0
    for cell in tile_cells:
        br0, bc0, br1, bc1 = cell.bbox
        outside_tile = br0 < r0 or bc0 < c0 or br1 > r1 or bc1 > c1
        if outside_tile:
            fallback_reads += 1
            image_crop = image_cyx[:, br0:br1, bc0:bc1]
            nuc_crop = _slice_or_compute(nuc_labels, br0, br1, bc0, bc1)
            wc_crop = _slice_or_compute(wc_labels, br0, br1, bc0, bc1)
            bbox = (br0, bc0, br1, bc1)
        else:
            sr0 = br0 - r0
            sc0 = bc0 - c0
            sr1 = br1 - r0
            sc1 = bc1 - c0
            image_crop = image_tile[:, sr0:sr1, sc0:sc1]
            nuc_crop = nuc_tile[sr0:sr1, sc0:sc1] if nuc_tile is not None else None
            wc_crop = wc_tile[sr0:sr1, sc0:sc1] if wc_tile is not None else None
            bbox = (br0, bc0, br1, bc1)

        expansion_image_crop: np.ndarray | None = None
        expansion_nuc_crop: np.ndarray | None = None
        expansion_wc_crop: np.ndarray | None = None
        expansion_bbox: tuple[int, int, int, int] | None = None
        if expansion_enabled or environment_expansion_enabled:
            pad_px = max(1, int(round(20.0 / pixel_size_microns)))
            er0 = max(0, br0 - pad_px)
            ec0 = max(0, bc0 - pad_px)
            er1 = min(H, br1 + pad_px)
            ec1 = min(W, bc1 + pad_px)
            expansion_bbox = (er0, ec0, er1, ec1)
            expansion_outside_tile = er0 < r0 or ec0 < c0 or er1 > r1 or ec1 > c1
            if expansion_outside_tile:
                fallback_reads += 1
                expansion_image_crop = image_cyx[:, er0:er1, ec0:ec1]
                expansion_nuc_crop = _slice_or_compute(nuc_labels, er0, er1, ec0, ec1)
                expansion_wc_crop = _slice_or_compute(wc_labels, er0, er1, ec0, ec1)
            else:
                esr0 = er0 - r0
                esc0 = ec0 - c0
                esr1 = er1 - r0
                esc1 = ec1 - c0
                expansion_image_crop = image_tile[:, esr0:esr1, esc0:esc1]
                expansion_nuc_crop = nuc_tile[esr0:esr1, esc0:esc1] if nuc_tile is not None else None
                expansion_wc_crop = wc_tile[esr0:esr1, esc0:esc1] if wc_tile is not None else None

        measurement = _measure_single_cell(
            cell=cell,
            image_crop=image_crop,
            nuc_crop=nuc_crop,
            wc_crop=wc_crop,
            bbox=bbox,
            expansion_image_crop=expansion_image_crop,
            expansion_nuc_crop=expansion_nuc_crop,
            expansion_wc_crop=expansion_wc_crop,
            expansion_bbox=expansion_bbox,
            synth_geoms=synth_geoms,
            percentiles=percentiles,
            ch_names=ch_names,
            erosion_enabled=erosion_enabled,
            expansion_enabled=expansion_enabled,
            environment_expansion_enabled=environment_expansion_enabled,
            pixel_size_microns=pixel_size_microns,
        )
        results[cell.cell_id] = measurement

    return results, fallback_reads


def _write_measurement_jsonl_row(
    fh,
    cell_id: int,
    measurements: dict[str, float],
) -> None:
    """Write one `{cell_id, measurements}` record as JSONL."""
    json.dump({"cell_id": cell_id, "measurements": measurements}, fh, separators=(",", ":"))
    fh.write("\n")


def _validate_measure_cells_inputs(neighbours: int, downsample_factor: float) -> None:
    """Validate public input arguments for tiled measurement."""
    if neighbours < 0:
        raise ValueError("neighbours must be >= 0")
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be > 0")


def _prepare_measurement_image_and_masks(
    tiff_file: Path,
    image_shape: tuple[int, int],
    nuc_labels: da.Array | None,
    wc_labels: da.Array | None,
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
        from ..io.image_loading import maybe_downsample

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


def _tile_measure_kwargs(
    *,
    image_cyx: np.ndarray,
    ch_names: Sequence[str],
    nuc_labels: da.Array | np.ndarray | None,
    wc_labels: da.Array | np.ndarray | None,
    image_shape: tuple[int, int],
    tile_size: int,
    tile_overlap: int,
    synth_geoms: dict[int, Polygon],
    percentiles: Sequence[float],
    erosion_enabled: bool,
    expansion_enabled: bool,
    environment_expansion_enabled: bool,
    pixel_size_microns: float,
) -> dict[str, object]:
    """Build kwargs passed unchanged to each tile measurement task."""
    return {
        "image_cyx": image_cyx,
        "ch_names": ch_names,
        "nuc_labels": nuc_labels,
        "wc_labels": wc_labels,
        "image_shape": image_shape,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "synth_geoms": synth_geoms,
        "percentiles": percentiles,
        "erosion_enabled": erosion_enabled,
        "expansion_enabled": expansion_enabled,
        "environment_expansion_enabled": environment_expansion_enabled,
        "pixel_size_microns": pixel_size_microns,
    }


def _iter_tile_measurements(
    tile_groups: dict[tuple[int, int], list[CellMatch]],
    threads: int,
    tile_kwargs: dict[str, object],
) -> Iterator[tuple[dict[int, dict[str, float]], int]]:
    """Yield measured tile results in serial or parallel mode."""
    if threads <= 1 or len(tile_groups) <= 1:
        for key, group in tile_groups.items():
            yield _measure_tile(tile_key=key, tile_cells=group, **tile_kwargs)
        return

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(_measure_tile, tile_key=key, tile_cells=group, **tile_kwargs)
            for key, group in tile_groups.items()
        ]
        for future in as_completed(futures):
            yield future.result()


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
        # Flush any unresolved IDs (e.g. non-contiguous id sets).
        for cell_id in sorted(stream_pending):
            _write_measurement_jsonl_row(stream_fh, cell_id, stream_pending[cell_id])
    stream_fh.close()


def measure_cells_tiled(
    cells: Sequence[CellMatch],
    nuc_labels: da.Array | None,
    wc_labels: da.Array | None,
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
) -> dict[int, dict[str, float]]:
    """Compute cell measurements from a TIFF image using tile-owned batching.

    The function groups cells by centroid-owned tiles, reads each tile window
    once (plus any fallback per-cell reads when a bbox falls outside tile
    coverage), computes per-cell measurements, and optionally streams results
    to JSONL in deterministic cell-id order.

    Parameters
    ----------
    cells:
        Matched cells to measure.
    nuc_labels:
        Nuclear label array (or ``None`` in whole-cell-only mode).
    wc_labels:
        Whole-cell label array (or ``None`` in nuclear-only mode).
    synth_geoms:
        Mapping from ``cell_id`` to synthesised cell polygons for
        ``match_source="watershed_synth"`` cells.
    tiff_file:
        Path to intensity TIFF image; loaded and normalized to ``(C, Y, X)``.
    image_shape:
        Expected segmentation shape ``(H, W)`` used for input validation.
    percentiles:
        Optional percentile values to compute per channel and compartment.
    tile_size:
        Nominal tile edge length in pixels.
    tile_overlap:
        Extra pixels added around each tile read window.
    threads:
        Number of tile workers. Values ``<=1`` run serially.
    erosion_enabled:
        Whether to compute equal-area erosion-bin measurements.
    expansion_enabled:
        Whether to compute equal-area expansion-bin measurements.
    environment_expansion_enabled:
        Whether to compute 20 µm pericellular environment measurements.
    neighbours:
        Number of nearest neighbours for measurement aggregation (0 disables).
    pixel_size_microns:
        Pixel size used for converting the fixed 20 µm expansion radius to pixels
        and for µm-scaled shape metrics. When downsampling is applied, effective
        pixel size increases proportionally.
    downsample_factor:
        Optional downsampling factor (e.g. 2.0, 4.0) to reduce memory usage.
        Values <= 1.0 or that round to step < 2 result in no downsampling.
    jsonl_path:
        Optional output path for streamed JSONL rows shaped as
        ``{"cell_id": int, "measurements": {...}}``.
    return_results:
        If ``True``, return in-memory ``{cell_id: measurements}``.
        If ``False``, still performs measurement work and JSONL streaming but
        returns an empty dict.

    Returns
    -------
    dict[int, dict[str, float]]
        Per-cell measurement mapping when ``return_results=True``; otherwise
        an empty dict.

    Raises
    ------
    ValueError
        If TIFF spatial shape does not match ``image_shape``.
    """
    if not cells:
        return {}
    _validate_measure_cells_inputs(neighbours, downsample_factor)

    image_cyx, ch_names, nuc_labels, wc_labels, effective_pixel_size = _prepare_measurement_image_and_masks(
        tiff_file=tiff_file,
        image_shape=image_shape,
        nuc_labels=nuc_labels,
        wc_labels=wc_labels,
        pixel_size_microns=pixel_size_microns,
        downsample_factor=downsample_factor,
    )

    tile_groups = _group_cells_by_tile(cells, tile_size=tile_size)
    needs_neighbour_aggregation = neighbours > 0
    collect_results = return_results or needs_neighbour_aggregation
    results: dict[int, dict[str, float]] = {}
    stream_pending: dict[int, dict[str, float]] = {}
    next_stream_id = min(cell.cell_id for cell in cells)
    stream_fh: TextIO | None = None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        stream_fh = jsonl_path.open("w", encoding="utf-8")

    tile_kwargs = _tile_measure_kwargs(
        image_cyx=image_cyx,
        ch_names=ch_names,
        nuc_labels=nuc_labels,
        wc_labels=wc_labels,
        image_shape=image_shape,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        synth_geoms=synth_geoms,
        percentiles=percentiles,
        erosion_enabled=erosion_enabled,
        expansion_enabled=expansion_enabled,
        environment_expansion_enabled=environment_expansion_enabled,
        pixel_size_microns=effective_pixel_size,
    )
    fallback_reads = 0

    try:
        for tile_result, tile_fallback in _iter_tile_measurements(tile_groups, threads, tile_kwargs):
            if collect_results:
                results.update(tile_result)
            if not needs_neighbour_aggregation:
                next_stream_id = _flush_stream_rows(stream_fh, stream_pending, next_stream_id, tile_result)
            fallback_reads += tile_fallback

        if needs_neighbour_aggregation:
            _add_neighbour_measurements(
                measurements_by_cell=results,
                cells=cells,
                neighbours=neighbours,
                pixel_size_microns=effective_pixel_size,
            )
    finally:
        _finalize_stream(
            stream_fh,
            needs_neighbour_aggregation=needs_neighbour_aggregation,
            cells=cells,
            results=results,
            stream_pending=stream_pending,
        )

    if fallback_reads > 0:
        logger.info("Tile measurement fallback direct bbox reads: %d", fallback_reads)
    return results if return_results else {}
