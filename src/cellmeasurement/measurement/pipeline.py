from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import dask.array as da
import numpy as np
import tifffile
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from skimage.draw import polygon as draw_polygon
from skimage.measure import regionprops
from skimage.morphology import disk

from ..segmentation.cell import CellMatch

logger = logging.getLogger(__name__)

# Pre-computed 3x3 disk structuring element used throughout for single-pixel
# morphological erosion/dilation operations.  Frozen to prevent accidental mutation.
_DISK_1 = disk(1).astype(bool)  # type: ignore[assignment]
_DISK_1.flags.writeable = False


def _normalize_image_cyx(arr: np.ndarray, axes: str | None = None) -> np.ndarray:
    """Normalize loaded TIFF data to (C, Y, X)."""
    if axes is not None and len(axes) == arr.ndim:
        work = np.asarray(arr)
        axis_list = list(axes)

        # Drop singleton axes not used for channel/spatial dimensions.
        for idx in range(len(axis_list) - 1, -1, -1):
            ax = axis_list[idx]
            if ax in {"C", "S", "Y", "X"}:
                continue
            if work.shape[idx] != 1:
                raise ValueError(
                    f"Unsupported TIFF layout: non-singleton axis '{ax}' in shape {work.shape} (axes='{axes}')"
                )
            work = np.take(work, 0, axis=idx)
            axis_list.pop(idx)

        channel_axis = axis_list.index("C") if "C" in axis_list else \
            (axis_list.index("S") if "S" in axis_list else None)
        if "Y" not in axis_list or "X" not in axis_list:
            raise ValueError(f"Could not find Y/X axes in TIFF layout (axes='{axes}', shape={arr.shape})")
        y_axis = axis_list.index("Y")
        x_axis = axis_list.index("X")

        if channel_axis is None:
            work = np.transpose(work, (y_axis, x_axis))
            return work[np.newaxis, ...]

        work = np.transpose(work, (channel_axis, y_axis, x_axis))
        return work

    # Heuristic fallback when axis metadata is unavailable.
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF image dimensions: shape={arr.shape}")
    if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= arr.shape[2]:
        return arr
    if arr.shape[2] <= arr.shape[0] and arr.shape[2] <= arr.shape[1]:
        return np.moveaxis(arr, 2, 0)
    raise ValueError(f"Unsupported 3D TIFF image layout: shape={arr.shape}")


def _load_tiff_image(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load TIFF intensity image and return (C, Y, X) data with channel names."""
    axes: str | None = None
    try:
        with tifffile.TiffFile(path) as tf:
            if tf.series:
                axes = tf.series[0].axes
    except Exception:
        axes = None

    try:
        arr = tifffile.memmap(path)
    except (ValueError, OSError, tifffile.TiffFileError, NotImplementedError):
        logger.warning(
            "tifffile.memmap failed for %s; falling back to tifffile.imread (higher memory usage).",
            path,
        )
        arr = tifffile.imread(path)

    image_cyx = _normalize_image_cyx(np.asarray(arr), axes=axes)
    ch_names = [f"Channel {i + 1}" for i in range(int(image_cyx.shape[0]))]
    return image_cyx, ch_names


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


def _basic_shape_metrics(cell_mask: np.ndarray, nuc_mask: np.ndarray) -> dict[str, float]:
    """Compute cell/nucleus shape metrics from binary masks."""
    r = _largest_region(cell_mask)
    if r is None:
        return {}

    perimeter = float(r.perimeter) if r.perimeter > 0 else 0.0
    circularity = float(4 * math.pi * r.area / (r.perimeter**2)) if r.perimeter > 0 else 0.0
    out: dict[str, float] = {
        "Cell: Area px": float(r.area),
        "Cell: Circularity": circularity,
        "Cell: Length px": perimeter,
        "Cell: Max diameter px": _axis_major_length(r),
        "Cell: Min diameter px": _axis_minor_length(r),
        "Cell: Solidity": float(r.solidity) if r.solidity is not None else 0.0,
    }

    nr = _largest_region(nuc_mask)
    if nr is not None:
        n_area = float(nr.area)
        out["Nucleus: Area px"] = n_area
        out["Nucleus: Circularity"] = float(4 * math.pi * nr.area / (nr.perimeter**2)) if nr.perimeter > 0 else 0.0
        out["Nucleus: Length px"] = float(nr.perimeter) if nr.perimeter > 0 else 0.0
        out["Nucleus: Max diameter px"] = _axis_major_length(nr)
        out["Nucleus: Min diameter px"] = _axis_minor_length(nr)
        out["Nucleus: Solidity"] = float(nr.solidity) if nr.solidity is not None else 0.0
        out["Nucleus/Cell area ratio"] = n_area / float(r.area) if r.area > 0 else 0.0

    return out


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
    """Populate standard intensity summary stats for each channel and compartment."""
    labels = {"CELL": "Cell", "NUCLEUS": "Nucleus", "CYTOPLASM": "Cytoplasm", "MEMBRANE": "Membrane"}
    for ci, ch in enumerate(ch_names):
        ch_img = image_cyx[ci]
        for comp, mask in comp_masks.items():
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
    """Populate configured percentiles for each channel and compartment."""
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
    """Populate per-bin erosion area/depth and channel intensity measurements."""
    for comp in ("CELL", "NUCLEUS"):
        base = comp_masks[comp]
        base_area = int(np.count_nonzero(base))
        if base_area == 0:
            continue

        comp_name = comp.capitalize()
        bin_boundaries = _erosion_bins_for_mask(base, n_bins=n_bins)
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
    synth_geoms: dict[int, Polygon],
    percentiles: Sequence[float],
) -> dict[str, float]:
    """Compute all measurement families for a single cell crop."""
    cell_mask, nuc_mask = _cell_masks_from_crops(cell, nuc_crop, wc_crop, synth_geoms, bbox)
    if not np.any(cell_mask):
        return {}

    measurements: dict[str, float] = {}
    measurements.update(_basic_shape_metrics(cell_mask, nuc_mask))
    comps = _compartment_masks(cell_mask, nuc_mask)
    channel_names = [f"Channel {i + 1}" for i in range(int(image_crop.shape[0]))]
    _add_intensity_measurements(measurements, image_crop, channel_names, comps)
    _add_percentiles(measurements, image_crop, channel_names, comps, percentiles)
    _add_erosion_measurements(measurements, image_crop, channel_names, comps, n_bins=5)
    return measurements


def _measure_tile(
    tile_key: tuple[int, int],
    tile_cells: Sequence[CellMatch],
    image_cyx: np.ndarray,
    nuc_labels: da.Array | None,
    wc_labels: da.Array | None,
    image_shape: tuple[int, int],
    tile_size: int,
    tile_overlap: int,
    synth_geoms: dict[int, Polygon],
    percentiles: Sequence[float],
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

        measurement = _measure_single_cell(
            cell=cell,
            image_crop=image_crop,
            nuc_crop=nuc_crop,
            wc_crop=wc_crop,
            bbox=bbox,
            synth_geoms=synth_geoms,
            percentiles=percentiles,
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
        Mapping from ``cell_id`` to synthesized cell polygons for
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

    image_cyx, _ = _load_tiff_image(tiff_file)
    if image_cyx.shape[1:] != image_shape:
        raise ValueError(
            f"TIFF image shape {tuple(image_cyx.shape[1:])} does not match segmentation shape {image_shape}."
        )

    tile_groups = _group_cells_by_tile(cells, tile_size=tile_size)
    results: dict[int, dict[str, float]] = {} if return_results else {}
    stream_pending: dict[int, dict[str, float]] = {}
    next_stream_id = min(cell.cell_id for cell in cells)
    stream_fh = None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        stream_fh = jsonl_path.open("w", encoding="utf-8")

    def _flush_stream_rows(tile_result: dict[int, dict[str, float]]) -> None:
        nonlocal next_stream_id
        if stream_fh is None:
            return
        stream_pending.update(tile_result)
        while next_stream_id in stream_pending:
            _write_measurement_jsonl_row(stream_fh, next_stream_id, stream_pending.pop(next_stream_id))
            next_stream_id += 1

    fallback_reads = 0

    try:
        if threads <= 1 or len(tile_groups) <= 1:
            for key, group in tile_groups.items():
                tile_result, tile_fallback = _measure_tile(
                    tile_key=key,
                    tile_cells=group,
                    image_cyx=image_cyx,
                    nuc_labels=nuc_labels,
                    wc_labels=wc_labels,
                    image_shape=image_shape,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    synth_geoms=synth_geoms,
                    percentiles=percentiles,
                )
                if return_results:
                    results.update(tile_result)
                _flush_stream_rows(tile_result)
                fallback_reads += tile_fallback
        else:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                future_map = {
                    executor.submit(
                        _measure_tile,
                        key,
                        group,
                        image_cyx,
                        nuc_labels,
                        wc_labels,
                        image_shape,
                        tile_size,
                        tile_overlap,
                        synth_geoms,
                        percentiles,
                    ): key
                    for key, group in tile_groups.items()
                }
                for future in as_completed(future_map):
                    tile_result, tile_fallback = future.result()
                    if return_results:
                        results.update(tile_result)
                    _flush_stream_rows(tile_result)
                    fallback_reads += tile_fallback
    finally:
        if stream_fh is not None:
            # Flush any unresolved IDs (e.g. non-contiguous id sets).
            for cell_id in sorted(stream_pending):
                _write_measurement_jsonl_row(stream_fh, cell_id, stream_pending[cell_id])
            stream_fh.close()

    if fallback_reads > 0:
        logger.info("Tile measurement fallback direct bbox reads: %d", fallback_reads)
    return results
