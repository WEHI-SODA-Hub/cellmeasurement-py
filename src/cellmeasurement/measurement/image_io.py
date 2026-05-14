from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile

logger = logging.getLogger(__name__)
OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def _normalize_image_cyx(arr: np.ndarray, axes: str | None = None) -> np.ndarray:
    """Normalize loaded TIFF data to (C, Y, X)."""
    if axes is not None and len(axes) == arr.ndim:
        work = np.asarray(arr)
        axis_list = list(axes)

        # Drop singleton axes not used for channel/spatial dimensions.
        for idx in range(len(axis_list) - 1, -1, -1):
            ax = axis_list[idx]
            if ax in {"C", "S", "Q", "Y", "X"}:
                continue
            if work.shape[idx] != 1:
                raise ValueError(
                    f"Unsupported TIFF layout: non-singleton axis '{ax}' in shape {work.shape} (axes='{axes}')"
                )
            work = np.take(work, 0, axis=idx)
            axis_list.pop(idx)

        channel_axis = axis_list.index("C") if "C" in axis_list else \
            (axis_list.index("S") if "S" in axis_list else (axis_list.index("Q") if "Q" in axis_list else None))
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


def _channel_names_from_ome(tf: tifffile.TiffFile) -> list[str]:
    """Extract channel names from OME-XML metadata when available."""
    ome_text = tf.ome_metadata
    if not ome_text and tf.pages:
        first_desc = tf.pages[0].description
        if isinstance(first_desc, str) and first_desc.strip().startswith("<"):
            ome_text = first_desc
    if not ome_text:
        return []

    root = ET.fromstring(ome_text)
    ns = {"ome": OME_NS}
    channels = root.findall(".//ome:Channel", ns)
    out = []
    for ch in channels:
        name = ch.get("Name") or ch.get("ID") or ""
        out.append(str(name))
    return [name for name in out if name]


def _channel_names_from_mibi_json(tf: tifffile.TiffFile) -> list[str]:
    """Extract channel names from per-page MIBI JSON descriptions."""
    if not tf.pages:
        return []
    try:
        first_desc = tf.pages[0].description
        if not isinstance(first_desc, str):
            return []
        json.loads(first_desc)  # probe JSON format
        out: list[str] = []
        for page in tf.pages:
            desc = page.description
            if not isinstance(desc, str):
                return []
            parsed = json.loads(desc)
            name = parsed.get("channel.target", "")
            out.append(str(name))
        return [name for name in out if name]
    except (ValueError, TypeError, AttributeError):
        return []


def _is_mibi_tiff(tf: tifffile.TiffFile) -> bool:
    """Detect MIBI-style TIFF by probing first-page JSON for ``channel.target``."""
    if not tf.pages:
        return False
    first_desc = tf.pages[0].description
    if not isinstance(first_desc, str):
        return False
    try:
        parsed = json.loads(first_desc)
    except (ValueError, TypeError):
        return False
    return isinstance(parsed, dict) and "channel.target" in parsed


def _parse_ome_root(tf: tifffile.TiffFile) -> ET.Element | None:
    """Return parsed OME-XML root, or ``None`` if unavailable/unparseable."""
    ome_text = tf.ome_metadata
    if not ome_text and tf.pages:
        first_desc = tf.pages[0].description
        if isinstance(first_desc, str) and first_desc.strip().startswith("<"):
            ome_text = first_desc
    if not ome_text:
        return None
    try:
        return ET.fromstring(ome_text)
    except ET.ParseError:
        return None


def _is_comet_tiff(tf: tifffile.TiffFile) -> bool:
    """Detect Lunaphore COMET TIFFs from OME instrument metadata."""
    root = _parse_ome_root(tf)
    if root is None:
        return False
    ns = {"ome": OME_NS}

    def _norm(value: str | None) -> str:
        return "" if value is None else value.strip().lower()

    microscope_hits = False
    for microscope in root.findall(".//ome:Microscope", ns):
        manufacturer = _norm(microscope.get("Manufacturer"))
        model = _norm(microscope.get("Model"))
        if "lunaphore" in manufacturer and "comet" in model:
            microscope_hits = True
            break

    detector_hits = False
    for detector in root.findall(".//ome:Detector", ns):
        manufacturer = _norm(detector.get("Manufacturer"))
        model = _norm(detector.get("Model"))
        if "lunaphore" in manufacturer and "comet" in model:
            detector_hits = True
            break

    objective_hits = False
    for objective in root.findall(".//ome:Objective", ns):
        manufacturer = _norm(objective.get("Manufacturer"))
        model = _norm(objective.get("Model"))
        if "lunaphore" in manufacturer and "comet" in model:
            objective_hits = True
            break

    return microscope_hits or detector_hits or objective_hits


def _select_non_mibi_fullres_pages(tf: tifffile.TiffFile) -> list[tifffile.TiffPage]:
    """Select only full-resolution pages for non-MIBI multi-page TIFFs."""
    if not tf.pages:
        return []
    first_shape = tuple(int(v) for v in tf.pages[0].shape)
    selected = [page for page in tf.pages if tuple(int(v) for v in page.shape) == first_shape]
    return selected if selected else [tf.pages[0]]


def _channel_names_from_imagej(tf: tifffile.TiffFile) -> list[str]:
    """Extract channel names from ImageJ Labels metadata."""
    ij = tf.imagej_metadata
    if not ij or "Labels" not in ij:
        return []
    return [str(lbl) for lbl in ij["Labels"] if str(lbl)]


def _extract_channel_names(tf: tifffile.TiffFile) -> list[str]:
    """Extract channel names using OME, MIBI JSON, then ImageJ metadata strategies."""
    try:
        names = _channel_names_from_ome(tf)
        if names:
            return names
    except Exception as exc:
        logger.debug("Failed OME channel extraction: %s", exc)

    names = _channel_names_from_mibi_json(tf)
    if names:
        return names

    names = _channel_names_from_imagej(tf)
    if names:
        return names

    return []


def _load_tiff_image(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load TIFF intensity image and return (C, Y, X) data with channel names."""
    t0 = time.perf_counter()
    axes: str | None = None
    ch_names: list[str] = []
    mode = "unknown"
    total_pages = 0
    selected_pages = 0
    try:
        with tifffile.TiffFile(path) as tf:
            total_pages = len(tf.pages)
            if tf.series:
                axes = tf.series[0].axes
            if _is_mibi_tiff(tf):
                mode = "mibi-pages"
                arr = np.stack([page.asarray() for page in tf.pages], axis=0)
                selected_pages = len(tf.pages)
                ch_names = _channel_names_from_mibi_json(tf)
            elif _is_comet_tiff(tf):
                mode = "comet-memmap"
                try:
                    arr = tifffile.memmap(path)
                except (ValueError, OSError, tifffile.TiffFileError, NotImplementedError):
                    logger.warning(
                        "tifffile.memmap failed for COMET TIFF %s; falling back to tifffile.imread.",
                        path,
                    )
                    arr = tifffile.imread(path)
                ch_names = _extract_channel_names(tf)
            elif len(tf.pages) > 1:
                mode = "multipage-fullres"
                selected = _select_non_mibi_fullres_pages(tf)
                selected_pages = len(selected)
                arr = np.stack([page.asarray() for page in selected], axis=0)
                try:
                    ch_names = _channel_names_from_ome(tf)
                except Exception as exc:
                    logger.debug("Failed OME channel extraction: %s", exc)
                    ch_names = []
                if not ch_names:
                    ch_names = _channel_names_from_imagej(tf)
            else:
                mode = "single-page"
                arr = tf.asarray()
                selected_pages = 1
                ch_names = _extract_channel_names(tf)
    except Exception:
        axes = None
        arr = tifffile.imread(path)
        mode = "imread-fallback"

    image_cyx = _normalize_image_cyx(np.asarray(arr), axes=axes)
    n_channels = int(image_cyx.shape[0])
    if not ch_names or len(ch_names) != n_channels:
        if ch_names and len(ch_names) != n_channels:
            logger.warning(
                "Found %d channel names but TIFF has %d channels; using fallback names.",
                len(ch_names),
                n_channels,
            )
        ch_names = [f"Channel {i + 1}" for i in range(n_channels)]
    logger.info(
        "Loaded TIFF (%s): mode=%s, pages=%d, selected=%d, channels=%d, elapsed=%.2fs",
        path,
        mode,
        total_pages,
        selected_pages,
        n_channels,
        time.perf_counter() - t0,
    )
    return image_cyx, ch_names
