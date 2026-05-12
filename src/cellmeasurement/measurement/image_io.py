from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile

logger = logging.getLogger(__name__)


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
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
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
    axes: str | None = None
    ch_names: list[str] = []
    try:
        with tifffile.TiffFile(path) as tf:
            if _is_mibi_tiff(tf):
                arr = np.stack([page.asarray() for page in tf.pages], axis=0)
                ch_names = _channel_names_from_mibi_json(tf)
            elif len(tf.pages) > 1:
                # Treat multi-page non-MIBI TIFFs (e.g. OPAL/QPTIFF) as channel-stacked.
                arr = np.stack([page.asarray() for page in tf.pages], axis=0)
                try:
                    ch_names = _channel_names_from_ome(tf)
                except Exception as exc:
                    logger.debug("Failed OME channel extraction: %s", exc)
                    ch_names = []
                if not ch_names:
                    ch_names = _channel_names_from_imagej(tf)
            else:
                if tf.series:
                    axes = tf.series[0].axes
                arr = tf.asarray()
                ch_names = _extract_channel_names(tf)
    except Exception:
        axes = None
        arr = tifffile.imread(path)

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
    return image_cyx, ch_names
