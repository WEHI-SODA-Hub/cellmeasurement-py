from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import tempfile

import dask
import dask.array as da
import geopandas as gpd
import numpy as np
from skimage.measure import block_reduce
import spatialdata as sd
import tifffile

logger = logging.getLogger(__name__)


def _rasterize_boundaries(boundaries: gpd.GeoDataFrame, H: int, W: int) -> np.ndarray:
    """Burn GeoDataFrame polygons into an integer label array.

    Label values match the GeoDataFrame index (expected to be 1-based integers
    after normalisation in :func:`load_mask`; 0 is background).
    """
    import rasterio.features

    shapes = [
        (geom, int(idx))  # type: ignore[arg-type]
        for idx, geom in boundaries.geometry.items()
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        return np.zeros((H, W), dtype=np.int32)
    return rasterio.features.rasterize(shapes, out_shape=(H, W), dtype=np.int32)


@dataclass
class SegmentationMask:
    """Unified segmentation mask wrapper for zarr- or TIFF-backed inputs."""

    labels: da.Array
    shape: tuple[int, int]
    boundaries: gpd.GeoDataFrame | None = None
    temp_store_path: Path | None = None

    def cleanup_temp_store(self) -> None:
        """Delete temporary on-disk zarr storage if present."""
        if self.temp_store_path is None:
            return
        if self.temp_store_path.exists():
            shutil.rmtree(self.temp_store_path)
        self.temp_store_path = None


def _shape_from_sdata_images(sdata: sd.SpatialData) -> tuple[int, int]:
    """Return ``(height, width)`` from the first image layer in a zarr store."""
    if not sdata.images:
        raise ValueError("No image layers found in zarr store; cannot infer mask shape.")

    img_name = next(iter(sdata.images))
    img_tree = sdata.images[img_name]
    scale_names = [k for k in img_tree.keys() if k.startswith("scale")]
    scale0 = "scale0" if "scale0" in scale_names else scale_names[0]
    img_shape = img_tree[scale0]["image"].shape

    # sopa images are stored as (C, Y, X); take the last two dims.
    return (int(img_shape[-2]), int(img_shape[-1]))


def _load_zarr_with_boundaries(mask_path: Path, parquet_path: str) -> SegmentationMask:
    sdata: sd.SpatialData = sd.read_zarr(mask_path)
    H, W = _shape_from_sdata_images(sdata)

    boundaries_path: Path = mask_path / parquet_path
    if not boundaries_path.exists():
        raise ValueError(
            f"Boundaries file not found: {boundaries_path}. "
            f"Check that '{parquet_path}' is correct."
        )
    boundaries = gpd.read_parquet(boundaries_path)
    # Normalise to 1-based integer index so label 0 is always background.
    boundaries = boundaries.reset_index(drop=True)
    boundaries.index = boundaries.index + 1

    labels = da.from_delayed(
        dask.delayed(_rasterize_boundaries)(boundaries, H, W),
        shape=(H, W),
        dtype=np.int32,
    )
    return SegmentationMask(labels=labels, shape=(H, W), boundaries=boundaries)


def _default_chunks(shape: tuple[int, int]) -> tuple[int, int]:
    return (max(1, min(2048, shape[0])), max(1, min(2048, shape[1])))


def _load_tiff_as_temp_zarr(mask_path: Path, temp_dir: Path | None) -> SegmentationMask:
    try:
        arr = tifffile.memmap(mask_path)
    except (ValueError, OSError, tifffile.TiffFileError, NotImplementedError):
        logger.warning(
            "tifffile.memmap failed for %s; falling back to tifffile.imread "
            "(higher memory usage).",
            mask_path,
        )
        arr = tifffile.imread(mask_path)
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D label image, got shape={arr.shape} for {mask_path}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Mask must use an integer label dtype, got {arr.dtype} for {mask_path}")
    if arr.dtype.itemsize > 4:
        raise ValueError(
            f"Mask label dtype must be <= 32-bit integer, got {arr.dtype} for {mask_path}"
        )

    H, W = int(arr.shape[0]), int(arr.shape[1])
    chunks = _default_chunks((H, W))

    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_root = Path(tempfile.mkdtemp(prefix="cellmeasurement-mask-", dir=str(temp_dir)))
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="cellmeasurement-mask-"))

    labels_store = temp_root / "labels.zarr"

    # Persist as chunked zarr so downstream processing stays on a zarr-backed path.
    da.from_array(arr, chunks=chunks).to_zarr(labels_store, overwrite=True)
    labels = da.from_zarr(labels_store)

    return SegmentationMask(
        labels=labels,
        shape=(H, W),
        boundaries=None,
        temp_store_path=temp_root,
    )


def load_mask(
    mask_path: Path,
    parquet_path: str = "shapes/cellpose_boundaries/shapes.parquet",
    temp_dir: Path | None = None,
) -> SegmentationMask:
    """Load a segmentation mask from zarr or TIFF and return ``SegmentationMask``.

    Args:
        mask_path: Path to either a sopa zarr store directory or a TIFF label mask.
        parquet_path: Boundaries parquet path relative to the zarr store (zarr inputs).
        temp_dir: Parent directory for temporary zarr stores created from TIFF inputs.

    Returns:
        A :class:`SegmentationMask` with dask-backed labels and optional boundaries.

    Raises:
        ValueError: If input is invalid or required boundaries are missing.
    """
    suffix = mask_path.suffix.lower()
    if mask_path.is_file() and suffix in {".tif", ".tiff"}:
        return _load_tiff_as_temp_zarr(mask_path, temp_dir)
    if mask_path.is_dir():
        return _load_zarr_with_boundaries(mask_path, parquet_path)
    raise ValueError(f"Unsupported mask path: {mask_path}. Provide a zarr directory or TIFF file.")


def validate_grid_compatibility(mask_a: SegmentationMask, mask_b: SegmentationMask) -> None:
    """Assert that two masks share the same pixel grid.

    Both nuclear and whole-cell masks produced by the same sopa run share an
    identical grid, so this is a sanity check only.  Mismatches indicate that
    the masks come from different runs and paired matching will be incorrect.

    Args:
        mask_a: First segmentation mask (e.g. nuclear).
        mask_b: Second segmentation mask (e.g. whole-cell).

    Raises:
        ValueError: If the label arrays differ in spatial shape.
    """
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"Mask shapes are incompatible for paired matching: "
            f"Mask A has shape {mask_a.shape}, "
            f"Mask B has shape {mask_b.shape}."
        )


def maybe_downsample(
    image_cyx: np.ndarray,
    nuc_mask: np.ndarray | None,
    cell_mask: np.ndarray,
    downsample_factor: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Optionally downsample multi-channel image and label masks by an integer factor.

    Downsampling is skipped if the factor is <= 1.0 or rounds to a step < 2.
    This is useful for reducing memory pressure when working with large images or many channels.

    **Image downsampling:** Uses reshape+mean to preserve intensity scale while reducing spatial
    resolution.
    **Mask downsampling:** Uses block_reduce with func=np.max to preserve label values and avoid
    fragmentation.
    **Coordinate tracking:** Downsampled coordinates must be multiplied by the step factor to map
    back to original space.

    Parameters
    ----------
    image_cyx : np.ndarray
        Multi-channel image, shape (C, H, W), typically float32 or uint16.
    nuc_mask : np.ndarray | None
        Nuclear label mask, shape (H, W), int64, or None if skipping nuclear mask.
    cell_mask : np.ndarray
        Whole-cell label mask, shape (H, W), int64.
    downsample_factor : float
        Downsampling factor (e.g. 2.0, 4.0). Rounded to nearest integer; values < 2 result in no
        downsampling.

    Returns
    -------
    tuple of (image_ds, nuc_mask_ds, cell_mask_ds)
        Downsampled arrays:
        - image_ds: float32, shape (C, H', W') where H' = (H // step) and W' = (W // step)
        - nuc_mask_ds: int64 or None, same spatial shape as image_ds
        - cell_mask_ds: int64, same spatial shape as image_ds

    Notes
    -----
    - Input image is NOT modified; returned image_ds is a new array.
    - Masks are cropped to divisible dimensions before downsampling to avoid zero-padding bias at
      edges.
    - When downsampling is applied, pixel-size measurements (erosion/expansion) should scale
      proportionally: effective_pixel_size = pixel_size_microns * step.
    """
    if downsample_factor <= 1.0:
        return image_cyx, nuc_mask, cell_mask

    step = int(round(downsample_factor))
    if step < 2:
        return image_cyx, nuc_mask, cell_mask

    # Crop to divisible dimensions to avoid zero-padding bias in block_reduce at edges.
    c, h, w = image_cyx.shape
    h_divisible = (h // step) * step
    w_divisible = (w // step) * step

    if h_divisible == 0 or w_divisible == 0:
        return image_cyx, nuc_mask, cell_mask

    # Crop all arrays to the divisible shape.
    image_crop = image_cyx[:, :h_divisible, :w_divisible]
    cell_crop = cell_mask[:h_divisible, :w_divisible]
    nuc_crop = nuc_mask[:h_divisible, :w_divisible] if nuc_mask is not None else None

    # Downsample image: reshape+mean is faster for regular blocks than block_reduce.
    # This preserves intensity scales better than max or other aggregations.
    image_ds = image_crop.reshape(c, h_divisible // step, step, w_divisible // step, step).mean(axis=(2, 4))
    image_ds = image_ds.astype(np.float32, copy=False)

    # Downsample masks: block_reduce with max preserves label values and avoids fragmentation.
    cell_ds = block_reduce(cell_crop, block_size=(step, step), func=np.max)
    cell_ds = cell_ds.astype(np.int64, copy=False)

    nuc_ds = None
    if nuc_crop is not None:
        nuc_ds = block_reduce(nuc_crop, block_size=(step, step), func=np.max)
        nuc_ds = nuc_ds.astype(np.int64, copy=False)

    return image_ds, nuc_ds, cell_ds
