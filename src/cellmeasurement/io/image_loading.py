"""Image and mask loading utilities, including downsampling support."""

from __future__ import annotations

import numpy as np
from skimage.measure import block_reduce


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
