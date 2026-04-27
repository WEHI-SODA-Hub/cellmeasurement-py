from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dask
import dask.array as da
import geopandas as gpd
import numpy as np
import spatialdata as sd


def _rasterize_boundaries(boundaries: gpd.GeoDataFrame, H: int, W: int) -> np.ndarray:
    """Burn GeoDataFrame polygons into an integer label array.

    Label values match the GeoDataFrame index (expected to be 1-based integers
    after normalisation in :func:`load_mask`; 0 is background).
    """
    import rasterio.features

    shapes = [
        (geom, int(idx))
        for idx, geom in boundaries.geometry.items()
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        return np.zeros((H, W), dtype=np.int32)
    return rasterio.features.rasterize(shapes, out_shape=(H, W), dtype=np.int32)


@dataclass
class SegmentationMask:
    """Segmentation mask loaded from a sopa zarr store.

    The zarr store is used for image-level metadata (spatial dimensions).
    Cell boundaries are loaded from a parquet file alongside the store.
    """

    sdata: sd.SpatialData
    boundaries: gpd.GeoDataFrame
    method: str

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` derived from the first image in the zarr store."""
        img_name = next(iter(self.sdata.images))

        img_tree = self.sdata.images[img_name]
        scale_names = [k for k in img_tree.keys() if k.startswith("scale")]

        scale0 = "scale0" if "scale0" in scale_names else scale_names[0]
        img_shape = img_tree[scale0]["image"].shape

        # sopa images are stored as (C, Y, X); take the last two dims.
        return (int(img_shape[-2]), int(img_shape[-1]))

    @property
    def labels(self) -> da.Array:
        """Dask-backed 2-D ``(height, width)`` integer label array.

        Label values correspond to the GeoDataFrame index (1-based after
        normalisation); background pixels carry value 0.  The rasterization
        is lazy — it runs when the array is first computed.
        """
        H, W = self.shape
        return da.from_delayed(
            dask.delayed(_rasterize_boundaries)(self.boundaries, H, W),
            shape=(H, W),
            dtype=np.int32,
        )


def load_mask(
    mask_path: Path,
    segmentation_method: str = "cellpose",
) -> SegmentationMask:
    """Load a sopa zarr store and return a :class:`SegmentationMask`.

    The boundaries parquet file is resolved as
    ``<mask_path>/shapes/<segmentation_method>_boundaries/shapes.parquet``.

    Args:
        mask_path: Path to the sopa zarr store directory.
        segmentation_method: Segmentation tool that produced the boundaries
            (e.g. ``"cellpose"``, ``"mesmer"``, ``"cellsam"``).

    Returns:
        A :class:`SegmentationMask` wrapping the spatialdata store.

    Raises:
        ValueError: If the boundaries parquet file does not exist.
    """
    sdata: sd.SpatialData = sd.read_zarr(mask_path)

    boundaries_path: Path = (
        mask_path / "shapes" / f"{segmentation_method}_boundaries" / "shapes.parquet"
    )
    if not boundaries_path.exists():
        raise ValueError(
            f"Boundaries file not found: {boundaries_path}. "
            f"Check that '{segmentation_method}' is the correct segmentation method."
        )
    boundaries = gpd.read_parquet(boundaries_path)
    # Normalise to 1-based integer index so label 0 is always background.
    boundaries = boundaries.reset_index(drop=True)
    boundaries.index = boundaries.index + 1

    return SegmentationMask(sdata=sdata, boundaries=boundaries, method=segmentation_method)  # type: ignore[arg-type]


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
            f"'{mask_a.method}' has shape {mask_a.shape}, "
            f"'{mask_b.method}' has shape {mask_b.shape}."
        )
