from pathlib import Path
from dataclasses import dataclass

import spatialdata as sd
import geopandas as gpd


@dataclass
class SegmentationMask:
    """Segmentation mask loaded from a zarr store."""
    sdata: sd.SpatialData
    boundaries: gpd.GeoDataFrame


def load_mask(mask_path: Path,
              boundaries_path: str = "shapes/cellpose_boundaries/shapes.parquet"
              ) -> SegmentationMask:
    """
    Load a mask image file using SpatialData and validate the presence of a segmentation layer.

    Args:
        mask_path (Path): Path to the mask image file
        seg_name (str, optional): Name of the segmentation layer to load. Must correspond to the
            name of a parquet file in the zarr store. Defaults to "cellpose".
    Returns:
        SegmentationMask: Segmentation mask loaded from the zarr store
    """
    sdata: sd.SpatialData = sd.read_zarr(mask_path)

    parquet_path: Path = mask_path / boundaries_path
    if not parquet_path.exists():
        raise ValueError(f"No segmentation layer named {boundaries_path} found in {mask_path}")

    boundaries: gpd.GeoDataFrame = gpd.read_parquet(parquet_path)

    return SegmentationMask(sdata, boundaries)
