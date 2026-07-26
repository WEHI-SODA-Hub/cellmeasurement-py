from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

import cellmeasurement.measurement.data_access as data_access_module
from cellmeasurement.measurement.data_access import (
    TiffDataAccessSpec,
    TiffTileDataAccess,
    _resolve_zarr_array,
    infer_cyx_shape,
    open_image_tiff,
)
from cellmeasurement.measurement.image_io import inspect_tiff_image
from cellmeasurement.measurement.models import TileBounds


def test_resolve_zarr_array_handles_group_with_level_zero_array() -> None:
    store = zarr.storage.MemoryStore()
    root = zarr.group(store=store)
    root.create_array("0", shape=(3, 4, 5), dtype="u2")

    opened = zarr.open(store, mode="r")
    resolved = _resolve_zarr_array(opened)
    assert tuple(int(v) for v in resolved.shape) == (3, 4, 5)


def test_resolve_zarr_array_handles_nested_groups() -> None:
    store = zarr.storage.MemoryStore()
    root = zarr.group(store=store)
    lvl_group = root.create_group("multiscales")
    lvl_group.create_array("0", shape=(2, 6, 7), dtype="u1")

    opened = zarr.open(store, mode="r")
    resolved = _resolve_zarr_array(opened)
    assert tuple(int(v) for v in resolved.shape) == (2, 6, 7)


def test_tiff_tile_data_access_is_lazy_on_init(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "lazy_init.tiff"
    tifffile.imwrite(path, np.arange(25, dtype=np.uint16).reshape(5, 5))
    spec = TiffDataAccessSpec(
        path=str(path),
        axes=None,
        source_shape=(5, 5),
        cyx_shape=(1, 5, 5),
    )

    opened = {"count": 0}

    class _ExplodingTiffFile:
        def __init__(self, *_args, **_kwargs):
            opened["count"] += 1
            raise RuntimeError("boom")

    monkeypatch.setattr(data_access_module.tifffile, "TiffFile", _ExplodingTiffFile)

    access = TiffTileDataAccess(tiff_spec=spec, nuc_labels=None, wc_labels=None)
    assert opened["count"] == 0

    with pytest.raises(RuntimeError, match="boom"):
        access.read_tile_image(TileBounds(r0=0, c0=0, r1=1, c1=1))
    assert opened["count"] == 1


def test_tiff_tile_data_access_opens_on_first_read_and_closes(tmp_path: Path) -> None:
    path = tmp_path / "window_read.tiff"
    image = np.arange(25, dtype=np.uint16).reshape(5, 5)
    tifffile.imwrite(path, image)
    spec = TiffDataAccessSpec(
        path=str(path),
        axes=None,
        source_shape=(5, 5),
        cyx_shape=(1, 5, 5),
    )

    access = TiffTileDataAccess(tiff_spec=spec, nuc_labels=None, wc_labels=None)
    assert access._tiff is None
    assert access._array is None
    assert access._layout is None

    tile = access.read_tile_image(TileBounds(r0=1, c0=1, r1=4, c1=4))
    assert tile.shape == (1, 3, 3)
    np.testing.assert_array_equal(tile[0], image[1:4, 1:4])
    assert access._tiff is not None
    assert access._array is not None
    assert access._layout is not None

    access.close()
    assert access._tiff is None
    assert access._store is None
    assert access._array is None
    assert access._layout is None


def test_tiff_tile_data_access_reads_all_channels_when_shaped_splits_ome_series(
    shaped_split_ome_tiff,
) -> None:
    names = ["DAPI", "CD8", "Pan-CK"]
    path, data = shaped_split_ome_tiff(names, size_y=6, size_x=7)

    cyx_shape, axes, source_shape, ch_names = inspect_tiff_image(path)
    assert ch_names == names

    spec = TiffDataAccessSpec(
        path=str(path),
        axes=axes,
        source_shape=source_shape,
        cyx_shape=cyx_shape,
    )
    access = TiffTileDataAccess(tiff_spec=spec, nuc_labels=None, wc_labels=None)
    try:
        tile = access.read_tile_image(TileBounds(r0=1, c0=2, r1=5, c1=6))
        assert tile.shape == (3, 4, 4)
        np.testing.assert_array_equal(tile, data[:, 1:5, 2:6])

        bbox = access.read_bbox_image((0, 0, 6, 7))
        assert bbox.shape == (3, 6, 7)
        np.testing.assert_array_equal(bbox, data)
    finally:
        access.close()


def test_open_image_tiff_leaves_non_ome_tiff_untouched(tmp_path: Path) -> None:
    path = tmp_path / "plain_stack.tiff"
    data = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    tifffile.imwrite(path, data)

    with open_image_tiff(path) as tf:
        assert tf.series
        assert tuple(int(v) for v in tf.series[0].shape) == (2, 4, 5)


def test_open_image_tiff_keeps_shaped_when_ome_has_no_extra_channels(tmp_path: Path) -> None:
    """A well-formed single-channel OME-TIFF must not be re-interpreted."""
    path = tmp_path / "single_channel.ome.tif"
    tifffile.imwrite(path, np.zeros((4, 5), dtype=np.uint16), ome=True)

    with open_image_tiff(path) as tf:
        assert infer_cyx_shape(
            tuple(int(v) for v in tf.series[0].shape),
            str(tf.series[0].axes) if tf.series[0].axes else None,
        ) == (1, 4, 5)


def test_open_image_tiff_does_not_reopen_well_formed_ome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A single 'ome' series must not trigger the is_shaped=False re-open path."""
    path = tmp_path / "well_formed.ome.tif"
    data = np.stack([np.full((4, 5), i + 1, dtype=np.uint16) for i in range(6)])
    tifffile.imwrite(path, data, ome=True)

    reopens = {"count": 0}
    real_tifffile_open = tifffile.TiffFile

    def _spy(file, *args, **kwargs):
        if kwargs.get("is_shaped") is False:
            reopens["count"] += 1
        return real_tifffile_open(file, *args, **kwargs)

    monkeypatch.setattr(data_access_module.tifffile, "TiffFile", _spy)

    with open_image_tiff(path) as tf:
        assert tuple(int(v) for v in tf.series[0].shape) == (6, 4, 5)
    assert reopens["count"] == 0
