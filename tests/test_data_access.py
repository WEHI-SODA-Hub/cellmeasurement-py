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
)
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
