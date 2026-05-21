from __future__ import annotations

import zarr

from cellmeasurement.measurement.data_access import _resolve_zarr_array


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
