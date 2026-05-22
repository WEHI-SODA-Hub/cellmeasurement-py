from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import tifffile
import zarr

from .models import TileBounds

MaskKind = Literal["nucleus", "whole_cell"]
_CHANNEL_AXIS_NAMES = ("C", "S", "Q", "I")


@dataclass(frozen=True)
class SharedArraySpec:
    """Shared-memory metadata required to rebuild a NumPy view in workers."""

    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class TiffDataAccessSpec:
    """Serializable metadata needed for worker-local TIFF window reads."""

    path: str
    axes: str | None
    source_shape: tuple[int, ...]
    cyx_shape: tuple[int, int, int]


@dataclass(frozen=True)
class _CyxLayout:
    source_ndim: int
    channel_axis: int | None
    y_axis: int
    x_axis: int
    dropped_axes: tuple[int, ...]
    kept_axes: tuple[str, ...]
    channel_axis_name: str | None
    transpose_order: tuple[int, int, int] | None
    cyx_shape: tuple[int, int, int]


def create_shared_array(arr: np.ndarray) -> tuple[SharedArraySpec, SharedMemory]:
    """Copy an array to shared memory and return its spec and backing handle."""
    contiguous = np.ascontiguousarray(arr)
    shm = SharedMemory(create=True, size=contiguous.nbytes)
    shared_arr = np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=shm.buf)
    shared_arr[...] = contiguous
    spec = SharedArraySpec(
        name=shm.name,
        shape=tuple(int(v) for v in contiguous.shape),
        dtype=contiguous.dtype.str,
    )
    return spec, shm


def open_shared_array(spec: SharedArraySpec) -> tuple[np.ndarray, SharedMemory]:
    """Open a shared-memory array view from a parent-created spec."""
    shm = SharedMemory(name=spec.name)
    arr = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype), buffer=shm.buf)
    return arr, shm


def infer_cyx_shape(shape: tuple[int, ...], axes: str | None) -> tuple[int, int, int]:
    """Infer normalized (C, Y, X) shape from source shape/axes metadata."""
    return _build_cyx_layout(shape=shape, axes=axes).cyx_shape


def _is_array_like(node: Any) -> bool:
    return hasattr(node, "shape") and hasattr(node, "dtype") and hasattr(node, "__getitem__")


def _iter_node_keys(node: Any) -> Iterable[str]:
    keys_fn = getattr(node, "keys", None)
    if callable(keys_fn):
        try:
            for key in keys_fn():
                yield str(key)
        except Exception:
            return


def _resolve_zarr_array(node: Any, *, seen: set[int] | None = None) -> Any:
    """Resolve an array-like node from a zarr root that may be a Group."""
    if _is_array_like(node):
        return node

    if seen is None:
        seen = set()
    node_id = id(node)
    if node_id in seen:
        raise ValueError("Could not resolve zarr array from TIFF store (cycle detected).")
    seen.add(node_id)

    preferred_keys = ("0", "data", "image")
    for key in preferred_keys:
        try:
            child = node[key]
        except Exception:
            continue
        try:
            return _resolve_zarr_array(child, seen=seen)
        except ValueError:
            continue

    array_keys_fn = getattr(node, "array_keys", None)
    if callable(array_keys_fn):
        try:
            array_keys = sorted(str(k) for k in array_keys_fn())
        except Exception:
            array_keys = []
        for key in array_keys:
            try:
                child = node[key]
            except Exception:
                continue
            if _is_array_like(child):
                return child

    for key in _iter_node_keys(node):
        try:
            child = node[key]
        except Exception:
            continue
        try:
            return _resolve_zarr_array(child, seen=seen)
        except ValueError:
            continue

    raise ValueError("Could not resolve zarr array from TIFF store.")


def _build_cyx_layout(shape: tuple[int, ...], axes: str | None) -> _CyxLayout:
    if axes is not None and len(axes) == len(shape):
        axis_list = list(axes)
        shape_list = list(shape)
        dropped_axes: list[int] = []
        for idx in range(len(axis_list) - 1, -1, -1):
            ax = axis_list[idx]
            if ax in {*_CHANNEL_AXIS_NAMES, "Y", "X"}:
                continue
            if int(shape_list[idx]) != 1:
                raise ValueError(
                    f"Unsupported TIFF layout: non-singleton axis '{ax}' in shape {shape} (axes='{axes}')"
                )
            dropped_axes.append(idx)
            axis_list.pop(idx)
            shape_list.pop(idx)

        channel_axis: int | None = None
        channel_axis_name: str | None = None
        for ax_name in _CHANNEL_AXIS_NAMES:
            if ax_name in axis_list:
                channel_axis = axis_list.index(ax_name)
                channel_axis_name = ax_name
                break
        if "Y" not in axis_list or "X" not in axis_list:
            raise ValueError(f"Could not find Y/X axes in TIFF layout (axes='{axes}', shape={shape})")
        y_axis = axis_list.index("Y")
        x_axis = axis_list.index("X")
        if channel_axis is None:
            cyx_shape = (1, int(shape_list[y_axis]), int(shape_list[x_axis]))
            return _CyxLayout(
                source_ndim=len(shape),
                channel_axis=None,
                y_axis=y_axis,
                x_axis=x_axis,
                dropped_axes=tuple(sorted(dropped_axes)),
                kept_axes=tuple(axis_list),
                channel_axis_name=None,
                transpose_order=None,
                cyx_shape=cyx_shape,
            )

        cyx_shape = (
            int(shape_list[channel_axis]),
            int(shape_list[y_axis]),
            int(shape_list[x_axis]),
        )
        transpose_order = (channel_axis, y_axis, x_axis)
        return _CyxLayout(
            source_ndim=len(shape),
            channel_axis=channel_axis,
            y_axis=y_axis,
            x_axis=x_axis,
            dropped_axes=tuple(sorted(dropped_axes)),
            kept_axes=tuple(axis_list),
            channel_axis_name=channel_axis_name,
            transpose_order=transpose_order,
            cyx_shape=cyx_shape,
        )

    # Heuristic fallback when axis metadata is unavailable.
    if len(shape) == 2:
        return _CyxLayout(
            source_ndim=2,
            channel_axis=None,
            y_axis=0,
            x_axis=1,
            dropped_axes=(),
            kept_axes=("Y", "X"),
            channel_axis_name=None,
            transpose_order=None,
            cyx_shape=(1, int(shape[0]), int(shape[1])),
        )
    if len(shape) != 3:
        raise ValueError(f"Unsupported TIFF image dimensions without axes metadata: shape={shape}")

    if shape[0] <= shape[1] and shape[0] <= shape[2]:
        return _CyxLayout(
            source_ndim=3,
            channel_axis=0,
            y_axis=1,
            x_axis=2,
            dropped_axes=(),
            kept_axes=("C", "Y", "X"),
            channel_axis_name="C",
            transpose_order=(0, 1, 2),
            cyx_shape=(int(shape[0]), int(shape[1]), int(shape[2])),
        )
    if shape[2] <= shape[0] and shape[2] <= shape[1]:
        return _CyxLayout(
            source_ndim=3,
            channel_axis=2,
            y_axis=0,
            x_axis=1,
            dropped_axes=(),
            kept_axes=("Y", "X", "C"),
            channel_axis_name="C",
            transpose_order=(2, 0, 1),
            cyx_shape=(int(shape[2]), int(shape[0]), int(shape[1])),
        )
    raise ValueError(f"Unsupported 3D TIFF image layout: shape={shape}")


class TileDataAccess(Protocol):
    """Abstract tile/bbox data reads from image and segmentation masks."""

    def read_tile_image(self, bounds: TileBounds) -> np.ndarray: ...

    def read_tile_mask(self, kind: MaskKind, bounds: TileBounds) -> np.ndarray | None: ...

    def read_bbox_image(self, bbox: tuple[int, int, int, int]) -> np.ndarray: ...

    def read_bbox_mask(self, kind: MaskKind, bbox: tuple[int, int, int, int]) -> np.ndarray | None: ...

    def close(self) -> None: ...


@dataclass
class InMemoryTileDataAccess:
    """Data access over in-memory arrays (including shared-memory-backed views)."""

    image_cyx: np.ndarray
    nuc_labels: np.ndarray | None
    wc_labels: np.ndarray | None

    def read_tile_image(self, bounds: TileBounds) -> np.ndarray:
        return self.image_cyx[:, bounds.r0:bounds.r1, bounds.c0:bounds.c1]

    def read_tile_mask(self, kind: MaskKind, bounds: TileBounds) -> np.ndarray | None:
        source = self.nuc_labels if kind == "nucleus" else self.wc_labels
        if source is None:
            return None
        return source[bounds.r0:bounds.r1, bounds.c0:bounds.c1]

    def read_bbox_image(self, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bbox
        return self.image_cyx[:, r0:r1, c0:c1]

    def read_bbox_mask(self, kind: MaskKind, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        source = self.nuc_labels if kind == "nucleus" else self.wc_labels
        if source is None:
            return None
        r0, c0, r1, c1 = bbox
        return source[r0:r1, c0:c1]

    def close(self) -> None:
        return


@dataclass
class TiffTileDataAccess:
    """Windowed TIFF-backed image reads, with in-memory masks for label lookups."""

    tiff_spec: TiffDataAccessSpec
    nuc_labels: np.ndarray | None
    wc_labels: np.ndarray | None
    _tiff: tifffile.TiffFile | None = None
    _store: Any = None
    _array: Any = None
    _layout: _CyxLayout | None = None

    def __post_init__(self) -> None:
        # Delay TIFF/Zarr open until the first read; this avoids parent-process
        # file/store opens in process mode where workers initialize their own access.
        return

    def _ensure_open(self) -> None:
        if self._array is not None and self._layout is not None:
            return
        try:
            if self._tiff is None:
                self._tiff = tifffile.TiffFile(Path(self.tiff_spec.path))
            if self._store is None:
                series = self._tiff.series[0]
                self._store = series.aszarr()
            self._array = _resolve_zarr_array(zarr.open(self._store, mode="r"))
            source_shape = tuple(int(v) for v in self._array.shape)
            axes = self.tiff_spec.axes
            if axes is not None and len(axes) != len(source_shape):
                axes = None
            self._layout = _build_cyx_layout(shape=source_shape, axes=axes)
            if self._layout.cyx_shape != self.tiff_spec.cyx_shape:
                raise ValueError(
                    f"TIFF layout changed between inspection and worker open: expected {self.tiff_spec.cyx_shape}, "
                    f"got {self._layout.cyx_shape}"
                )
        except Exception:
            self.close()
            raise

    def _read_cyx_window(self, *, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        self._ensure_open()
        if self._array is None or self._layout is None:
            raise RuntimeError("TIFF data access not initialized")
        index: list[Any] = []
        kept_axis_pos = -1
        for axis_idx in range(self._layout.source_ndim):
            if axis_idx in self._layout.dropped_axes:
                index.append(0)
                continue
            kept_axis_pos += 1
            if kept_axis_pos == self._layout.y_axis:
                index.append(slice(r0, r1))
            elif kept_axis_pos == self._layout.x_axis:
                index.append(slice(c0, c1))
            elif self._layout.channel_axis is not None and kept_axis_pos == self._layout.channel_axis:
                index.append(slice(None))
            else:
                index.append(0)

        arr = np.asarray(self._array[tuple(index)])
        if self._layout.channel_axis is None:
            return arr[np.newaxis, ...]
        transpose_order = self._layout.transpose_order
        if transpose_order is None:
            return arr
        if tuple(transpose_order) == (0, 1, 2):
            return arr
        return np.transpose(arr, transpose_order)

    def read_tile_image(self, bounds: TileBounds) -> np.ndarray:
        return self._read_cyx_window(r0=bounds.r0, r1=bounds.r1, c0=bounds.c0, c1=bounds.c1)

    def read_tile_mask(self, kind: MaskKind, bounds: TileBounds) -> np.ndarray | None:
        source = self.nuc_labels if kind == "nucleus" else self.wc_labels
        if source is None:
            return None
        return source[bounds.r0:bounds.r1, bounds.c0:bounds.c1]

    def read_bbox_image(self, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bbox
        return self._read_cyx_window(r0=r0, r1=r1, c0=c0, c1=c1)

    def read_bbox_mask(self, kind: MaskKind, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        source = self.nuc_labels if kind == "nucleus" else self.wc_labels
        if source is None:
            return None
        r0, c0, r1, c1 = bbox
        return source[r0:r1, c0:c1]

    def close(self) -> None:
        self._array = None
        self._layout = None
        if self._store is not None:
            try:
                close_fn = getattr(self._store, "close", None)
                if callable(close_fn):
                    close_fn()
            finally:
                self._store = None
        if self._tiff is not None:
            self._tiff.close()
            self._tiff = None
