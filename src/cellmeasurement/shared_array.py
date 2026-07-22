"""Shared-memory NumPy array helpers.

Large label and image arrays are copied into shared memory once and then
attached read-only by every worker process, so a process pool never pays to
pickle a multi-gigabyte array per task.

These live outside :mod:`cellmeasurement.measurement` because both the
measurement pipeline and geometry extraction depend on them, and geometry
extraction should not have to import the TIFF/zarr stack to get them.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np

__all__ = ["SharedArraySpec", "create_shared_array", "open_shared_array"]


@dataclass(frozen=True)
class SharedArraySpec:
    """Shared-memory metadata required to rebuild a NumPy view in workers."""

    name: str
    shape: tuple[int, ...]
    dtype: str


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
