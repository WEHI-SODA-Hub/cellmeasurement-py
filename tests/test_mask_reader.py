from __future__ import annotations

import numpy as np
import pytest
import tifffile

from cellmeasurement.io.mask_reader import load_mask


def test_load_mask_tiff_converts_to_temp_zarr(tmp_path):
    arr = np.array([[0, 1, 1], [2, 0, 3]], dtype=np.uint16)
    tiff_path = tmp_path / "labels.tiff"
    tifffile.imwrite(tiff_path, arr)

    mask = load_mask(tiff_path, temp_dir=tmp_path)

    assert mask.shape == arr.shape
    assert mask.boundaries is None
    assert mask.temp_store_path is not None
    assert mask.temp_store_path.exists()
    assert mask.temp_store_path.parent == tmp_path
    assert mask.labels.dtype == arr.dtype
    np.testing.assert_array_equal(mask.labels.compute(), arr)

    temp_store = mask.temp_store_path
    mask.cleanup_temp_store()
    assert mask.temp_store_path is None
    assert not temp_store.exists()


def test_load_mask_tiff_rejects_non_2d(tmp_path):
    arr = np.zeros((2, 4, 4), dtype=np.uint16)
    tiff_path = tmp_path / "labels_3d.tiff"
    tifffile.imwrite(tiff_path, arr)

    with pytest.raises(ValueError, match="2D"):
        load_mask(tiff_path, temp_dir=tmp_path)


def test_load_mask_tiff_rejects_non_integer(tmp_path):
    arr = np.array([[0.0, 1.5], [2.5, 0.0]], dtype=np.float32)
    tiff_path = tmp_path / "labels_float.tiff"
    tifffile.imwrite(tiff_path, arr)

    with pytest.raises(ValueError, match="integer"):
        load_mask(tiff_path, temp_dir=tmp_path)


def test_load_mask_tiff_rejects_gt_uint32(tmp_path):
    arr = np.array([[0, 1], [2, 3]], dtype=np.uint64)
    tiff_path = tmp_path / "labels_u64.tiff"
    tifffile.imwrite(tiff_path, arr)

    with pytest.raises(ValueError, match="<= 32-bit"):
        load_mask(tiff_path, temp_dir=tmp_path)


def test_load_mask_tiff_falls_back_to_imread_when_memmap_fails(tmp_path, monkeypatch, caplog):
    arr = np.array([[0, 1], [2, 3]], dtype=np.uint16)
    tiff_path = tmp_path / "labels_fallback.tiff"
    tifffile.imwrite(tiff_path, arr)

    monkeypatch.setattr(tifffile, "memmap", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("no memmap")))

    mask = load_mask(tiff_path, temp_dir=tmp_path)

    assert "falling back to tifffile.imread" in caplog.text
    np.testing.assert_array_equal(mask.labels.compute(), arr)
