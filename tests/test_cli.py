from __future__ import annotations

from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import pytest
import typer
from shapely.geometry import Polygon

from cellmeasurement import cli
from cellmeasurement.io.mask_reader import SegmentationMask


def _one_pixel_labels() -> da.Array:
    return da.from_array(np.array([[1]], dtype=np.int32), chunks=(1, 1))


def test_extract_export_geometries_uses_labels_when_boundaries_missing(monkeypatch):
    mask = SegmentationMask(labels=_one_pixel_labels(), shape=(1, 1), boundaries=None)
    expected = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    called = {"extract": False}

    def fake_extract_label_geometries(labels, simplify, tolerance):
        called["extract"] = True
        assert labels.shape == (1, 1)
        assert simplify is True
        assert tolerance == 0.5
        return {1: expected}

    def fail_boundaries(*args, **kwargs):
        raise AssertionError("boundaries_to_geometries should not be called when boundaries is None")

    monkeypatch.setattr(cli, "extract_label_geometries", fake_extract_label_geometries)
    monkeypatch.setattr(cli, "boundaries_to_geometries", fail_boundaries)

    geoms = cli._extract_export_geometries(mask, simplify=True, tolerance=0.5)
    assert called["extract"] is True
    assert geoms == {1: expected}


def test_extract_export_geometries_uses_boundaries_when_available(monkeypatch):
    boundaries = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
    boundaries.index = boundaries.index + 1
    mask = SegmentationMask(labels=_one_pixel_labels(), shape=(1, 1), boundaries=boundaries)
    expected = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

    called = {"boundaries": False}

    def fake_boundaries_to_geometries(gdf, simplify, tolerance):
        called["boundaries"] = True
        assert gdf is boundaries
        assert simplify is False
        assert tolerance == 1.25
        return {1: expected}

    def fail_extract(*args, **kwargs):
        raise AssertionError("extract_label_geometries should not be called when boundaries exist")

    monkeypatch.setattr(cli, "boundaries_to_geometries", fake_boundaries_to_geometries)
    monkeypatch.setattr(cli, "extract_label_geometries", fail_extract)

    geoms = cli._extract_export_geometries(mask, simplify=False, tolerance=1.25)
    assert called["boundaries"] is True
    assert geoms == {1: expected}


def test_main_cleans_temp_store_by_default(monkeypatch, tmp_path):
    class DummyMask:
        def __init__(self):
            self.labels = _one_pixel_labels()
            self.shape = (1, 1)
            self.boundaries = None
            self.temp_store_path = tmp_path / "temp-mask-store"
            self.temp_store_path.mkdir()
            self.cleaned = False

        def cleanup_temp_store(self):
            self.cleaned = True
            self.temp_store_path.rmdir()
            self.temp_store_path = None

    dummy = DummyMask()

    def fake_load_mask(mask_path, parquet_path, temp_dir):
        assert isinstance(mask_path, Path)
        return dummy

    monkeypatch.setattr(cli, "load_mask", fake_load_mask)
    monkeypatch.setattr(cli, "match_rois", lambda nuc, wc, dist_threshold, estimate_cell_boundary_dist: ([], {}))
    monkeypatch.setattr(cli, "_extract_export_geometries", lambda mask, simplify, tolerance: {})
    monkeypatch.setattr(cli, "write_geojson", lambda **kwargs: 0)

    cli.main(
        nuclear_mask=tmp_path / "nuc.tiff",
        whole_cell_mask=None,
        output_file=tmp_path / "out.geojson",
    )

    assert dummy.cleaned is True


def test_main_keeps_temp_store_when_flag_enabled(monkeypatch, tmp_path):
    class DummyMask:
        def __init__(self):
            self.labels = _one_pixel_labels()
            self.shape = (1, 1)
            self.boundaries = None
            self.temp_store_path = tmp_path / "temp-mask-store"
            self.temp_store_path.mkdir()
            self.cleaned = False

        def cleanup_temp_store(self):
            self.cleaned = True
            self.temp_store_path.rmdir()
            self.temp_store_path = None

    dummy = DummyMask()

    monkeypatch.setattr(cli, "load_mask", lambda mask_path, parquet_path, temp_dir: dummy)
    monkeypatch.setattr(cli, "match_rois", lambda nuc, wc, dist_threshold, estimate_cell_boundary_dist: ([], {}))
    monkeypatch.setattr(cli, "_extract_export_geometries", lambda mask, simplify, tolerance: {})
    monkeypatch.setattr(cli, "write_geojson", lambda **kwargs: 0)

    cli.main(
        nuclear_mask=tmp_path / "nuc.tiff",
        whole_cell_mask=None,
        keep_temp_zarr=True,
        output_file=tmp_path / "out.geojson",
    )

    assert dummy.cleaned is False
    assert dummy.temp_store_path is not None
    assert dummy.temp_store_path.exists()


def test_main_mixed_zarr_tiff_validates_and_cleans_only_temp(monkeypatch, tmp_path):
    class DummyMask:
        def __init__(self, with_temp: bool):
            self.labels = _one_pixel_labels()
            self.shape = (1, 1)
            self.boundaries = None
            self.cleaned = False
            if with_temp:
                self.temp_store_path = tmp_path / "temp-mask-store"
                self.temp_store_path.mkdir()
            else:
                self.temp_store_path = None

        def cleanup_temp_store(self):
            self.cleaned = True
            if self.temp_store_path is not None:
                self.temp_store_path.rmdir()
                self.temp_store_path = None

    tiff_mask = DummyMask(with_temp=True)
    zarr_mask = DummyMask(with_temp=False)

    def fake_load_mask(mask_path, parquet_path, temp_dir):
        if mask_path.suffix.lower() in {".tif", ".tiff"}:
            return tiff_mask
        return zarr_mask

    validated = {"called": False}

    def fake_validate(mask_a, mask_b):
        validated["called"] = True
        assert mask_a.shape == mask_b.shape

    monkeypatch.setattr(cli, "load_mask", fake_load_mask)
    monkeypatch.setattr(cli, "validate_grid_compatibility", fake_validate)
    monkeypatch.setattr(cli, "match_rois", lambda nuc, wc, dist_threshold, estimate_cell_boundary_dist: ([], {}))
    monkeypatch.setattr(cli, "_extract_export_geometries", lambda mask, simplify, tolerance: {})
    monkeypatch.setattr(cli, "write_geojson", lambda **kwargs: 0)

    cli.main(
        nuclear_mask=tmp_path / "nuc.tiff",
        whole_cell_mask=tmp_path / "wc.zarr",
        output_file=tmp_path / "out.geojson",
    )

    assert validated["called"] is True
    assert tiff_mask.cleaned is True
    assert zarr_mask.cleaned is False


def test_main_preserves_primary_error_when_cleanup_fails(monkeypatch, tmp_path, caplog):
    class DummyMask:
        def __init__(self):
            self.labels = _one_pixel_labels()
            self.shape = (1, 1)
            self.boundaries = None
            self.temp_store_path = tmp_path / "temp-mask-store"
            self.temp_store_path.mkdir()

        def cleanup_temp_store(self):
            raise OSError("cleanup failed")

    dummy = DummyMask()

    monkeypatch.setattr(cli, "load_mask", lambda mask_path, parquet_path, temp_dir: dummy)
    monkeypatch.setattr(
        cli,
        "match_rois",
        lambda nuc, wc, dist_threshold, estimate_cell_boundary_dist: (_ for _ in ()).throw(RuntimeError("pipeline failed")),
    )
    monkeypatch.setattr(cli, "_extract_export_geometries", lambda mask, simplify, tolerance: {})
    monkeypatch.setattr(cli, "write_geojson", lambda **kwargs: 0)

    with pytest.raises(RuntimeError, match="pipeline failed"):
        cli.main(
            nuclear_mask=tmp_path / "nuc.tiff",
            whole_cell_mask=None,
            output_file=tmp_path / "out.geojson",
        )

    assert "Failed to clean temporary zarr store" in caplog.text


def test_main_measurements_on_by_default(monkeypatch, tmp_path):
    class DummyMask:
        def __init__(self):
            self.labels = _one_pixel_labels()
            self.shape = (1, 1)
            self.boundaries = None
            self.temp_store_path = None

        def cleanup_temp_store(self):
            self.temp_store_path = None

    dummy = DummyMask()
    called = {"measure": False}

    monkeypatch.setattr(cli, "load_mask", lambda mask_path, parquet_path, temp_dir: dummy)
    monkeypatch.setattr(cli, "match_rois", lambda nuc, wc, dist_threshold, estimate_cell_boundary_dist: ([], {}))
    monkeypatch.setattr(cli, "_extract_export_geometries", lambda mask, simplify, tolerance: {})

    def fake_measure(**kwargs):
        called["measure"] = True
        assert kwargs["pixel_size_microns"] == 0.5
        assert kwargs["environment_expansion_enabled"] is False
        assert kwargs["neighbours"] == 0
        return {}

    monkeypatch.setattr(cli, "measure_cells_tiled", fake_measure)
    monkeypatch.setattr(cli, "write_geojson", lambda **kwargs: 0)

    cli.main(
        nuclear_mask=tmp_path / "nuc.tiff",
        output_file=tmp_path / "out.geojson",
        tiff_file=tmp_path / "img.tiff",
    )

    assert called["measure"] is True


def test_main_warns_when_measurements_requested_without_tiff(monkeypatch, tmp_path, caplog):
    class DummyMask:
        def __init__(self):
            self.labels = _one_pixel_labels()
            self.shape = (1, 1)
            self.boundaries = None
            self.temp_store_path = None

        def cleanup_temp_store(self):
            self.temp_store_path = None

    dummy = DummyMask()
    called = {"measure": False, "writer_measurements": "unset", "writer_jsonl": "unset"}

    monkeypatch.setattr(cli, "load_mask", lambda mask_path, parquet_path, temp_dir: dummy)
    monkeypatch.setattr(cli, "match_rois", lambda nuc, wc, dist_threshold, estimate_cell_boundary_dist: ([], {}))
    monkeypatch.setattr(cli, "_extract_export_geometries", lambda mask, simplify, tolerance: {})

    def fake_measure(**kwargs):
        called["measure"] = True
        return {}

    def fake_write_geojson(**kwargs):
        called["writer_measurements"] = kwargs.get("measurements_by_cell")
        called["writer_jsonl"] = kwargs.get("measurements_jsonl_path")
        return 0

    monkeypatch.setattr(cli, "measure_cells_tiled", fake_measure)
    monkeypatch.setattr(cli, "write_geojson", fake_write_geojson)

    cli.main(
        nuclear_mask=tmp_path / "nuc.tiff",
        output_file=tmp_path / "out.geojson",
        measurements=True,
    )

    assert called["measure"] is False
    assert called["writer_measurements"] is None
    assert called["writer_jsonl"] is None
    assert "--tiff-file" in caplog.text


def test_main_runs_measurements_when_enabled_with_tiff(monkeypatch, tmp_path):
    class DummyMask:
        def __init__(self):
            self.labels = _one_pixel_labels()
            self.shape = (1, 1)
            self.boundaries = None
            self.temp_store_path = None

        def cleanup_temp_store(self):
            self.temp_store_path = None

    dummy = DummyMask()
    called = {"measure": False, "writer_measurements": None, "writer_jsonl": None}

    cell = cli.match_rois(
        da.from_array(np.array([[1]], dtype=np.int32), chunks=(1, 1)),
        da.from_array(np.array([[1]], dtype=np.int32), chunks=(1, 1)),
        estimate_cell_boundary_dist=3.0,
    )[0][0]

    monkeypatch.setattr(cli, "load_mask", lambda mask_path, parquet_path, temp_dir: dummy)
    monkeypatch.setattr(cli, "match_rois", lambda nuc, wc, dist_threshold, estimate_cell_boundary_dist: ([cell], {}))
    monkeypatch.setattr(cli, "_extract_export_geometries", lambda mask, simplify, tolerance: {})

    def fake_measure(**kwargs):
        called["measure"] = True
        assert kwargs["tiff_file"] == tmp_path / "img.tiff"
        assert kwargs["jsonl_path"].name.endswith(".measurements.jsonl.tmp")
        assert kwargs["return_results"] is False
        assert kwargs["erosion_enabled"] is True
        assert kwargs["expansion_enabled"] is True
        assert kwargs["environment_expansion_enabled"] is False
        assert kwargs["neighbours"] == 0
        assert kwargs["pixel_size_microns"] == 0.5
        kwargs["jsonl_path"].write_text('{"cell_id":1,"measurements":{"Cell: Area µm^2":1.0}}\\n', encoding="utf-8")
        return {}

    def fake_write_geojson(**kwargs):
        called["writer_measurements"] = kwargs.get("measurements_by_cell")
        called["writer_jsonl"] = kwargs.get("measurements_jsonl_path")
        return 1

    monkeypatch.setattr(cli, "measure_cells_tiled", fake_measure)
    monkeypatch.setattr(cli, "write_geojson", fake_write_geojson)

    cli.main(
        nuclear_mask=tmp_path / "nuc.tiff",
        output_file=tmp_path / "out.geojson",
        measurements=True,
        tiff_file=tmp_path / "img.tiff",
    )

    assert called["measure"] is True
    assert called["writer_measurements"] is None
    assert called["writer_jsonl"] is not None


def test_main_passes_step_toggles_to_measurements(monkeypatch, tmp_path):
    class DummyMask:
        def __init__(self):
            self.labels = _one_pixel_labels()
            self.shape = (1, 1)
            self.boundaries = None
            self.temp_store_path = None

        def cleanup_temp_store(self):
            self.temp_store_path = None

    dummy = DummyMask()
    called = {"erosion_enabled": None, "expansion_enabled": None}
    match_called = {"dist_threshold": None}

    cell = cli.match_rois(
        da.from_array(np.array([[1]], dtype=np.int32), chunks=(1, 1)),
        da.from_array(np.array([[1]], dtype=np.int32), chunks=(1, 1)),
        estimate_cell_boundary_dist=3.0,
    )[0][0]

    monkeypatch.setattr(cli, "load_mask", lambda mask_path, parquet_path, temp_dir: dummy)

    def fake_match_rois(nuc, wc, dist_threshold, estimate_cell_boundary_dist):
        match_called["dist_threshold"] = dist_threshold
        return [cell], {}

    monkeypatch.setattr(cli, "match_rois", fake_match_rois)
    monkeypatch.setattr(cli, "_extract_export_geometries", lambda mask, simplify, tolerance: {})

    def fake_measure(**kwargs):
        called["erosion_enabled"] = kwargs["erosion_enabled"]
        called["expansion_enabled"] = kwargs["expansion_enabled"]
        called["environment_expansion_enabled"] = kwargs["environment_expansion_enabled"]
        called["neighbours"] = kwargs["neighbours"]
        called["pixel_size_microns"] = kwargs["pixel_size_microns"]
        kwargs["jsonl_path"].write_text('{"cell_id":1,"measurements":{"Cell: Area µm^2":1.0}}\\n', encoding="utf-8")
        return {}

    monkeypatch.setattr(cli, "measure_cells_tiled", fake_measure)
    monkeypatch.setattr(cli, "write_geojson", lambda **kwargs: 1)

    cli.main(
        nuclear_mask=tmp_path / "nuc.tiff",
        output_file=tmp_path / "out.geojson",
        measurements=True,
        tiff_file=tmp_path / "img.tiff",
        erosion_steps=False,
        expansion_steps=False,
        environment_expansion=True,
        neighbours=3,
        dist_threshold=12.0,
        pixel_size_microns=0.8,
    )

    assert called["erosion_enabled"] is False
    assert called["expansion_enabled"] is False
    assert called["environment_expansion_enabled"] is True
    assert called["neighbours"] == 3
    assert called["pixel_size_microns"] == 0.8
    assert match_called["dist_threshold"] == 12.0


def test_main_rejects_negative_neighbours(tmp_path):
    with pytest.raises(typer.Exit):
        cli.main(
            nuclear_mask=tmp_path / "nuc.tiff",
            output_file=tmp_path / "out.geojson",
            measurements=False,
            neighbours=-1,
        )


def test_main_rejects_nonpositive_dist_threshold(tmp_path):
    with pytest.raises(typer.Exit):
        cli.main(
            nuclear_mask=tmp_path / "nuc.tiff",
            output_file=tmp_path / "out.geojson",
            measurements=False,
            dist_threshold=0.0,
        )
