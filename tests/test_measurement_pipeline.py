from __future__ import annotations

from pathlib import Path
import json

import dask
import dask.array as da
import numpy as np
import pytest
import tifffile

import cellmeasurement.measurement.pipeline as measurement_pipeline
from cellmeasurement.measurement import measure_cells_tiled
from cellmeasurement.segmentation.cell import CellMatch


def _write_tiff(path: Path, arr: np.ndarray) -> None:
    tifffile.imwrite(path, arr)


def test_measure_cells_tiled_basic(tmp_path: Path):
    img = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 10, 20, 30, 40, 50, 0],
            [0, 15, 25, 35, 45, 55, 0],
            [0, 20, 30, 40, 50, 60, 0],
            [0, 25, 35, 45, 55, 65, 0],
            [0, 30, 40, 50, 60, 70, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )  # (Y, X) single channel
    tiff_path = tmp_path / "img.tiff"
    _write_tiff(tiff_path, img)

    wc = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint32,
    )
    nuc = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint32,
    )

    cell = CellMatch(
        cell_id=1,
        nucleus_label=1,
        whole_cell_label=1,
        bbox=(1, 1, 6, 6),
        centroid=(3.0, 3.0),
        nucleus_area_px=9,
        cell_area_px=25,
        overlap_px=9,
        overlap_fraction=1.0,
        match_source="overlap_1to1",
    )

    measured = measure_cells_tiled(
        cells=[cell],
        nuc_labels=da.from_array(nuc, chunks=(7, 7)),
        wc_labels=da.from_array(wc, chunks=(7, 7)),
        synth_geoms={},
        tiff_file=tiff_path,
        image_shape=(7, 7),
        percentiles=[50.0],
        tile_size=7,
        tile_overlap=0,
        threads=1,
    )

    assert 1 in measured
    props = measured[1]
    assert props["Cell: Area µm^2"] == 6.25
    assert "Cell: Area px" not in props
    assert "Cell: Length px" not in props
    assert "Cell: Max diameter px" not in props
    assert "Cell: Min diameter px" not in props
    assert "Nucleus: Length px" not in props
    assert props["Channel 1: Cell: Mean"] == 40.0
    assert props["Channel 1: Nucleus: Mean"] == 40.0
    assert props["Channel 1: Cell: Percentile: 50.0"] == 40.0
    assert any(k.startswith("Cell: ErosionBin_") for k in props)
    assert any(k.startswith("Cell: ExpansionBin_") for k in props)


def test_measure_cells_tiled_process_workers_match_serial(tmp_path: Path):
    img = np.zeros((10, 10), dtype=np.uint16)
    img[1:3, 1:3] = 5
    img[6:8, 6:8] = 11
    tiff_path = tmp_path / "img_parallel.tiff"
    _write_tiff(tiff_path, img)

    wc = np.zeros((10, 10), dtype=np.uint32)
    wc[1:3, 1:3] = 1
    wc[6:8, 6:8] = 2
    nuc = np.zeros((10, 10), dtype=np.uint32)
    nuc[1:2, 1:2] = 1
    nuc[6:7, 6:7] = 2

    cells = [
        CellMatch(
            cell_id=1,
            nucleus_label=1,
            whole_cell_label=1,
            bbox=(1, 1, 3, 3),
            centroid=(1.5, 1.5),
            nucleus_area_px=1,
            cell_area_px=4,
            overlap_px=1,
            overlap_fraction=1.0,
            match_source="overlap_1to1",
        ),
        CellMatch(
            cell_id=2,
            nucleus_label=2,
            whole_cell_label=2,
            bbox=(6, 6, 8, 8),
            centroid=(6.5, 6.5),
            nucleus_area_px=1,
            cell_area_px=4,
            overlap_px=1,
            overlap_fraction=1.0,
            match_source="overlap_1to1",
        ),
    ]

    common_kwargs = dict(
        cells=cells,
        nuc_labels=da.from_array(nuc, chunks=(5, 5)),
        wc_labels=da.from_array(wc, chunks=(5, 5)),
        synth_geoms={},
        tiff_file=tiff_path,
        image_shape=(10, 10),
        percentiles=[50.0],
        tile_size=5,
        tile_overlap=0,
        erosion_enabled=False,
        expansion_enabled=False,
    )

    serial = measure_cells_tiled(threads=1, **common_kwargs)
    parallel = measure_cells_tiled(threads=2, **common_kwargs)
    assert parallel == serial


def test_measure_cells_tiled_process_workers_use_worker_local_tiff_reads(monkeypatch, tmp_path: Path):
    img = np.zeros((10, 10), dtype=np.uint16)
    img[1:3, 1:3] = 5
    img[6:8, 6:8] = 11
    tiff_path = tmp_path / "img_process_tiff_reads.tiff"
    _write_tiff(tiff_path, img)

    wc = np.zeros((10, 10), dtype=np.uint32)
    wc[1:3, 1:3] = 1
    wc[6:8, 6:8] = 2
    nuc = np.zeros((10, 10), dtype=np.uint32)
    nuc[1:2, 1:2] = 1
    nuc[6:7, 6:7] = 2

    cells = [
        CellMatch(
            cell_id=1,
            nucleus_label=1,
            whole_cell_label=1,
            bbox=(1, 1, 3, 3),
            centroid=(1.5, 1.5),
            nucleus_area_px=1,
            cell_area_px=4,
            overlap_px=1,
            overlap_fraction=1.0,
            match_source="overlap_1to1",
        ),
        CellMatch(
            cell_id=2,
            nucleus_label=2,
            whole_cell_label=2,
            bbox=(6, 6, 8, 8),
            centroid=(6.5, 6.5),
            nucleus_area_px=1,
            cell_area_px=4,
            overlap_px=1,
            overlap_fraction=1.0,
            match_source="overlap_1to1",
        ),
    ]

    def _fail_prepare(*args, **kwargs):
        raise AssertionError("full image materialization path should not run in process mode")

    monkeypatch.setattr(measurement_pipeline, "_prepare_measurement_image_and_masks", _fail_prepare)

    measured = measure_cells_tiled(
        cells=cells,
        nuc_labels=da.from_array(nuc, chunks=(5, 5)),
        wc_labels=da.from_array(wc, chunks=(5, 5)),
        synth_geoms={},
        tiff_file=tiff_path,
        image_shape=(10, 10),
        tile_size=5,
        tile_overlap=0,
        threads=2,
        erosion_enabled=False,
        expansion_enabled=False,
    )
    assert set(measured.keys()) == {1, 2}


def test_measure_cells_tiled_validates_image_shape(tmp_path: Path):
    img = np.zeros((3, 3), dtype=np.uint16)
    tiff_path = tmp_path / "img_small.tiff"
    _write_tiff(tiff_path, img)

    cell = CellMatch(
        cell_id=1,
        nucleus_label=None,
        whole_cell_label=1,
        bbox=(0, 0, 1, 1),
        centroid=(0.0, 0.0),
        nucleus_area_px=0,
        cell_area_px=1,
        overlap_px=0,
        overlap_fraction=0.0,
        match_source="wc_only",
    )
    wc = da.from_array(np.array([[1]], dtype=np.uint32), chunks=(1, 1))

    with pytest.raises(ValueError, match="does not match segmentation shape"):
        measure_cells_tiled(
            cells=[cell],
            nuc_labels=None,
            wc_labels=wc,
            synth_geoms={},
            tiff_file=tiff_path,
            image_shape=(4, 4),
        )


def test_measure_cells_tiled_streams_jsonl(tmp_path: Path):
    img = np.zeros((5, 5), dtype=np.uint16)
    img[1:4, 1:4] = 10
    tiff_path = tmp_path / "img_stream.tiff"
    _write_tiff(tiff_path, img)

    wc = np.zeros((5, 5), dtype=np.uint32)
    wc[1:4, 1:4] = 1
    nuc = np.zeros((5, 5), dtype=np.uint32)
    nuc[2:3, 2:3] = 1

    cell = CellMatch(
        cell_id=1,
        nucleus_label=1,
        whole_cell_label=1,
        bbox=(1, 1, 4, 4),
        centroid=(2.0, 2.0),
        nucleus_area_px=1,
        cell_area_px=9,
        overlap_px=1,
        overlap_fraction=1.0,
        match_source="overlap_1to1",
    )

    jsonl_path = tmp_path / "measurements.jsonl"
    measured = measure_cells_tiled(
        cells=[cell],
        nuc_labels=da.from_array(nuc, chunks=(5, 5)),
        wc_labels=da.from_array(wc, chunks=(5, 5)),
        synth_geoms={},
        tiff_file=tiff_path,
        image_shape=(5, 5),
        tile_size=5,
        tile_overlap=0,
        threads=1,
        jsonl_path=jsonl_path,
        return_results=False,
    )

    assert measured == {}
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"cell_id":1' in lines[0]


def test_measure_cells_tiled_materializes_dask_labels_once(tmp_path: Path):
    img = np.ones((8, 8), dtype=np.uint16)
    tiff_path = tmp_path / "img_materialize_once.tiff"
    _write_tiff(tiff_path, img)

    wc = np.zeros((8, 8), dtype=np.uint32)
    wc[1:3, 1:3] = 1
    wc[5:7, 5:7] = 2
    nuc = np.zeros((8, 8), dtype=np.uint32)
    nuc[1:2, 1:2] = 1
    nuc[5:6, 5:6] = 2

    calls = {"nuc": 0, "wc": 0}

    def _load_nuc() -> np.ndarray:
        calls["nuc"] += 1
        return nuc

    def _load_wc() -> np.ndarray:
        calls["wc"] += 1
        return wc

    nuc_labels = da.from_delayed(dask.delayed(_load_nuc)(), shape=nuc.shape, dtype=nuc.dtype)
    wc_labels = da.from_delayed(dask.delayed(_load_wc)(), shape=wc.shape, dtype=wc.dtype)

    cells = [
        CellMatch(
            cell_id=1,
            nucleus_label=1,
            whole_cell_label=1,
            bbox=(1, 1, 3, 3),
            centroid=(1.5, 1.5),
            nucleus_area_px=1,
            cell_area_px=4,
            overlap_px=1,
            overlap_fraction=1.0,
            match_source="overlap_1to1",
        ),
        CellMatch(
            cell_id=2,
            nucleus_label=2,
            whole_cell_label=2,
            bbox=(5, 5, 7, 7),
            centroid=(5.5, 5.5),
            nucleus_area_px=1,
            cell_area_px=4,
            overlap_px=1,
            overlap_fraction=1.0,
            match_source="overlap_1to1",
        ),
    ]

    measured = measure_cells_tiled(
        cells=cells,
        nuc_labels=nuc_labels,
        wc_labels=wc_labels,
        synth_geoms={},
        tiff_file=tiff_path,
        image_shape=(8, 8),
        tile_size=4,
        tile_overlap=0,
        threads=1,
        erosion_enabled=False,
        expansion_enabled=False,
    )

    assert set(measured.keys()) == {1, 2}
    assert calls["nuc"] == 1
    assert calls["wc"] == 1


def test_measure_cells_tiled_uses_tiff_channel_names(tmp_path: Path):
    img = np.zeros((2, 5, 5), dtype=np.uint16)  # (C, Y, X)
    img[0, 1:4, 1:4] = 7
    img[1, 1:4, 1:4] = 11
    tiff_path = tmp_path / "img_channels.ome.tiff"
    tifffile.imwrite(
        tiff_path,
        img,
        metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "CD3"]}},
    )

    wc = np.zeros((5, 5), dtype=np.uint32)
    wc[1:4, 1:4] = 1
    nuc = np.zeros((5, 5), dtype=np.uint32)
    nuc[2:3, 2:3] = 1

    cell = CellMatch(
        cell_id=1,
        nucleus_label=1,
        whole_cell_label=1,
        bbox=(1, 1, 4, 4),
        centroid=(2.0, 2.0),
        nucleus_area_px=1,
        cell_area_px=9,
        overlap_px=1,
        overlap_fraction=1.0,
        match_source="overlap_1to1",
    )

    measured = measure_cells_tiled(
        cells=[cell],
        nuc_labels=da.from_array(nuc, chunks=(5, 5)),
        wc_labels=da.from_array(wc, chunks=(5, 5)),
        synth_geoms={},
        tiff_file=tiff_path,
        image_shape=(5, 5),
        tile_size=5,
        tile_overlap=0,
        threads=1,
    )

    props = measured[1]
    assert "DAPI: Cell: Mean" in props
    assert "CD3: Cell: Mean" in props


def test_measure_cells_tiled_can_disable_erosion_and_expansion(tmp_path: Path):
    img = np.zeros((5, 5), dtype=np.uint16)
    img[1:4, 1:4] = 10
    tiff_path = tmp_path / "img_no_steps.tiff"
    _write_tiff(tiff_path, img)

    wc = np.zeros((5, 5), dtype=np.uint32)
    wc[1:4, 1:4] = 1
    nuc = np.zeros((5, 5), dtype=np.uint32)
    nuc[2:3, 2:3] = 1

    cell = CellMatch(
        cell_id=1,
        nucleus_label=1,
        whole_cell_label=1,
        bbox=(1, 1, 4, 4),
        centroid=(2.0, 2.0),
        nucleus_area_px=1,
        cell_area_px=9,
        overlap_px=1,
        overlap_fraction=1.0,
        match_source="overlap_1to1",
    )

    measured = measure_cells_tiled(
        cells=[cell],
        nuc_labels=da.from_array(nuc, chunks=(5, 5)),
        wc_labels=da.from_array(wc, chunks=(5, 5)),
        synth_geoms={},
        tiff_file=tiff_path,
        image_shape=(5, 5),
        tile_size=5,
        tile_overlap=0,
        threads=1,
        erosion_enabled=False,
        expansion_enabled=False,
    )

    props = measured[1]
    assert not any(k.startswith("Cell: ErosionBin_") for k in props)
    assert not any(k.startswith("Cell: ExpansionBin_") for k in props)


def test_measure_cells_tiled_environment_expansion(tmp_path: Path):
    img = np.ones((15, 15), dtype=np.uint16) * 7
    tiff_path = tmp_path / "img_environment.tiff"
    _write_tiff(tiff_path, img)

    wc = np.zeros((15, 15), dtype=np.uint32)
    wc[6:9, 6:9] = 1

    cell = CellMatch(
        cell_id=1,
        nucleus_label=None,
        whole_cell_label=1,
        bbox=(6, 6, 9, 9),
        centroid=(7.0, 7.0),
        nucleus_area_px=0,
        cell_area_px=9,
        overlap_px=0,
        overlap_fraction=0.0,
        match_source="wc_only",
    )

    measured = measure_cells_tiled(
        cells=[cell],
        nuc_labels=None,
        wc_labels=da.from_array(wc, chunks=(15, 15)),
        synth_geoms={},
        tiff_file=tiff_path,
        image_shape=(15, 15),
        tile_size=15,
        tile_overlap=0,
        threads=1,
        expansion_enabled=False,
        environment_expansion_enabled=True,
        pixel_size_microns=10.0,  # 20 µm => 2 px dilation radius
    )

    props = measured[1]
    assert props["Cell: Environment_20um: Pixel_Count"] > 0
    assert "Channel 1: Cell: Environment_20um: Mean" in props
    assert props["Channel 1: Cell: Environment_20um: Mean"] == 7.0


def test_measure_cells_tiled_wc_only_skips_cytoplasm_keeps_membrane(tmp_path: Path):
    img = np.zeros((7, 7), dtype=np.uint16)
    img[1:6, 1:6] = 10
    tiff_path = tmp_path / "img_wc_only_compartments.tiff"
    _write_tiff(tiff_path, img)

    wc = np.zeros((7, 7), dtype=np.uint32)
    wc[1:6, 1:6] = 1

    cell = CellMatch(
        cell_id=1,
        nucleus_label=None,
        whole_cell_label=1,
        bbox=(1, 1, 6, 6),
        centroid=(3.0, 3.0),
        nucleus_area_px=0,
        cell_area_px=25,
        overlap_px=0,
        overlap_fraction=0.0,
        match_source="wc_only",
    )

    measured = measure_cells_tiled(
        cells=[cell],
        nuc_labels=None,
        wc_labels=da.from_array(wc, chunks=(7, 7)),
        synth_geoms={},
        tiff_file=tiff_path,
        image_shape=(7, 7),
        percentiles=[50.0],
        tile_size=7,
        tile_overlap=0,
        threads=1,
        erosion_enabled=False,
        expansion_enabled=False,
    )

    props = measured[1]
    assert not any(": Cytoplasm:" in key for key in props)
    assert "Channel 1: Membrane: Mean" in props
    assert "Channel 1: Membrane: Percentile: 50.0" in props


def test_measure_cells_tiled_nuc_only_skips_membrane_and_cytoplasm(tmp_path: Path):
    img = np.zeros((7, 7), dtype=np.uint16)
    img[2:5, 2:5] = 11
    tiff_path = tmp_path / "img_nuc_only_compartments.tiff"
    _write_tiff(tiff_path, img)

    nuc = np.zeros((7, 7), dtype=np.uint32)
    nuc[2:5, 2:5] = 1

    cell = CellMatch(
        cell_id=1,
        nucleus_label=1,
        whole_cell_label=None,
        bbox=(2, 2, 5, 5),
        centroid=(3.0, 3.0),
        nucleus_area_px=9,
        cell_area_px=9,
        overlap_px=0,
        overlap_fraction=0.0,
        match_source="nuc_only",
    )

    measured = measure_cells_tiled(
        cells=[cell],
        nuc_labels=da.from_array(nuc, chunks=(7, 7)),
        wc_labels=None,
        synth_geoms={},
        tiff_file=tiff_path,
        image_shape=(7, 7),
        percentiles=[50.0],
        tile_size=7,
        tile_overlap=0,
        threads=1,
        erosion_enabled=False,
        expansion_enabled=False,
    )

    props = measured[1]
    assert "Channel 1: Cell: Mean" in props
    assert "Channel 1: Nucleus: Mean" in props
    assert not any(": Membrane:" in key for key in props)
    assert not any(": Cytoplasm:" in key for key in props)


def test_measure_cells_tiled_neighbour_aggregation(tmp_path: Path):
    img = np.ones((10, 10), dtype=np.uint16)
    tiff_path = tmp_path / "img_neighbours.tiff"
    _write_tiff(tiff_path, img)

    wc = np.zeros((10, 10), dtype=np.uint32)
    wc[1:3, 1:3] = 1  # area 4
    wc[1:4, 5:8] = 2  # area 9

    cells = [
        CellMatch(
            cell_id=1,
            nucleus_label=None,
            whole_cell_label=1,
            bbox=(1, 1, 3, 3),
            centroid=(1.5, 1.5),
            nucleus_area_px=0,
            cell_area_px=4,
            overlap_px=0,
            overlap_fraction=0.0,
            match_source="wc_only",
        ),
        CellMatch(
            cell_id=2,
            nucleus_label=None,
            whole_cell_label=2,
            bbox=(1, 5, 4, 8),
            centroid=(2.0, 6.0),
            nucleus_area_px=0,
            cell_area_px=9,
            overlap_px=0,
            overlap_fraction=0.0,
            match_source="wc_only",
        ),
    ]

    measured = measure_cells_tiled(
        cells=cells,
        nuc_labels=None,
        wc_labels=da.from_array(wc, chunks=(10, 10)),
        synth_geoms={},
        tiff_file=tiff_path,
        image_shape=(10, 10),
        tile_size=10,
        tile_overlap=0,
        threads=1,
        erosion_enabled=False,
        expansion_enabled=False,
        neighbours=1,
        pixel_size_microns=1.0,
    )

    assert measured[1]["Neighbours: Mean: Cell: Area µm^2"] == 9.0
    assert measured[2]["Neighbours: Mean: Cell: Area µm^2"] == 4.0


def test_measure_cells_tiled_streams_jsonl_with_neighbours(tmp_path: Path):
    img = np.ones((8, 8), dtype=np.uint16)
    tiff_path = tmp_path / "img_stream_neighbours.tiff"
    _write_tiff(tiff_path, img)

    wc = np.zeros((8, 8), dtype=np.uint32)
    wc[1:3, 1:3] = 1
    wc[1:3, 4:6] = 2

    cells = [
        CellMatch(
            cell_id=1,
            nucleus_label=None,
            whole_cell_label=1,
            bbox=(1, 1, 3, 3),
            centroid=(1.5, 1.5),
            nucleus_area_px=0,
            cell_area_px=4,
            overlap_px=0,
            overlap_fraction=0.0,
            match_source="wc_only",
        ),
        CellMatch(
            cell_id=2,
            nucleus_label=None,
            whole_cell_label=2,
            bbox=(1, 4, 3, 6),
            centroid=(1.5, 4.5),
            nucleus_area_px=0,
            cell_area_px=4,
            overlap_px=0,
            overlap_fraction=0.0,
            match_source="wc_only",
        ),
    ]

    jsonl_path = tmp_path / "measurements-neighbours.jsonl"
    measured = measure_cells_tiled(
        cells=cells,
        nuc_labels=None,
        wc_labels=da.from_array(wc, chunks=(8, 8)),
        synth_geoms={},
        tiff_file=tiff_path,
        image_shape=(8, 8),
        tile_size=8,
        tile_overlap=0,
        threads=1,
        erosion_enabled=False,
        expansion_enabled=False,
        neighbours=1,
        jsonl_path=jsonl_path,
        return_results=False,
        pixel_size_microns=1.0,
    )

    assert measured == {}
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    payloads = [json.loads(line) for line in lines if line.strip()]
    assert payloads
    first_measurements = payloads[0]["measurements"]
    assert "Neighbours: Mean: Cell: Area µm^2" in first_measurements


def test_measure_cells_tiled_clips_circularity_and_filters_tiny_objects(tmp_path: Path):
    img = np.ones((7, 7), dtype=np.uint16)
    tiff_path = tmp_path / "img_circularity.tiff"
    _write_tiff(tiff_path, img)

    wc = np.zeros((7, 7), dtype=np.uint32)
    wc[1:6, 1:6] = 1
    nuc = np.zeros((7, 7), dtype=np.uint32)
    # Thin 3-pixel nucleus shape that can produce unstable circularity with pixel perimeter.
    nuc[3, 2:5] = 1

    cell = CellMatch(
        cell_id=1,
        nucleus_label=1,
        whole_cell_label=1,
        bbox=(1, 1, 6, 6),
        centroid=(3.0, 3.0),
        nucleus_area_px=3,
        cell_area_px=25,
        overlap_px=3,
        overlap_fraction=1.0,
        match_source="overlap_1to1",
    )

    measured = measure_cells_tiled(
        cells=[cell],
        nuc_labels=da.from_array(nuc, chunks=(7, 7)),
        wc_labels=da.from_array(wc, chunks=(7, 7)),
        synth_geoms={},
        tiff_file=tiff_path,
        image_shape=(7, 7),
        tile_size=7,
        tile_overlap=0,
        threads=1,
        erosion_enabled=False,
        expansion_enabled=False,
    )

    props = measured[1]
    assert 0.0 <= props["Cell: Circularity"] <= 1.0
    assert 0.0 <= props["Nucleus: Circularity"] <= 1.0
    assert props["Nucleus: Circularity"] == 0.0
