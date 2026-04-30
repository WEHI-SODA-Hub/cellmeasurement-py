from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import tifffile

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
