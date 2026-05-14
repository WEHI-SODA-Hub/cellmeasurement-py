from __future__ import annotations

import json

import numpy as np
import tifffile

from cellmeasurement.measurement import image_io


def test_load_tiff_image_multpage_non_mibi_stacks_pages(tmp_path):
    p0 = np.full((4, 5), 11, dtype=np.uint16)
    p1 = np.full((4, 5), 22, dtype=np.uint16)
    tiff_path = tmp_path / "opal_like.tiff"
    with tifffile.TiffWriter(tiff_path) as tw:
        tw.write(p0, description="plain description")
        tw.write(p1, description="plain description")

    image_cyx, ch_names = image_io._load_tiff_image(tiff_path)

    assert image_cyx.shape == (2, 4, 5)
    np.testing.assert_array_equal(image_cyx[0], p0)
    np.testing.assert_array_equal(image_cyx[1], p1)
    assert ch_names == ["Channel 1", "Channel 2"]


def test_load_tiff_image_mibi_uses_channel_target_names(tmp_path):
    p0 = np.full((3, 3), 1, dtype=np.uint16)
    p1 = np.full((3, 3), 2, dtype=np.uint16)
    tiff_path = tmp_path / "mibi_like.tiff"
    with tifffile.TiffWriter(tiff_path) as tw:
        tw.write(p0, description=json.dumps({"channel.target": "CD3"}))
        tw.write(p1, description=json.dumps({"channel.target": "CD8"}))

    image_cyx, ch_names = image_io._load_tiff_image(tiff_path)

    assert image_cyx.shape == (2, 3, 3)
    np.testing.assert_array_equal(image_cyx[0], p0)
    np.testing.assert_array_equal(image_cyx[1], p1)
    assert ch_names == ["CD3", "CD8"]


def test_load_tiff_image_name_mismatch_falls_back(monkeypatch, tmp_path):
    arr = np.arange(40, dtype=np.uint16).reshape(2, 4, 5)
    tiff_path = tmp_path / "mismatch.tiff"
    tifffile.imwrite(tiff_path, arr)

    monkeypatch.setattr(image_io, "_extract_channel_names", lambda _tf: ["OnlyOneName"])

    image_cyx, ch_names = image_io._load_tiff_image(tiff_path)

    assert image_cyx.shape == (2, 4, 5)
    assert ch_names == ["Channel 1", "Channel 2"]


def test_channel_names_from_ome_reads_channel_name():
    class _FakeTF:
        ome_metadata = (
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            '<Image ID="Image:0"><Pixels>'
            '<Channel ID="Channel:0:0" Name="DAPI" />'
            "</Pixels></Image></OME>"
        )
        pages = []

    assert image_io._channel_names_from_ome(_FakeTF()) == ["DAPI"]


def test_is_comet_tiff_detects_microscope_and_detector_metadata():
    class _FakeTF:
        ome_metadata = (
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            '<Instrument ID="Instrument:0">'
            '<Microscope Manufacturer="Lunaphore Technologies SA" Model="Comet" />'
            '<Detector ID="Detector:0" Manufacturer="Lunaphore" Model="COMET1_SV1" Type="CMOS" />'
            "</Instrument></OME>"
        )
        pages = []

    assert image_io._is_comet_tiff(_FakeTF()) is True


def test_is_comet_tiff_false_for_non_comet_ome():
    class _FakeTF:
        ome_metadata = (
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            '<Instrument ID="Instrument:0">'
            '<Microscope Manufacturer="Acme" Model="ScopeX" />'
            '<Detector ID="Detector:0" Manufacturer="Acme" Model="D1" Type="CMOS" />'
            "</Instrument></OME>"
        )
        pages = []

    assert image_io._is_comet_tiff(_FakeTF()) is False


def test_is_comet_tiff_detects_objective_metadata():
    class _FakeTF:
        ome_metadata = (
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            '<Instrument ID="Instrument:0">'
            '<Objective ID="Objective:0" Manufacturer="Lunaphore" Model="COMET" '
            'LensNA="0.75" WorkingDistance="1000.0" NominalMagnification="20.0" />'
            "</Instrument></OME>"
        )
        pages = []

    assert image_io._is_comet_tiff(_FakeTF()) is True


def test_select_non_mibi_fullres_pages_filters_by_first_page_shape():
    class _Page:
        def __init__(self, shape):
            self.shape = shape

    class _FakeTF:
        pages = [_Page((100, 200)), _Page((100, 200)), _Page((50, 100)), _Page((25, 50))]

    selected = image_io._select_non_mibi_fullres_pages(_FakeTF())
    assert len(selected) == 2
    assert all(page.shape == (100, 200) for page in selected)
