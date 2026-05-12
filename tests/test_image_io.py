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
