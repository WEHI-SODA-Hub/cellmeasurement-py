from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def _ome_xml(channel_names: list[str], size_y: int, size_x: int, dtype: str) -> str:
    channels = "".join(
        f'<Channel ID="Channel:0:{i}" Name="{name}" SamplesPerPixel="1"/>'
        for i, name in enumerate(channel_names)
    )
    tiffdata = "".join(
        f'<TiffData FirstC="{i}" FirstT="0" FirstZ="0" IFD="{i}" PlaneCount="1"/>'
        for i in range(len(channel_names))
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<OME xmlns="{OME_NS}"><Image ID="Image:0">'
        f'<Pixels DimensionOrder="XYCZT" ID="Pixels:0" Type="{dtype}" '
        f'SizeX="{size_x}" SizeY="{size_y}" SizeC="{len(channel_names)}" SizeZ="1" SizeT="1">'
        f"{channels}{tiffdata}</Pixels></Image></OME>"
    )


@pytest.fixture
def shaped_split_ome_tiff(tmp_path: Path):
    """Write an OME-TIFF whose channel axis tifffile's "shaped" detection splits apart.

    Reproduces the layout produced when an OME-TIFF is rewritten page-by-page with
    tifffile: page 0 carries the OME-XML, and every later page carries tifffile's own
    ``{"shape": [...]}`` description. tifffile then prefers its "shaped" series
    detection and reports one single-channel series per page instead of one (C, Y, X)
    series, hiding every channel but the first.
    """

    def _write(channel_names: list[str], *, size_y: int = 4, size_x: int = 5) -> tuple[Path, np.ndarray]:
        data = np.stack(
            [np.full((size_y, size_x), i + 1, dtype=np.uint16) for i in range(len(channel_names))]
        )
        path = tmp_path / "shaped_split.ome.tif"
        with tifffile.TiffWriter(path) as tw:
            tw.write(data[0], description=_ome_xml(channel_names, size_y, size_x, "uint16"))
            for plane in data[1:]:
                tw.write(plane)
        return path, data

    return _write
