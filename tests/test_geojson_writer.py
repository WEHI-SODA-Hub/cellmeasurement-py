from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from cellmeasurement.io.geojson_writer import _rasterise_features_to_mask


def _cell_feature(cell_id: int, polygon: Polygon) -> dict:
    return {
        "type": "Feature",
        "id": f"cell-{cell_id}",
        "geometry": polygon.__geo_interface__,
        "properties": {"objectType": "cell", "id": cell_id},
    }


def test_rasterise_preserves_cell_enclosed_in_hole():
    """A small cell sitting in a larger cell's hole must survive rasterisation.

    Reproduces the engulfing bug: ``constrain_cell_overlaps`` turns the larger
    cell into a donut whose hole is filled by the smaller cell.  The old
    rasteriser painted the larger cell's solid exterior over the smaller cell
    and then zeroed the hole, deleting the smaller cell from the mask while it
    remained in the GeoJSON.
    """
    # Larger cell is a donut; smaller cell exactly fills the hole.
    shell = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
    big = Polygon(shell, [hole])
    small = Polygon(hole)

    # Smaller cell has the lower id, so under the old ascending-id draw order it
    # was painted first and then overwritten by the larger, higher-id cell.
    features = [_cell_feature(1, small), _cell_feature(2, big)]

    mask = _rasterise_features_to_mask(features, height=10, width=10)

    labels = set(np.unique(mask).tolist())
    assert 1 in labels, "enclosed cell was engulfed and lost from the mask"
    assert 2 in labels
    # Centre of the hole belongs to the small cell, not the surrounding donut.
    assert mask[5, 5] == 1
    assert int((mask == 1).sum()) > 0


def test_rasterise_keeps_empty_hole_as_background():
    """A donut with no enclosed cell keeps its hole as background (0)."""
    shell = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
    donut = Polygon(shell, [hole])

    mask = _rasterise_features_to_mask([_cell_feature(1, donut)], height=10, width=10)

    assert mask[5, 5] == 0
    assert 1 in set(np.unique(mask).tolist())


def test_rasterise_smaller_cell_wins_overlap():
    """With overlap clipping disabled, a smaller cell is not engulfed by a larger one."""
    big = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    small = Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])  # fully inside big, overlapping

    # Larger cell has the higher id; under the old code it was drawn last and
    # engulfed the smaller cell.
    features = [_cell_feature(1, small), _cell_feature(2, big)]

    mask = _rasterise_features_to_mask(features, height=10, width=10)

    assert mask[5, 5] == 1, "smaller overlapping cell should retain its pixels"
    assert int((mask == 1).sum()) > 0
