from __future__ import annotations

from pathlib import Path
import json

from shapely.geometry import Polygon, shape

from cellmeasurement.io.geojson_writer import write_geojson
from cellmeasurement.geometry.overlap_constraint import constrain_cell_overlaps
from cellmeasurement.segmentation.cell import CellMatch


def _feature(cell_id: int, cell_poly: Polygon, nucleus_poly: Polygon | None = None) -> dict:
    feat = {
        "type": "Feature",
        "id": f"cell-{cell_id}",
        "geometry": cell_poly.__geo_interface__,
        "properties": {"objectType": "cell", "id": cell_id},
    }
    if nucleus_poly is not None:
        feat["nucleusGeometry"] = nucleus_poly.__geo_interface__
    return feat


def test_constrain_overlaps_trims_larger_cell():
    big = Polygon([(0, 0), (6, 0), (6, 6), (0, 6)])
    small = Polygon([(4, 2), (7, 2), (7, 5), (4, 5)])

    out = constrain_cell_overlaps([_feature(1, big), _feature(2, small)])

    assert len(out) == 2
    g0 = shape(out[0]["geometry"])
    g1 = shape(out[1]["geometry"])
    assert g0.intersection(g1).area < 1e-10
    assert g0.area < big.area


def test_constrain_overlaps_keeps_largest_fragment_only():
    # big minus splitter creates two fragments; only largest polygon is kept.
    big = Polygon([(0, 0), (8, 0), (8, 2), (0, 2)])
    splitter = Polygon([(3, -1), (5, -1), (5, 3), (3, 3)])

    out = constrain_cell_overlaps([_feature(1, big), _feature(2, splitter)])

    kept = shape(out[0]["geometry"])
    assert kept.geom_type == "Polygon"
    assert kept.area < big.area


def test_constrain_overlaps_drops_empty_cell():
    a = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    b = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])

    out = constrain_cell_overlaps([_feature(1, a), _feature(2, b)])

    assert len(out) == 1


def test_constrain_overlaps_clips_or_removes_nucleus_geometry():
    cell = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
    nucleus_outside = Polygon([(4, 4), (5, 4), (5, 5), (4, 5)])

    out = constrain_cell_overlaps([_feature(1, cell, nucleus_outside)])

    assert len(out) == 1
    assert "nucleusGeometry" not in out[0]


def test_write_geojson_can_disable_overlap_constraint(tmp_path: Path):
    cells = [
        CellMatch(
            cell_id=1,
            nucleus_label=None,
            whole_cell_label=1,
            bbox=(0, 0, 2, 2),
            centroid=(1.0, 1.0),
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
            bbox=(0, 0, 2, 2),
            centroid=(1.0, 1.0),
            nucleus_area_px=0,
            cell_area_px=4,
            overlap_px=0,
            overlap_fraction=0.0,
            match_source="wc_only",
        ),
    ]
    wc_geoms = {
        1: Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        2: Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
    }

    out_path_on = tmp_path / "constrained.geojson"
    out_path_off = tmp_path / "unconstrained.geojson"

    n_on = write_geojson(
        cells=cells,
        nuc_geoms=None,
        wc_geoms=wc_geoms,
        synth_geoms={},
        output_path=out_path_on,
        image_shape=(10, 10),
        constrain_overlaps=True,
    )
    n_off = write_geojson(
        cells=cells,
        nuc_geoms=None,
        wc_geoms=wc_geoms,
        synth_geoms={},
        output_path=out_path_off,
        image_shape=(10, 10),
        constrain_overlaps=False,
    )

    assert n_on == 1
    assert n_off == 2

    with out_path_on.open(encoding="utf-8") as f:
        constrained = json.load(f)
    with out_path_off.open(encoding="utf-8") as f:
        unconstrained = json.load(f)

    assert len(constrained["features"]) == 2  # annotation + 1 cell
    assert len(unconstrained["features"]) == 3  # annotation + 2 cells
