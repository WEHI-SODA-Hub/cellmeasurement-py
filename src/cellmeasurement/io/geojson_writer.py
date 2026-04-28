"""GeoJSON export for cell match results.

Converts pre-extracted Shapely polygon geometry dicts (produced by
:func:`~cellmeasurement.geometry.geometry.extract_label_geometries`) to a
QuPath-compatible GeoJSON FeatureCollection.

Output structure
----------------
* One top-level ``annotation`` feature covering the full image extent.
* One ``cell`` feature per :class:`~cellmeasurement.segmentation.cell.CellMatch`,
  with a ``geometry`` (cell boundary) and optional ``nucleusGeometry`` (nucleus
  boundary) as a top-level key, following QuPath convention.

Geometry sources per ``match_source``:

* ``overlap_1to1`` — polygon from ``wc_geoms[whole_cell_label]``;
  nucleus from ``nuc_geoms[nucleus_label]``.
* ``watershed_synth`` — polygon from ``synth_geoms[cell_id]``;
  nucleus from ``nuc_geoms[nucleus_label]``.
* ``wc_only`` — polygon from ``wc_geoms[whole_cell_label]``.
* ``nuc_only`` — polygon from ``nuc_geoms[nucleus_label]``
  (nucleus used as the cell boundary; no separate nucleusGeometry).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from shapely.geometry import Polygon, mapping

from ..geometry.overlap_constraint import constrain_cell_overlaps

if TYPE_CHECKING:
    from ..segmentation.cell import CellMatch

# SynthGeoms maps cell_id -> pre-simplified Polygon for watershed_synth cells.
SynthGeoms = dict[int, Polygon]


def _extract_geometries(
    cell: CellMatch,
    nuc_geoms: dict[int, Polygon] | None,
    wc_geoms: dict[int, Polygon] | None,
    synth_geoms: SynthGeoms,
) -> tuple[Polygon | None, Polygon | None]:
    """Return ``(cell_polygon, nucleus_polygon)`` for one cell.

    Looks up pre-computed geometry dicts rather than re-extracting from rasters.
    Logs a warning when an expected geometry entry is absent (label cropped away
    or too small to contour).
    """
    import logging
    log = logging.getLogger(__name__)

    src = cell.match_source
    cell_geom: Polygon | None = None
    nuc_geom: Polygon | None = None

    if src == "watershed_synth":
        cell_geom = synth_geoms.get(cell.cell_id)
        if cell_geom is None:
            log.warning("No synth geometry for cell_id=%d", cell.cell_id)
        if nuc_geoms is not None and cell.nucleus_label is not None:
            nuc_geom = nuc_geoms.get(cell.nucleus_label)

    elif src == "overlap_1to1":
        if wc_geoms is not None and cell.whole_cell_label is not None:
            cell_geom = wc_geoms.get(cell.whole_cell_label)
        if nuc_geoms is not None and cell.nucleus_label is not None:
            nuc_geom = nuc_geoms.get(cell.nucleus_label)

    elif src == "wc_only":
        if wc_geoms is not None and cell.whole_cell_label is not None:
            cell_geom = wc_geoms.get(cell.whole_cell_label)

    elif src == "nuc_only":
        if nuc_geoms is not None and cell.nucleus_label is not None:
            cell_geom = nuc_geoms.get(cell.nucleus_label)

    return cell_geom, nuc_geom


def _cell_to_feature(
    cell: CellMatch,
    nuc_geoms: dict[int, Polygon] | None,
    wc_geoms: dict[int, Polygon] | None,
    synth_geoms: SynthGeoms,
) -> dict | None:
    """Build a GeoJSON Feature dict for a single cell.

    Returns ``None`` if no valid cell geometry is available.
    """
    cell_geom, nuc_geom = _extract_geometries(cell, nuc_geoms, wc_geoms, synth_geoms)
    if cell_geom is None:
        return None

    feature: dict = {
        "type": "Feature",
        "id": f"cell-{cell.cell_id}",
        "geometry": mapping(cell_geom),
        "properties": {
            "objectType": "cell",
            "id": cell.cell_id,
            "nucleus_label": cell.nucleus_label,
            "whole_cell_label": cell.whole_cell_label,
            "match_source": cell.match_source,
        },
    }
    if nuc_geom is not None:
        feature["nucleusGeometry"] = mapping(nuc_geom)

    return feature


def write_geojson(
    cells: list[CellMatch],
    nuc_geoms: dict[int, Polygon] | None,
    wc_geoms: dict[int, Polygon] | None,
    synth_geoms: SynthGeoms,
    output_path: Path,
    image_shape: tuple[int, int],
    constrain_overlaps: bool = True,
    pretty: bool = False,
) -> int:
    """Write cell matches to a GeoJSON FeatureCollection.

    Produces a file compatible with QuPath: a whole-image ``annotation``
    feature followed by one ``cell`` feature per entry in *cells*.

    All polygon geometries must be pre-extracted and simplified before calling
    this function.  Use
    :func:`~cellmeasurement.geometry.geometry.extract_label_geometries` for
    label-backed masks. Synthesised-cell polygons are provided via *synth_geoms*.

    Args:
        cells: List of :class:`~cellmeasurement.segmentation.cell.CellMatch`
            objects returned by
            :func:`~cellmeasurement.segmentation.roi_matcher.match_rois`.
        nuc_geoms: Pre-extracted nucleus polygons keyed by nucleus label ID,
            or ``None`` when only whole-cell segmentation is available.
        wc_geoms: Pre-extracted whole-cell polygons keyed by whole-cell label
            ID, or ``None`` when only nuclear segmentation is available.
        synth_geoms: Pre-extracted synthesised-cell polygons keyed by
            ``cell_id`` (from :data:`SynthGeoms`).
        output_path: Destination ``.geojson`` file path.
        image_shape: ``(height, width)`` of the full image; used to build the
            annotation bounding-box feature.
        constrain_overlaps: Whether to run overlap clipping so no two cell
            polygons share area.
        pretty: Write indented (human-readable) JSON.

    Returns:
        Number of cell features written (excluding the annotation).
    """
    H, W = image_shape

    annotation: dict = {
        "type": "Feature",
        "id": "annotation-whole-image",
        "geometry": mapping(Polygon([(0, 0), (W, 0), (W, H), (0, H), (0, 0)])),
        "properties": {
            "objectType": "annotation",
            "type": "annotation",
            "name": "whole_image",
        },
    }

    features: list[dict] = []
    for cell in cells:
        feat = _cell_to_feature(cell, nuc_geoms, wc_geoms, synth_geoms)
        if feat is not None:
            features.append(feat)

    if constrain_overlaps:
        features = constrain_cell_overlaps(features)

    collection: dict = {
        "type": "FeatureCollection",
        "features": [annotation] + features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(collection, f, indent=2)
        else:
            json.dump(collection, f, separators=(",", ":"))

    return len(features)
