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

import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile
from shapely.geometry import Polygon, mapping, shape
from skimage.draw import polygon as draw_polygon

from ..geometry.overlap_constraint import constrain_cell_overlaps

if TYPE_CHECKING:
    from ..segmentation.cell import CellMatch

# SynthGeoms maps cell_id -> pre-simplified Polygon for watershed_synth cells.
SynthGeoms = dict[int, Polygon]


def _rasterise_features_to_mask(
    features: list[dict],
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterise final cell polygons to an integer label mask.

    Uses post-constraint feature geometries so raster output matches the final
    exported non-overlapping polygons.
    """
    max_id = max(
        (
            int(feat.get("properties", {}).get("id", 0))
            for feat in features
            if feat.get("properties", {}).get("objectType") == "cell"
        ),
        default=0,
    )
    dtype = np.int32 if max_id < (2**31) else np.int64
    mask = np.zeros((height, width), dtype=dtype)

    for feat in features:
        props = feat.get("properties", {})
        if props.get("objectType") != "cell":
            continue
        cell_id = int(props.get("id", 0))
        if cell_id <= 0:
            continue

        geom = shape(feat["geometry"])
        if geom.is_empty:
            continue

        polygons: list[Polygon]
        if geom.geom_type == "Polygon":
            polygons = [geom]  # type: ignore[list-item]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)  # type: ignore[assignment]
        else:
            continue

        for poly in polygons:
            exterior = np.asarray(poly.exterior.coords)
            if exterior.shape[0] < 3:
                continue
            rr, cc = draw_polygon(exterior[:, 1], exterior[:, 0], shape=(height, width))
            mask[rr, cc] = cell_id
            for hole in poly.interiors:
                interior = np.asarray(hole.coords)
                if interior.shape[0] < 3:
                    continue
                rr_h, cc_h = draw_polygon(interior[:, 1], interior[:, 0], shape=(height, width))
                mask[rr_h, cc_h] = 0

    return mask


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
    measurements: dict[str, float] | None = None,
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
    if measurements is not None:
        feature["properties"]["measurements"] = measurements
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
    measurements_by_cell: dict[int, dict[str, float]] | None = None,
    measurements_jsonl_path: Path | None = None,
    constrain_overlaps: bool = True,
    pretty: bool = False,
    gzip_output: bool = False,
    output_mask: Path | None = None,
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
        measurements_by_cell: Optional mapping of ``cell_id`` to precomputed
            measurement key/value pairs to attach under
            ``feature.properties.measurements``.
        measurements_jsonl_path: Optional JSONL path with one record per line:
            ``{"cell_id": int, "measurements": {...}}``.
        constrain_overlaps: Whether to run overlap clipping so no two cell
            polygons share area.
        pretty: Write indented (human-readable) JSON.
        gzip_output: Write gzip-compressed GeoJSON; appends ``.gz`` suffix if
            needed.
        output_mask: Optional TIFF path for rasterised final cell
            polygons.

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

    if measurements_by_cell is not None and measurements_jsonl_path is not None:
        raise ValueError("Provide either measurements_by_cell or measurements_jsonl_path, not both.")

    def _iter_measurements_jsonl(path: Path):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cell_id = int(rec["cell_id"])
                measurements = rec.get("measurements", {})
                yield cell_id, measurements if isinstance(measurements, dict) else {}

    jsonl_iter = None
    jsonl_current = None
    if measurements_jsonl_path is not None and measurements_jsonl_path.exists():
        jsonl_iter = _iter_measurements_jsonl(measurements_jsonl_path)
        jsonl_current = next(jsonl_iter, None)

    features: list[dict] = []
    for cell in sorted(cells, key=lambda c: c.cell_id):
        cell_measurements: dict[str, float] | None = None
        if measurements_by_cell is not None:
            cell_measurements = measurements_by_cell.get(cell.cell_id)
        elif jsonl_iter is not None:
            while jsonl_current is not None and jsonl_current[0] < cell.cell_id:
                jsonl_current = next(jsonl_iter, None)
            if jsonl_current is not None and jsonl_current[0] == cell.cell_id:
                cell_measurements = jsonl_current[1]
                jsonl_current = next(jsonl_iter, None)

        feat = _cell_to_feature(
            cell,
            nuc_geoms,
            wc_geoms,
            synth_geoms,
            measurements=cell_measurements,
        )
        if feat is not None:
            features.append(feat)

    if constrain_overlaps:
        features = constrain_cell_overlaps(features)

    collection: dict = {
        "type": "FeatureCollection",
        "features": [annotation] + features,
    }

    final_output_path = output_path
    if gzip_output and not str(final_output_path).endswith(".gz"):
        final_output_path = Path(str(final_output_path) + ".gz")

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_output:
        with gzip.open(final_output_path, "wt", encoding="utf-8") as f:
            if pretty:
                json.dump(collection, f, indent=2)
            else:
                json.dump(collection, f, separators=(",", ":"))
    else:
        with final_output_path.open("w", encoding="utf-8") as f:
            if pretty:
                json.dump(collection, f, indent=2)
            else:
                json.dump(collection, f, separators=(",", ":"))

    if output_mask is not None:
        raster_mask = _rasterise_features_to_mask(features, H, W)
        output_mask.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(output_mask), raster_mask)

    return len(features)
