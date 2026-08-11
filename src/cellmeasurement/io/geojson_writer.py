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
import logging
import time
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
logger = logging.getLogger(__name__)


# Label masks are long runs of a repeated ID, so deflate shrinks them ~90x.
MASK_COMPRESSION = "zlib"
MASK_COMPRESSION_ARGS = {"level": 1}

# Offsets in a classic TIFF are 32-bit, so the whole file must stay under 4 GB.
CLASSIC_TIFF_LIMIT_BYTES = 4 * 1024**3

# Narrowest first. Label IDs are positive, so unsigned fits and stays smaller.
_MASK_DTYPES = (np.uint16, np.uint32, np.uint64)


def _mask_dtype(max_label: int) -> np.dtype:
    """Return the narrowest unsigned dtype that can hold ``max_label``."""
    for dtype in _MASK_DTYPES:
        if max_label <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    return np.dtype(np.uint64)


_BIGTIFF_RETRY_EXCEPTIONS = (ValueError, OverflowError, OSError)
_BIGTIFF_RETRY_HINTS = (
    "bigtiff",
    "classic tiff",
    "4 gb",
    "4gb",
    "offset",
    "ifd",
    "too large",
    "out of range",
    "cannot fit",
)


def _is_classic_tiff_size_error(exc: Exception) -> bool:
    """Return True if an exception indicates classic-TIFF size/offset overflow."""
    msg = str(exc).lower()
    return any(hint in msg for hint in _BIGTIFF_RETRY_HINTS)


def _rasterise_features_to_mask(
    features: list[dict],
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterise final cell polygons to an integer label mask.

    Uses post-constraint feature geometries so raster output matches the final
    exported non-overlapping polygons.

    Cells are painted largest-area first, so a smaller cell is always drawn
    *after* any larger cell it touches and therefore wins the contested pixels.
    Interior rings (holes) are excluded from a cell's own footprint rather than
    zeroed globally, so a cell never erases another cell that legitimately
    occupies its hole.  Without this, a small cell nested in a larger cell's
    concavity (e.g. a donut produced by overlap-constraint clipping) was
    engulfed by the larger cell and disappeared from the mask while remaining in
    the GeoJSON.
    """
    cell_geoms: list[tuple[float, int, Polygon]] = []
    max_id = 0
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
        max_id = max(max_id, cell_id)
        cell_geoms.append((geom.area, cell_id, geom))

    mask = np.zeros((height, width), dtype=_mask_dtype(max_id))

    # Largest cells first; smaller cells painted last win any contested pixels,
    # mirroring the "trim the larger by the smaller" rule of
    # ``constrain_cell_overlaps`` and preventing a large cell from engulfing a
    # smaller neighbour even if overlap clipping is disabled upstream.
    cell_geoms.sort(key=lambda item: item[0], reverse=True)

    for _area, cell_id, geom in cell_geoms:
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

            # Rasterise within the polygon's bounding box to bound memory and
            # let holes be carved from the cell's footprint locally.
            minx, miny, maxx, maxy = poly.bounds
            r0 = max(int(np.floor(miny)), 0)
            c0 = max(int(np.floor(minx)), 0)
            r1 = min(int(np.ceil(maxy)) + 1, height)
            c1 = min(int(np.ceil(maxx)) + 1, width)
            if r1 <= r0 or c1 <= c0:
                continue

            local = np.zeros((r1 - r0, c1 - c0), dtype=bool)
            rr, cc = draw_polygon(exterior[:, 1] - r0, exterior[:, 0] - c0, shape=local.shape)
            local[rr, cc] = True
            for hole in poly.interiors:
                interior = np.asarray(hole.coords)
                if interior.shape[0] < 3:
                    continue
                rr_h, cc_h = draw_polygon(interior[:, 1] - r0, interior[:, 0] - c0, shape=local.shape)
                local[rr_h, cc_h] = False

            rr_f, cc_f = np.nonzero(local)
            mask[rr_f + r0, cc_f + c0] = cell_id

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
    t_start = time.perf_counter()
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
    t_feature_build_start = time.perf_counter()
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
    logger.info(
        "GeoJSON feature build complete in %.2fs: cells=%d, features=%d",
        time.perf_counter() - t_feature_build_start,
        len(cells),
        len(features),
    )

    if constrain_overlaps:
        t_overlap_start = time.perf_counter()
        features = constrain_cell_overlaps(features)
        logger.info(
            "GeoJSON overlap constraint phase complete in %.2fs: features=%d",
            time.perf_counter() - t_overlap_start,
            len(features),
        )

    collection: dict = {
        "type": "FeatureCollection",
        "features": [annotation] + features,
    }

    final_output_path = output_path
    if gzip_output and not str(final_output_path).endswith(".gz"):
        final_output_path = Path(str(final_output_path) + ".gz")

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    t_geojson_write_start = time.perf_counter()
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
    logger.info(
        "GeoJSON serialization write complete in %.2fs: path=%s, gzip=%s",
        time.perf_counter() - t_geojson_write_start,
        final_output_path,
        gzip_output,
    )

    if output_mask is not None:
        t_mask_start = time.perf_counter()
        raster_mask = _rasterise_features_to_mask(features, H, W)
        t_rasterise = time.perf_counter() - t_mask_start
        output_mask.parent.mkdir(parents=True, exist_ok=True)

        # The write dominates whole-slide runtime, and raw byte count is the cost.
        t_write_start = time.perf_counter()

        # BigTIFF is required when the written file exceeds the classic TIFF
        # limit. Start with a conservative raw-size check, then retry in
        # BigTIFF mode only for explicit classic-TIFF size/offset failures.
        bigtiff = raster_mask.nbytes >= CLASSIC_TIFF_LIMIT_BYTES
        try:
            tifffile.imwrite(
                str(output_mask),
                raster_mask,
                compression=MASK_COMPRESSION,
                compressionargs=MASK_COMPRESSION_ARGS,
                bigtiff=bigtiff,
            )
        except _BIGTIFF_RETRY_EXCEPTIONS as exc:
            if bigtiff or not _is_classic_tiff_size_error(exc):
                raise
            logger.warning(
                "Raster mask write failed in classic TIFF mode (%s); retrying as BigTIFF.",
                exc,
            )
            tifffile.imwrite(
                str(output_mask),
                raster_mask,
                compression=MASK_COMPRESSION,
                compressionargs=MASK_COMPRESSION_ARGS,
                bigtiff=True,
            )
        t_write = time.perf_counter() - t_write_start

        written = output_mask.stat().st_size
        logger.info(
            "Raster mask export complete in %.2fs (rasterise=%.2fs, write=%.2fs): "
            "path=%s, shape=%s, dtype=%s, %.2f GB raw -> %.2f GB on disk",
            time.perf_counter() - t_mask_start,
            t_rasterise,
            t_write,
            output_mask,
            raster_mask.shape,
            raster_mask.dtype.name,
            raster_mask.nbytes / 1e9,
            written / 1e9,
        )

    logger.info("GeoJSON export total complete in %.2fs", time.perf_counter() - t_start)
    return len(features)
