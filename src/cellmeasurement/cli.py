import gc
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from shapely.geometry import Polygon

from .geometry import boundaries_to_geometries, extract_label_geometries
from .io.geojson_writer import write_geojson
from .io.mask_reader import SegmentationMask, load_mask, validate_grid_compatibility
from .measurement import measure_cells_tiled
from .segmentation.roi_matcher import match_rois

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = typer.Typer(help="CLI for running cellmeasurement")


def _extract_export_geometries(
    mask: SegmentationMask,
    simplify: bool,
    tolerance: float,
) -> dict[int, Polygon]:
    """Build export geometries from boundaries when available, otherwise from labels."""
    if mask.boundaries is not None:
        return boundaries_to_geometries(mask.boundaries, simplify=simplify, tolerance=tolerance)
    return extract_label_geometries(mask.labels, simplify=simplify, tolerance=tolerance)


def _cleanup_mask_temp_store(mask: SegmentationMask, keep_temp_zarr: bool) -> None:
    """Clean up a mask's temporary zarr store without masking primary failures."""
    if mask.temp_store_path is None:
        return
    if keep_temp_zarr:
        typer.echo(f"Kept temporary zarr store: {mask.temp_store_path}")
        return
    try:
        mask.cleanup_temp_store()
    except Exception:
        logging.warning("Failed to clean temporary zarr store: %s", mask.temp_store_path, exc_info=True)


def _cleanup_measurements_jsonl(path: Path | None) -> None:
    """Remove temporary measurement JSONL output without masking primary failures."""
    if path is None or not path.exists():
        return
    try:
        path.unlink()
    except Exception:
        logging.warning("Failed to clean temporary measurements JSONL: %s", path, exc_info=True)


def _parse_percentiles(percentiles: str) -> list[float]:
    if not percentiles.strip():
        return []
    values: list[float] = []
    for token in percentiles.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


@app.command(
    help="""Cellmeasurement matches nuclear and whole-cell segmentation masks, calculates
measurements and exports to GeoJSON.

At least one of --nuclear-mask or --whole-cell-mask must be supplied.

When both masks are provided, the tool runs in paired mode: nuclei are matched
to whole-cell labels by pixel overlap, and any unmatched nucleus receives a
synthesised cell boundary via watershed expansion.

When only one mask is provided the tool runs in single-mask mode
(nuclear-only or whole-cell-only) and skips the matching step.
"""
)
def main(
    nuclear_mask: Annotated[
        Optional[Path],
        typer.Option(help="Nuclear segmentation mask (sopa zarr directory or TIFF label image)."),
    ] = None,
    whole_cell_mask: Annotated[
        Optional[Path],
        typer.Option(help="Whole-cell segmentation mask (sopa zarr directory or TIFF label image)."),
    ] = None,
    parquet_path: Annotated[
        str,
        typer.Option(
            help=(
                "For zarr inputs, path to the parquet file containing segmentation "
                "boundaries relative to the zarr root."
            )
        ),
    ] = "shapes/cellpose_boundaries/shapes.parquet",
    temp_dir: Annotated[
        Optional[Path],
        typer.Option(
            help=(
                "Parent directory for temporary zarr stores created from TIFF masks. "
                "If omitted, a system temp directory is used."
            )
        ),
    ] = None,
    keep_temp_zarr: Annotated[
        bool,
        typer.Option(
            "--keep-temp-zarr/--no-keep-temp-zarr",
            help="Keep temporary TIFF-converted zarr stores instead of deleting them at exit.",
        ),
    ] = False,
    estimate_cell_boundary_dist: Annotated[
        float,
        typer.Option(
            help=(
                "Radius in pixels for watershed expansion of unmatched nuclei.  "
                "Only used in paired mode."
            )
        ),
    ] = 3.0,
    dist_threshold: Annotated[
        float,
        typer.Option(
            help=(
                "Maximum centroid distance (pixels) for matching nuclei to whole cells "
                "before synthesis."
            )
        ),
    ] = 10.0,
    measurements: Annotated[
        bool,
        typer.Option(
            "--measurements/--no-measurements",
            help="Enable intensity/erosion/shape measurements (default: enabled).",
        ),
    ] = True,
    tiff_file: Annotated[
        Optional[Path],
        typer.Option(
            help=(
                "Multi-channel TIFF image used for intensity measurements. "
                "Required when --measurements is enabled."
            )
        ),
    ] = None,
    percentiles: Annotated[
        str,
        typer.Option(
            help=(
                "Comma-separated percentiles for intensity measurement "
                '(e.g. "70,80,90,95").'
            )
        ),
    ] = "",
    erosion_steps: Annotated[
        bool,
        typer.Option(
            "--erosion-steps/--no-erosion-steps",
            help="Enable/disable equal-area erosion-bin measurements.",
        ),
    ] = True,
    expansion_steps: Annotated[
        bool,
        typer.Option(
            "--expansion-steps/--no-expansion-steps",
            help="Enable/disable equal-area 20um expansion-bin measurements.",
        ),
    ] = True,
    environment_expansion: Annotated[
        bool,
        typer.Option(
            "--environment-expansion/--no-environment-expansion",
            help="Enable/disable 20um pericellular environment measurements.",
        ),
    ] = False,
    neighbours: Annotated[
        int,
        typer.Option(
            "--neighbours",
            "--neighbors",
            help=(
                "Number of nearest neighbours for aggregated measurements "
                "(0 disables neighbour aggregation)."
            )
        ),
    ] = 0,
    pixel_size_microns: Annotated[
        float,
        typer.Option(
            help=(
                "Pixel size in microns, used for 20um expansion conversion "
                "and µm-scaled shape measurements."
            )
        ),
    ] = 0.5,
    downsample_factor: Annotated[
        float,
        typer.Option(
            help=(
                "Downsample factor for image and masks (e.g. 2.0, 4.0). "
                "Reduces memory usage and improves performance for large images. "
                "Values <= 1.0 or that round to step < 2 result in no downsampling. "
                "When downsampling is applied, effective pixel size increases proportionally."
            )
        ),
    ] = 1.0,
    tile_size: Annotated[
        int,
        typer.Option(help="Tile size in pixels for batched measurement image reads."),
    ] = 2048,
    tile_overlap: Annotated[
        int,
        typer.Option(help="Tile overlap in pixels for measurement reads."),
    ] = 200,
    threads: Annotated[
        int,
        typer.Option(help="Number of tile workers for measurements."),
    ] = 1,
    output_file: Annotated[
        Path, typer.Option(help="Output path for the GeoJSON file.")
    ] = Path("cellmeasurement.geojson"),
    simplify_rois: Annotated[
        bool,
        typer.Option(
            "--simplify-rois/--no-simplify-rois",
            help="Apply Douglas-Peucker simplification to polygon boundaries.",
        ),
    ] = True,
    tolerance: Annotated[
        float,
        typer.Option(help="Simplification tolerance in pixels (lower = more detail)."),
    ] = 0.5,
    pretty_json: Annotated[
        bool,
        typer.Option("--pretty-json/--no-pretty-json", help="Write indented JSON output."),
    ] = False,
    gzip_output: Annotated[
        bool,
        typer.Option(
            "--gzip/--no-gzip",
            help="Write gzip-compressed GeoJSON output (appends .gz when needed).",
        ),
    ] = False,
    output_mask: Annotated[
        Optional[Path],
        typer.Option(
            help=(
                "Optional TIFF path to export a rasterised label mask from final "
                "non-overlapping cell geometries."
            ),
        ),
    ] = None,
    constrain_overlaps: Annotated[
        bool,
        typer.Option(
            "--constrain-overlaps/--no-constrain-overlaps",
            help="Clip overlapping cell polygons before GeoJSON export.",
        ),
    ] = True,
) -> None:
    if nuclear_mask is None and whole_cell_mask is None:
        typer.echo(
            "Error: at least one of --nuclear-mask or --whole-cell-mask is required.", err=True
        )
        raise typer.Exit(code=1)
    if tile_size <= 0:
        typer.echo("Error: --tile-size must be > 0.", err=True)
        raise typer.Exit(code=1)
    if tile_overlap < 0:
        typer.echo("Error: --tile-overlap must be >= 0.", err=True)
        raise typer.Exit(code=1)
    if threads <= 0:
        typer.echo("Error: --threads must be > 0.", err=True)
        raise typer.Exit(code=1)
    if pixel_size_microns <= 0:
        typer.echo("Error: --pixel-size-microns must be > 0.", err=True)
        raise typer.Exit(code=1)
    if neighbours < 0:
        typer.echo("Error: --neighbours must be >= 0.", err=True)
        raise typer.Exit(code=1)
    if dist_threshold <= 0:
        typer.echo("Error: --dist-threshold must be > 0.", err=True)
        raise typer.Exit(code=1)
    if downsample_factor <= 0:
        typer.echo("Error: --downsample-factor must be > 0.", err=True)
        raise typer.Exit(code=1)
    if downsample_factor > 1.0:
        step = int(round(downsample_factor))
        if step < 2:
            typer.echo(
                f"Warning: --downsample-factor {downsample_factor} rounds to step={step}; "
                "no downsampling will be applied.",
                err=False,
            )

    nuc_mask: SegmentationMask | None = None
    wc_mask: SegmentationMask | None = None
    measurements_jsonl_path: Path | None = None

    try:
        if nuclear_mask is not None:
            nuc_mask = load_mask(nuclear_mask, parquet_path=parquet_path, temp_dir=temp_dir)
            typer.echo(f"Nuclear mask loaded: {nuc_mask.shape} px")

        if whole_cell_mask is not None:
            wc_mask = load_mask(whole_cell_mask, parquet_path=parquet_path, temp_dir=temp_dir)
            typer.echo(f"Whole-cell mask loaded: {wc_mask.shape} px")

        if nuc_mask is not None and wc_mask is not None:
            validate_grid_compatibility(nuc_mask, wc_mask)
            typer.echo("Grid compatibility validated.")

        nuc_arr = nuc_mask.labels if nuc_mask is not None else None
        wc_arr = wc_mask.labels if wc_mask is not None else None
        image_shape = (nuc_mask or wc_mask).shape

        typer.echo("Matching ROIs...")
        cells, synth_geoms = match_rois(
            nuc_arr,
            wc_arr,
            dist_threshold=dist_threshold,
            estimate_cell_boundary_dist=estimate_cell_boundary_dist,
            downsample_factor=downsample_factor,
        )
        typer.echo(f"Matched {len(cells)} cells.")

        measurements_by_cell: dict[int, dict[str, float]] | None = None
        if measurements:
            if tiff_file is None:
                logging.warning(
                    "Measurements requested but --tiff-file was not provided; proceeding without measurements."
                )
                typer.echo(
                    "Warning: measurements requested but --tiff-file is missing; exporting without measurements.",
                    err=True,
                )
            else:
                percentile_values = _parse_percentiles(percentiles)
                measurements_jsonl_path = output_file.with_name(
                    f"{output_file.stem}.measurements.jsonl.tmp"
                )
                typer.echo("Computing measurements...")
                measure_cells_tiled(
                    cells=cells,
                    nuc_labels=nuc_arr,
                    wc_labels=wc_arr,
                    synth_geoms=synth_geoms,
                    tiff_file=tiff_file,
                    image_shape=image_shape,
                    percentiles=percentile_values,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    threads=threads,
                    erosion_enabled=erosion_steps,
                    expansion_enabled=expansion_steps,
                    environment_expansion_enabled=environment_expansion,
                    neighbours=neighbours,
                    pixel_size_microns=pixel_size_microns,
                    downsample_factor=downsample_factor,
                    jsonl_path=measurements_jsonl_path,
                    return_results=False,
                )
                typer.echo("Computed measurements and streamed them to temporary JSONL.")

        # Free label arrays — no longer needed after matching.
        del nuc_arr, wc_arr
        gc.collect()

        typer.echo("Extracting label geometries for export...")
        nuc_geoms = (
            _extract_export_geometries(nuc_mask, simplify=simplify_rois, tolerance=tolerance)
            if nuc_mask is not None else None
        )
        wc_geoms = (
            _extract_export_geometries(wc_mask, simplify=simplify_rois, tolerance=tolerance)
            if wc_mask is not None else None
        )

        geojson_output_path = output_file
        if gzip_output and not str(geojson_output_path).endswith(".gz"):
            geojson_output_path = Path(str(geojson_output_path) + ".gz")

        typer.echo(f"Writing GeoJSON to {geojson_output_path}...")
        n_written = write_geojson(
            cells=cells,
            nuc_geoms=nuc_geoms,
            wc_geoms=wc_geoms,
            synth_geoms=synth_geoms,
            output_path=output_file,
            image_shape=image_shape,
            measurements_by_cell=measurements_by_cell,
            measurements_jsonl_path=measurements_jsonl_path,
            constrain_overlaps=constrain_overlaps,
            pretty=pretty_json,
            gzip_output=gzip_output,
            output_mask=output_mask,
        )
        typer.echo(f"Exported {n_written} cell features to {geojson_output_path}.")
        if output_mask is not None:
            typer.echo(f"Exported rasterisation mask to {output_mask}.")
    finally:
        for loaded_mask in (nuc_mask, wc_mask):
            if loaded_mask is None:
                continue
            _cleanup_mask_temp_store(loaded_mask, keep_temp_zarr)
        _cleanup_measurements_jsonl(measurements_jsonl_path)


if __name__ == "__main__":
    app()
