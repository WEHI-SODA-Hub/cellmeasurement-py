import gc
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from .io.geometry import boundaries_to_geometries
from .io.geojson_writer import write_geojson
from .io.mask_reader import SegmentationMask, load_mask, validate_grid_compatibility
from .segmentation.roi_matcher import match_rois

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = typer.Typer(help="CLI for running cellmeasurement")


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
        typer.Option(help="Nuclear segmentation mask (sopa zarr store directory)."),
    ] = None,
    whole_cell_mask: Annotated[
        Optional[Path],
        typer.Option(help="Whole-cell segmentation mask (sopa zarr store directory)."),
    ] = None,
    segmentation_method: Annotated[
        str,
        typer.Option(
            help=(
                "Segmentation method used to produce the boundaries "
                "(e.g. 'cellpose', 'mesmer', 'cellsam').  "
                "Determines the parquet path: "
                "shapes/<method>_boundaries/shapes.parquet."
            )
        ),
    ] = "cellpose",
    synthesis_dist: Annotated[
        float,
        typer.Option(
            help=(
                "Radius in pixels for watershed expansion of unmatched nuclei.  "
                "Only used in paired mode."
            )
        ),
    ] = 3.0,
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
) -> None:
    if nuclear_mask is None and whole_cell_mask is None:
        typer.echo(
            "Error: at least one of --nuclear-mask or --whole-cell-mask is required.", err=True
        )
        raise typer.Exit(code=1)

    nuc_mask: SegmentationMask | None = None
    wc_mask: SegmentationMask | None = None

    if nuclear_mask is not None:
        nuc_mask = load_mask(nuclear_mask, segmentation_method=segmentation_method)
        typer.echo(f"Nuclear mask loaded: {nuc_mask.shape} px")

    if whole_cell_mask is not None:
        wc_mask = load_mask(whole_cell_mask, segmentation_method=segmentation_method)
        typer.echo(f"Whole-cell mask loaded: {wc_mask.shape} px")

    if nuc_mask is not None and wc_mask is not None:
        validate_grid_compatibility(nuc_mask, wc_mask)
        typer.echo("Grid compatibility validated.")

    nuc_arr = nuc_mask.labels if nuc_mask is not None else None
    wc_arr = wc_mask.labels if wc_mask is not None else None
    image_shape = (nuc_mask or wc_mask).shape  # type: ignore[union-attr]

    typer.echo("Matching ROIs...")
    cells, synth_geoms = match_rois(nuc_arr, wc_arr, synthesis_dist=synthesis_dist)
    typer.echo(f"Matched {len(cells)} cells.")

    # Free label arrays — no longer needed after matching.
    del nuc_arr, wc_arr
    gc.collect()

    typer.echo("Extracting label geometries for export...")
    nuc_geoms = (
        boundaries_to_geometries(nuc_mask.boundaries, simplify=simplify_rois, tolerance=tolerance)
        if nuc_mask is not None else None
    )
    wc_geoms = (
        boundaries_to_geometries(wc_mask.boundaries, simplify=simplify_rois, tolerance=tolerance)
        if wc_mask is not None else None
    )

    typer.echo(f"Writing GeoJSON to {output_file}...")
    n_written = write_geojson(
        cells=cells,
        nuc_geoms=nuc_geoms,
        wc_geoms=wc_geoms,
        synth_geoms=synth_geoms,
        output_path=output_file,
        image_shape=image_shape,
        pretty=pretty_json,
    )
    typer.echo(f"Exported {n_written} cell features to {output_file}.")


if __name__ == "__main__":
    app()
