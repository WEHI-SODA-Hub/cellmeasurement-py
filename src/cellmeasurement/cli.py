import typer
from typing import Annotated
from pathlib import Path

from .io.mask_reader import SegmentationMask, load_mask

app = typer.Typer(help="CLI for running cellmeasurement")


@app.command(
    help="""Cellmeasurement matches nuclear and whole-cell segmentation masks, calculates
measurements and exports to GeoJSON"""
)
def main(
    nuclear_mask: Annotated[
        Path, typer.Option(..., help="Nuclear segmentation mask file in TIFF format")
    ],
    whole_cell_mask: Annotated[
        Path, typer.Option(..., help="Whole-cell segmentation mask file in TIFF format")
    ],
    output_file: Annotated[
        Path, typer.Option(help="Output path for GeoJSON file")
    ] = Path("cellmeasurement.geojson"),
):

    sd_nuc_mask: SegmentationMask = load_mask(nuclear_mask)
    sd_wc_mask: SegmentationMask = load_mask(whole_cell_mask)

    print(sd_nuc_mask)
    print(sd_wc_mask)


if __name__ == "__main__":
    app()
