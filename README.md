# cellmeasurement-py

`cellmeasurement-py` matches nuclear and whole-cell segmentation ROIs, computes
cell-level measurements, and exports QuPath-compatible GeoJSON (optionally
gzip-compressed) plus optional rasterised label masks.

This repository is the structured Python rewrite of the original Groovy
implementation in [WEHI-SODA-Hub/cellmeasurement](https://github.com/WEHI-SODA-Hub/cellmeasurement).

## What it does

The CLI supports three segmentation modes:

1. **Paired mode** (`--nuclear-mask` + `--whole-cell-mask`): overlap-first ROI
   matching, distance-threshold fallback, and watershed synthesis for unmatched
   nuclei.
2. **Nuclear-only mode** (`--nuclear-mask` only): exports nuclear objects as
   cells.
3. **Whole-cell-only mode** (`--whole-cell-mask` only): exports whole-cell
   objects directly.

Measurements (enabled by default) are computed from a source image TIFF and include:

- Shape metrics (area/length/diameter/solidity; µm-scaled where relevant)
- Compartment intensity stats (cell/nucleus/cytoplasm/membrane)
- Optional percentiles
- Equal-area erosion bins
- Equal-area 20 µm expansion bins
- Optional 20 µm environment measurements
- Optional nearest-neighbour aggregate means

## Input formats

Mask inputs can be either:

- **SOPA zarr** segmentation directories (with parquet boundaries), or
- **2-D TIFF** integer label masks (TIFF masks are converted to temporary zarr stores internally)

Intensity image input for measurements:

- **Multi-channel TIFF** (`--tiff-file`)

## Installation

### Requirements

- Python **3.13+**
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended)

### Install from source

```bash
git clone https://github.com/WEHI-SODA-Hub/cellmeasurement-py
cd cellmeasurement-py
uv sync
uv build
```

## Usage

Entrypoint:

```bash
cellmeasurement [OPTIONS]
```

If you are using `uv`, you can run `uv run cellmeasurement` to run the CLI.

### Minimal paired run (matching + measurements + GeoJSON)

```bash
uv run cellmeasurement \
  --nuclear-mask /abs/path/nuclear_mask.tiff \
  --whole-cell-mask /abs/path/whole_cell_mask.tiff \
  --tiff-file /abs/path/image.ome.tif \
  --output-file /abs/path/cellmeasurement.geojson
```

### Zarr input example

```bash
uv run cellmeasurement \
  --nuclear-mask /abs/path/nuc.zarr \
  --whole-cell-mask /abs/path/wc.zarr \
  --parquet-path shapes/cellpose_boundaries/shapes.parquet \
  --tiff-file /abs/path/image.ome.tif \
  --output-file /abs/path/out.geojson
```

### Export options example (gzip + rasterised mask)

```bash
uv run cellmeasurement \
  --nuclear-mask /abs/path/nuclear_mask.tiff \
  --whole-cell-mask /abs/path/whole_cell_mask.tiff \
  --tiff-file /abs/path/image.ome.tif \
  --output-file /abs/path/out.geojson \
  --gzip \
  --output-mask /abs/path/out_labels.tiff
```

## Parameters

| Option                                               | Description                                    | Default                    |
|------------------------------------------------------|------------------------------------------------|----------------------------|
| `--nuclear-mask PATH`                                | Nuclear segmentation mask (zarr or TIFF)       | `None`                     |
| `--whole-cell-mask PATH`                             | Whole-cell segmentation mask (zarr or TIFF)    | `None`                     |
| `--parquet-path TEXT`                                | Relative parquet path for zarr boundary shapes | `shapes/cellpose_boundaries/shapes.parquet` |
| `--temp-dir PATH`                                    | Parent directory for temporary TIFF→zarr       | `None`                     |
| `--keep-temp-zarr / --no-keep-temp-zarr`             | Keep/delete temporary TIFF-converted zarr      | `--no-keep-temp-zarr`      |
| `--tiff-file PATH`                                   | Multi-channel TIFF used for measurements       | `None`                     |
| `--output-file PATH`                                 | GeoJSON output path                            | `cellmeasurement.geojson`  |
| `--measurements / --no-measurements`                 | Enable/disable measurements                    | `--measurements`           |
| `--dist-threshold FLOAT`                             | Max centroid distance for secondary matching   | `10.0`                     |
| `--estimate-cell-boundary-dist FLOAT`                | Watershed expansion radius for unmatched nucs  | `3.0`                      |
| `--pixel-size-microns FLOAT`                         | Pixel size for µm scaling and 20 µm features   | `0.5`                      |
| `--downsample-factor FLOAT`                          | Optional image/mask downsampling factor        | `1.0`                      |
| `--tile-size INT`                                    | Tile size (px) for measurement image reads     | `2048`                     |
| `--tile-overlap INT`                                 | Tile overlap (px) for measurement reads        | `200`                      |
| `--threads INT`                                      | Number of tile workers for measurements        | `1`                        |
| `--erosion-steps / --no-erosion-steps`               | Equal-area erosion-bin measurements            | `--erosion-steps`          |
| `--expansion-steps / --no-expansion-steps`           | Equal-area 20 µm expansion-bin measurements    | `--expansion-steps`        |
| `--environment-expansion / --no-environment-expansion` | 20 µm environment compartment measurements   | `--no-environment-expansion` |
| `--neighbours INT` (`--neighbors`)                   | Number of nearest neighbours to aggregate      | `0`                        |
| `--percentiles "p1,p2,..."`                          | Extra intensity percentiles                    | `""`                       |
| `--gzip / --no-gzip`                                 | Write `.geojson.gz` output                     | `--no-gzip`                |
| `--output-mask PATH`                                 | Optional rasterised label-mask TIFF output     | `None`                     |
| `--simplify-rois / --no-simplify-rois`               | Douglas-Peucker polygon simplification         | `--simplify-rois`          |
| `--tolerance FLOAT`                                  | Polygon simplification tolerance (px)          | `0.5`                      |
| `--pretty-json / --no-pretty-json`                   | Write pretty-printed GeoJSON                   | `--no-pretty-json`         |
| `--constrain-overlaps / --no-constrain-overlaps`     | Clip overlapping output polygons               | `--constrain-overlaps`     |
| `--version`                                          | Print CLI version and exit                     | `False`                    |

## Performance notes

- Measurement is the most expensive stage.
- `--downsample-factor` can reduce memory and runtime substantially for large images.
- `--threads` controls tile-level measurement parallelism.
- Use `--no-measurements` for quick geometry-only QC/export runs.

## Development

Install dev dependencies and run tests:

```bash
uv sync --dev
uv run pytest -q
```

Type-checking/linting (if installed in your environment):

```bash
uv run pyright
uv run ruff check .
```

## Repository structure

- `src/cellmeasurement/cli.py` — CLI entrypoint and pipeline wiring
- `src/cellmeasurement/segmentation/` — ROI matching and synthesis logic
- `src/cellmeasurement/measurement/` — measurement pipeline and helpers
- `src/cellmeasurement/io/` — mask readers/writers and shared I/O utilities
- `cellmeasurement-groovy/` — original Groovy implementation/reference

## License

See [LICENSE](LICENSE).
