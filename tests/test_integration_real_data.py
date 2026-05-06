from __future__ import annotations

import json
from pathlib import Path

import pytest

from cellmeasurement import cli


@pytest.mark.integration
def test_real_image_end_to_end_cli(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    image_path = data_dir / "test_data.tiff"
    wc_path = data_dir / "test_data_whole-cell.tiff"
    nuc_path = data_dir / "test_data_nuclear.tiff"

    required = [image_path, wc_path, nuc_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip("Integration inputs missing in data/: " + ", ".join(missing))

    output_path = tmp_path / "real-data-output.geojson"

    cli.main(
        nuclear_mask=nuc_path,
        whole_cell_mask=wc_path,
        tiff_file=image_path,
        measurements=True,
        percentiles="90",
        erosion_steps=False,
        expansion_steps=False,
        environment_expansion=False,
        neighbours=0,
        tile_size=512,
        tile_overlap=64,
        threads=2,
        output_file=output_path,
        constrain_overlaps=True,
    )

    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["type"] == "FeatureCollection"
    features = payload.get("features", [])
    assert len(features) > 0

    cell_features = [f for f in features if f.get("properties", {}).get("objectType") == "cell"]
    assert len(cell_features) > 0

    first_cell = cell_features[0]
    assert "measurements" in first_cell["properties"]
    assert "Cell: Area µm^2" in first_cell["properties"]["measurements"]
