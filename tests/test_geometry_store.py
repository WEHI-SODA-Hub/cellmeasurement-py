"""Tests for resumable, process-parallel label geometry extraction."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from cellmeasurement.geometry.geometry import (
    _build_batches,
    _label_bboxes,
    _labels_digest,
    extract_label_geometries,
    mask_to_geometry,
)
from cellmeasurement.geometry.geometry_store import (
    GeometryShardStore,
    arrays_to_polygons,
    polygons_to_arrays,
)


def _labels_grid(n_rows: int = 6, n_cols: int = 7, step: int = 12) -> np.ndarray:
    """Build a grid of well-separated square labels."""
    arr = np.zeros((n_rows * step, n_cols * step), dtype=np.uint32)
    label = 0
    for r in range(n_rows):
        for c in range(n_cols):
            label += 1
            arr[r * step + 2 : r * step + step - 2, c * step + 2 : c * step + step - 2] = label
    return arr


def _reference_geometries(labels: np.ndarray, tolerance: float = 0.5) -> dict[int, Polygon]:
    """Polygonize every label directly, independent of the batching machinery."""
    out: dict[int, Polygon] = {}
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        rows, cols = np.nonzero(labels == label_id)
        r0, c0 = int(rows.min()), int(cols.min())
        crop = labels[r0 : int(rows.max()) + 1, c0 : int(cols.max()) + 1]
        poly = mask_to_geometry(crop == label_id, True, tolerance, r0, c0)
        if poly is not None:
            out[int(label_id)] = poly
    return out


# -- shard encoding ------------------------------------------------------


def test_round_trip_preserves_geometry_exactly():
    poly = Polygon([(1.5, 2.0), (9.5, 2.0), (9.5, 8.5), (1.5, 8.5)])
    ids, ring_counts, poly_ring_counts, coords = polygons_to_arrays([(7, poly)])

    assert coords.dtype == np.float32
    restored = arrays_to_polygons(ids, ring_counts, poly_ring_counts, coords)

    assert list(restored) == [7]
    assert restored[7].equals(poly)


def test_round_trip_preserves_holes():
    shell = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(3, 3), (3, 6), (6, 6), (6, 3)]
    donut = Polygon(shell, [hole])

    restored = arrays_to_polygons(*polygons_to_arrays([(1, donut)]))

    assert len(restored[1].interiors) == 1
    assert restored[1].equals(donut)


def test_empty_batch_round_trips():
    """Batches where no label yields a contour still produce a readable shard."""
    ids, ring_counts, poly_ring_counts, coords = polygons_to_arrays([])

    assert ids.size == 0
    assert coords.shape == (0, 2)
    assert arrays_to_polygons(ids, ring_counts, poly_ring_counts, coords) == {}


def test_empty_shard_does_not_break_load(tmp_path):
    store = GeometryShardStore(tmp_path / "shards")
    store.write_shard(0, *polygons_to_arrays([]))
    store.write_shard(1, *polygons_to_arrays([(1, Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]))]))

    assert store.completed_batches() == {0, 1}
    assert set(store.load()) == {1}


# -- shard store ---------------------------------------------------------


def test_shard_store_round_trip(tmp_path):
    store = GeometryShardStore(tmp_path / "shards")
    poly_a = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    poly_b = Polygon([(10, 10), (14, 10), (14, 14), (10, 14)])

    store.write_shard(0, *polygons_to_arrays([(1, poly_a)]))
    store.write_shard(1, *polygons_to_arrays([(2, poly_b)]))

    assert store.completed_batches() == {0, 1}
    loaded = store.load()
    assert set(loaded) == {1, 2}
    assert loaded[1].equals(poly_a)


def test_stray_temp_shard_is_discarded_not_counted(tmp_path):
    """A kill mid-write leaves a temp file; it must not read as completed work."""
    store = GeometryShardStore(tmp_path / "shards")
    store.write_shard(0, *polygons_to_arrays([(1, Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]))]))
    torn = store.directory / "geoms-000001.npz.tmp"
    torn.write_bytes(b"\x93NUMPY truncated garbage")

    assert store.completed_batches() == {0}
    assert not torn.exists()
    assert set(store.load()) == {1}


# -- bounding boxes and batching ----------------------------------------


def test_label_bboxes_are_exclusive_stop_crops():
    labels = np.zeros((10, 10), np.uint32)
    labels[2:5, 3:8] = 4

    r0, c0, r1, c1 = _label_bboxes(labels)[4]

    assert (r0, c0, r1, c1) == (2, 3, 5, 8)
    assert np.array_equal(labels[r0:r1, c0:c1], np.full((3, 5), 4, np.uint32))


def test_batches_are_deterministic_across_runs():
    bboxes = {label: (0, 0, 1, 1) for label in range(1, 12)}

    first = _build_batches(bboxes, batch_size=4)
    second = _build_batches(dict(reversed(list(bboxes.items()))), batch_size=4)

    assert [index for index, _ in first] == [0, 1, 2]
    assert first == second, "resume relies on identical batch numbering between runs"


# -- extraction ----------------------------------------------------------


@pytest.mark.parametrize("workers", [1, 3])
def test_matches_direct_polygonization(workers):
    labels = _labels_grid()
    expected = _reference_geometries(labels)

    result = extract_label_geometries(labels, workers=workers)

    assert set(result) == set(expected)
    for label_id, poly in expected.items():
        assert result[label_id].equals(poly), f"label {label_id} differs"


def test_checkpointed_result_matches_uncheckpointed(tmp_path):
    labels = _labels_grid()

    plain = extract_label_geometries(labels, batch_size=5)
    checkpointed = extract_label_geometries(
        labels, batch_size=5, checkpoint_dir=tmp_path / "ckpt"
    )

    assert set(plain) == set(checkpointed)
    for label_id, poly in plain.items():
        assert checkpointed[label_id].equals(poly)


def test_resume_skips_completed_batches_and_returns_full_result(tmp_path, monkeypatch):
    """Simulate a killed job: interrupt mid-run, then resume and get everything."""
    labels = _labels_grid()
    expected = extract_label_geometries(labels, batch_size=5)
    checkpoint = tmp_path / "ckpt"

    from cellmeasurement.geometry import geometry as geometry_module

    real_batch = geometry_module._polygonize_batch
    calls: list[int] = []

    class Boom(RuntimeError):
        pass

    def dying_batch(payload):
        if len(calls) >= 3:
            raise Boom("job killed")
        calls.append(payload[0])
        return real_batch(payload)

    monkeypatch.setattr(geometry_module, "_polygonize_batch", dying_batch)
    with pytest.raises(Boom):
        extract_label_geometries(labels, batch_size=5, checkpoint_dir=checkpoint)

    survived = GeometryShardStore(checkpoint).completed_batches()
    assert len(survived) == 3, "completed batches must persist across the failure"

    monkeypatch.setattr(geometry_module, "_polygonize_batch", real_batch)
    resumed_calls: list[int] = []

    def counting_batch(payload):
        resumed_calls.append(payload[0])
        return real_batch(payload)

    monkeypatch.setattr(geometry_module, "_polygonize_batch", counting_batch)
    resumed = extract_label_geometries(labels, batch_size=5, checkpoint_dir=checkpoint)

    assert not (set(resumed_calls) & survived), "resume recomputed a checkpointed batch"
    assert set(resumed) == set(expected)
    for label_id, poly in expected.items():
        assert resumed[label_id].equals(poly)


def test_stale_checkpoint_from_different_settings_is_discarded(tmp_path, caplog):
    """A checkpoint outlives its run; different settings must not inherit its shards."""
    labels = _labels_grid(3, 3)
    checkpoint = tmp_path / "ckpt"
    extract_label_geometries(labels, tolerance=0.5, batch_size=3, checkpoint_dir=checkpoint)
    stale = GeometryShardStore(checkpoint).completed_batches()
    assert stale

    with caplog.at_level("WARNING"):
        result = extract_label_geometries(
            labels, tolerance=2.0, batch_size=3, checkpoint_dir=checkpoint
        )

    assert "different settings" in caplog.text
    expected = _reference_geometries(labels, tolerance=2.0)
    for label_id, poly in expected.items():
        assert result[label_id].equals(poly), "stale geometry survived a settings change"


def test_matching_settings_keep_checkpoint(tmp_path):
    labels = _labels_grid(3, 3)
    checkpoint = tmp_path / "ckpt"
    extract_label_geometries(labels, tolerance=0.5, batch_size=3, checkpoint_dir=checkpoint)

    store = GeometryShardStore(checkpoint)
    assert store.reconcile_fingerprint(
        {
            "simplify": True,
            "tolerance": 0.5,
            "batch_size": 3,
            "n_labels": 9,
            "image_shape": [36, 36],
            "labels_digest": _labels_digest(labels),
        }
    ) is True


def test_same_shape_settings_different_mask_discards_checkpoint(tmp_path, caplog):
    """Identical resolution/settings/cell count but a different mask must not reuse shards."""
    labels_a = _labels_grid(3, 3)
    # Flip preserves shape and the set of label ids (so image_shape and n_labels
    # match) while relocating every cell, so only the content digest differs.
    labels_b = np.ascontiguousarray(np.fliplr(labels_a))
    assert labels_a.shape == labels_b.shape
    assert set(np.unique(labels_a)) == set(np.unique(labels_b))
    assert _labels_digest(labels_a) != _labels_digest(labels_b)

    checkpoint = tmp_path / "ckpt"
    extract_label_geometries(labels_a, tolerance=0.5, batch_size=3, checkpoint_dir=checkpoint)

    with caplog.at_level("WARNING"):
        result = extract_label_geometries(
            labels_b, tolerance=0.5, batch_size=3, checkpoint_dir=checkpoint
        )

    assert "different settings" in caplog.text
    expected = _reference_geometries(labels_b, tolerance=0.5)
    for label_id, poly in expected.items():
        assert result[label_id].equals(poly), "stale geometry survived a mask change"


def test_resume_disabled_recomputes_everything(tmp_path):
    labels = _labels_grid(2, 2)
    checkpoint = tmp_path / "ckpt"
    extract_label_geometries(labels, batch_size=2, checkpoint_dir=checkpoint)

    result = extract_label_geometries(
        labels, batch_size=2, checkpoint_dir=checkpoint, resume=False
    )

    assert len(result) == 4


def test_empty_mask_returns_empty():
    assert extract_label_geometries(np.zeros((8, 8), np.uint32)) == {}
