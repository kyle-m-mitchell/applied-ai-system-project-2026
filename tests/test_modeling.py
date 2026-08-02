"""Synthetic tests for offline model splitting, gating, and export contracts."""

from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from src.modeling import (  # noqa: E402
    FeaturePrediction,
    artist_group_split_assignments,
    train_feature_models,
    write_model_report,
    write_prediction_jsonl,
)


def test_artist_split_is_deterministic_grouped_and_approximately_70_15_15():
    artists = [f"Artist {index}" for index in range(20) for _ in range(3)]
    first = artist_group_split_assignments(artists)
    second = artist_group_split_assignments(reversed(artists))

    by_artist = {}
    for artist, split in zip(artists, first, strict=True):
        by_artist.setdefault(artist, set()).add(split)
    assert all(len(splits) == 1 for splits in by_artist.values())
    assert {name: list(by_artist.values()).count({name}) for name in {
        "train", "calibration", "test"
    }} == {"train": 14, "calibration": 3, "test": 3}

    # Reordering rows changes output order, not the stable artist assignment.
    reverse_map = dict(zip(reversed(artists), second, strict=True))
    assert all(next(iter(splits)) == reverse_map[artist] for artist, splits in by_artist.items())


def _synthetic_frame():
    rng = np.random.default_rng(20260802)
    row_count = 360
    x1 = rng.uniform(0.0, 1.0, row_count)
    x2 = rng.uniform(0.0, 1.0, row_count)
    x3 = rng.normal(0.0, 1.0, row_count)
    # Deliberately nonlinear so the HGB-vs-linear comparison is real.
    target = np.clip(0.1 + 0.75 * ((x1 > 0.5) ^ (x2 > 0.5)) + rng.normal(0, 0.03, row_count), 0, 1)
    target[::5] = np.nan
    return pd.DataFrame(
        {
            "track_id": np.arange(1, row_count + 1),
            "artist": [f"artist-{index // 6}" for index in range(row_count)],
            "librosa__one": x1,
            "librosa__two": x2,
            "librosa__three": x3,
            "energy": target,
        }
    )


def test_training_uses_both_baselines_and_exports_only_missing_targets():
    frame = _synthetic_frame()
    run = train_feature_models(
        frame,
        feature_columns=("librosa__one", "librosa__two", "librosa__three"),
        targets=("energy",),
    )
    feature_report = run.report["features"]["energy"]
    metrics = feature_report["test_metrics"]
    assert set(metrics) >= {
        "mae", "r2", "dummy_mae", "ridge_mae", "interval_coverage"
    }
    assert set(feature_report["release_gates"]) == {
        "beats_dummy_by_5pct",
        "beats_ridge_by_5pct",
        "interval_coverage_75_to_90pct",
        "retains_at_least_30pct",
        "calibrated_width_available",
    }
    assert len(run.predictions) == int(frame["energy"].isna().sum())
    assert all(prediction.feature == "energy" for prediction in run.predictions)
    assert all(
        prediction.value is not None if prediction.released else prediction.value is None
        for prediction in run.predictions
    )


def test_prediction_and_report_writers_are_deterministic_strict_json(tmp_path):
    rows = (
        FeaturePrediction(2, "energy", None, None, None, None, "v1", False),
        FeaturePrediction(1, "energy", 0.7, 0.8, 0.6, 0.75, "v1", True),
    )
    output = tmp_path / "predictions.jsonl"
    write_prediction_jsonl(output, rows)
    payloads = [json.loads(line) for line in output.read_text().splitlines()]
    assert [payload["track_id"] for payload in payloads] == [1, 2]
    assert set(payloads[0]) == {
        "track_id", "feature", "value", "confidence", "interval_low",
        "interval_high", "model_version", "released",
    }

    report_path = tmp_path / "report.json"
    write_model_report(report_path, {"released": False, "metric": 0.25})
    assert json.loads(report_path.read_text()) == {"metric": 0.25, "released": False}
