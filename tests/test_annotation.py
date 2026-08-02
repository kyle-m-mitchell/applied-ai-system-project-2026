"""Synthetic, network-free tests for the local human mood-labeling backend."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.annotation import (
    MoodAnnotation,
    annotation_item_from_record,
    append_annotation,
    assess_annotation_readiness,
    load_annotations,
    select_annotation_sample,
)


def _records():
    genres = ("rock", "jazz", "electronic")
    return [
        {
            "catalog_id": "fma",
            "track_id": index,
            "title": f"Track {index}",
            "artist": f"Artist {index}",
            "genre_top": genres[(index - 1) % len(genres)],
            "source_url": f"https://example.test/tracks/{index}",
            # These sensitive fields must never survive projection.
            "predicted_energy": 0.99,
            "mood_profile": {"label": "upbeat"},
            "model_confidence": 0.99,
        }
        for index in range(1, 13)
    ]


def test_sample_is_source_order_independent_balanced_and_prediction_free():
    forward = select_annotation_sample(_records(), size=6)
    backward = select_annotation_sample(reversed(_records()), size=6)
    assert [item.key for item in forward] == [item.key for item in backward]
    assert {genre: sum(item.genre == genre for item in forward) for genre in {
        "rock", "jazz", "electronic"
    }} == {"rock": 2, "jazz": 2, "electronic": 2}
    assert all(
        set(item.to_dict())
        == {"catalog_id", "track_id", "title", "artist", "genre", "external_id", "source_url"}
        for item in forward
    )


def test_annotation_projection_rejects_duplicate_identity_and_unsafe_url():
    record = _records()[0] | {"source_url": "javascript:alert(1)"}
    assert annotation_item_from_record(record).source_url is None
    with pytest.raises(ValueError, match="duplicate annotation identity"):
        select_annotation_sample([record, record], size=2)


def _annotation(item, *, rater, role, quadrant="upbeat"):
    return MoodAnnotation.create(
        item=item,
        rater_id=rater,
        role=role,
        valence=0.8,
        arousal=0.8,
        quadrant=quadrant,
        confidence=4,
        now=datetime(2026, 8, 2, tzinfo=UTC),
    )


def test_jsonl_labels_validate_deduplicate_and_count_independent_audits(tmp_path):
    first, second = select_annotation_sample(_records(), size=2)
    labels_path = tmp_path / "labels.jsonl"
    primary = _annotation(first, rater="rater-a", role="primary")
    audit = _annotation(first, rater="rater-b", role="audit")
    second_primary = _annotation(second, rater="rater-a", role="primary", quadrant="calm")
    for annotation in (primary, audit, second_primary):
        append_annotation(labels_path, annotation)
    with pytest.raises(ValueError, match="already labeled"):
        append_annotation(labels_path, primary)

    loaded = load_annotations(labels_path)
    readiness = assess_annotation_readiness(
        loaded, target_primary_tracks=2, target_independent_audits=1
    )
    assert loaded == (primary, audit, second_primary)
    assert readiness.status == "experimental"  # thresholds cannot self-promote
    assert readiness.ready_for_future_review is True
    assert readiness.primary_tracks == 2
    assert readiness.independent_audit_pairs == 1
    assert readiness.quadrant_agreement == 1.0


def test_annotation_values_have_explicit_bounded_scales():
    item = select_annotation_sample(_records(), size=1)[0]
    with pytest.raises(ValueError, match="confidence"):
        MoodAnnotation.create(
            item=item,
            rater_id="rater-a",
            role="primary",
            valence=0.5,
            arousal=0.5,
            quadrant="upbeat",
            confidence=6,
        )
    with pytest.raises(ValueError, match="valence"):
        MoodAnnotation.create(
            item=item,
            rater_id="rater-a",
            role="primary",
            valence=-0.1,
            arousal=0.5,
            quadrant="upbeat",
            confidence=3,
        )
