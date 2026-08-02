"""Tests for rank/percentile fusion of the text and structured legs."""

from __future__ import annotations

import pytest

from src.contracts import CatalogTrack, MusicIntent, RetrievalHit, SourceType
from src.fusion import FUSION_VERSION, fuse_pool, fusion_version, percentile_ranks


def _track(track_id, genre="jazz") -> CatalogTrack:
    return CatalogTrack.model_validate(
        {
            "id": track_id, "title": f"T{track_id}", "artist": "A", "genre": genre,
            "mood": "chill", "energy": 0.5, "tempo_bpm": 100, "valence": 0.5,
            "danceability": 0.5, "acousticness": 0.5,
            "description": "A placeholder track used only for fusion-logic tests.",
            "tags": ("one", "two"), "contexts": ("alpha", "beta"), "instruments": ("piano",),
            "instrumental": True, "explicit": False, "era": "2020s",
        }
    )


def _hit(track_id, score, genre="jazz") -> RetrievalHit:
    return RetrievalHit(
        source_type=SourceType.CATALOG, source_id=f"catalog:{track_id}", content_hash="h",
        fields_used=("genre",), score=score, matched_terms=("x",), lexical_score=score,
        track=_track(track_id, genre),
    )


def test_percentile_ranks_bounds_and_ties():
    assert percentile_ranks({1: 0.1, 2: 0.5, 3: 0.9}) == {1: 0.0, 2: 0.5, 3: 1.0}
    assert percentile_ranks({1: 5.0, 2: 5.0}) == {1: 0.5, 2: 0.5}  # ties share mean rank
    assert percentile_ranks({7: 3.0}) == {7: 1.0}
    assert percentile_ranks({}) == {}


def test_fuse_pool_is_a_noop_without_structured_signal():
    hits = (_hit(1, 0.9), _hit(2, 0.5))
    out = fuse_pool(MusicIntent(), hits)  # no genre/mood/goals -> reproduction
    assert out == hits  # unchanged objects and order


def test_fuse_pool_promotes_the_structurally_preferred_track():
    # Text likes blues (id 1); the listener asked for jazz (id 2 sits lower in text).
    hits = (_hit(1, 0.90, genre="blues"), _hit(2, 0.40, genre="jazz"))
    out = fuse_pool(MusicIntent(genre="jazz"), hits)
    assert [h.track.id for h in out] == [2, 1]  # jazz lifted above blues
    jazz = next(h for h in out if h.track.id == 2)
    assert jazz.structured_score == 1.0            # exact genre match recorded
    assert 0.0 <= jazz.score <= 1.0                # fused score stays in range
    assert jazz.fusion_version == FUSION_VERSION


def test_custom_fusion_weights_get_a_distinct_audit_identifier():
    hits = (_hit(1, 0.90, genre="blues"), _hit(2, 0.40, genre="jazz"))
    out = fuse_pool(MusicIntent(genre="jazz"), hits, w_text=0.25, w_structured=0.75)
    assert all(hit.fusion_version == fusion_version(0.25, 0.75) for hit in out)
    assert out[0].fusion_version != FUSION_VERSION


def test_fuse_pool_records_structured_zero_as_evaluated():
    hits = (_hit(1, 0.9, genre="blues"),)
    out = fuse_pool(MusicIntent(genre="jazz"), hits)
    # blues was evaluated against a jazz request and missed: 0.0, not None.
    assert out[0].structured_score == 0.0
