"""Tests for MMR diversity re-ranking."""

from __future__ import annotations

import pytest

from src.contracts import CatalogTrack, RetrievalHit, SourceType
from src.ranking import mmr_rerank


def _track(track_id: int, genre: str, mood: str) -> CatalogTrack:
    return CatalogTrack.model_validate(
        {
            "id": track_id,
            "title": f"Track {track_id}",
            "artist": f"Artist {track_id}",
            "genre": genre,
            "mood": mood,
            "energy": 0.5,
            "tempo_bpm": 100,
            "valence": 0.5,
            "danceability": 0.5,
            "acousticness": 0.5,
            "description": "A placeholder track used only for ranking-logic tests.",
            "tags": ("one", "two"),
            "contexts": ("alpha", "beta"),
            "instruments": ("piano",),
            "instrumental": True,
            "explicit": False,
            "era": "2020s",
        }
    )


def _hit(track_id: int, genre: str, mood: str, score: float) -> RetrievalHit:
    return RetrievalHit(
        source_type=SourceType.CATALOG,
        source_id=f"catalog:{track_id}",
        content_hash="h",
        fields_used=("genre",),
        score=score,
        matched_terms=("x",),
        track=_track(track_id, genre, mood),
    )


def test_mmr_diversifies_over_clustered_results():
    hits = [
        _hit(1, "lofi", "chill", 0.90),
        _hit(2, "lofi", "chill", 0.85),
        _hit(3, "lofi", "chill", 0.80),
        _hit(4, "ambient", "calm", 0.60),
        _hit(5, "metal", "intense", 0.50),
    ]
    out = mmr_rerank(hits, 3)
    ids = [hit.track.id for hit in out]

    assert out[0].track.id == 1  # relevance still leads
    assert len({hit.track.genre for hit in out}) == 3  # spread across genres
    assert 5 in ids and 2 not in ids  # a diverse low-score beats a duplicate high-score


def test_lambda_one_is_pure_relevance():
    hits = [
        _hit(1, "lofi", "chill", 0.9),
        _hit(2, "lofi", "chill", 0.85),
        _hit(3, "metal", "intense", 0.5),
    ]
    out = mmr_rerank(hits, 2, lambda_=1.0)
    assert [hit.track.id for hit in out] == [1, 2]


def test_deterministic_and_bounds():
    hits = [_hit(i, "lofi" if i % 2 else "jazz", "chill", 1.0 - i * 0.1) for i in range(1, 6)]
    assert mmr_rerank(hits, 4) == mmr_rerank(hits, 4)
    assert len(mmr_rerank(hits, 99)) == len(hits)  # k larger than available
    with pytest.raises(ValueError, match="at least 1"):
        mmr_rerank(hits, 0)
