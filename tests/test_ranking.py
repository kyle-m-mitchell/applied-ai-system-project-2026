"""Tests for MMR diversity re-ranking."""

from __future__ import annotations

import pytest

from src.contracts import CatalogTrack, RetrievalHit, SourceType
from src.ranking import _similarity, mmr_rerank


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


def test_unknown_genres_are_not_treated_as_the_same_family():
    # Two genres outside GENRE_TO_FAMILY both map to family None. A bare
    # `None == None` would wrongly read them as siblings (0.4) and let MMR
    # suppress one for the other. Sharing no family, similarity must be 0.0.
    assert _similarity(_track(1, "polka", "chill"), _track(2, "klezmer", "somber")) == 0.0
    # sanity: a genuinely shared family (both "mellow") still reads as siblings
    assert _similarity(_track(3, "lofi", "chill"), _track(4, "ambient", "calm")) == 0.4
    # sanity: same genre is unchanged (exact 1.0 / same-genre-different-mood 0.7)
    assert _similarity(_track(5, "polka", "chill"), _track(6, "polka", "chill")) == 1.0
    assert _similarity(_track(7, "polka", "chill"), _track(8, "polka", "somber")) == 0.7


def test_diversity_does_not_penalize_distinct_unknown_genres():
    # Anchor is an unknown genre (polka). For the 2nd slot MMR chooses between
    # another unknown genre (klezmer, higher relevance) and a known one (lofi).
    # Before the fix, klezmer was wrongly penalized as polka's "sibling" (0.4)
    # and lost to lofi -> [1, 3]. With no false penalty, klezmer wins -> [1, 2].
    hits = [
        _hit(1, "polka", "chill", 0.90),
        _hit(2, "klezmer", "somber", 0.60),
        _hit(3, "lofi", "calm", 0.55),
    ]
    ids = [hit.track.id for hit in mmr_rerank(hits, 2)]
    assert ids == [1, 2]


def test_deterministic_and_bounds():
    hits = [_hit(i, "lofi" if i % 2 else "jazz", "chill", 1.0 - i * 0.1) for i in range(1, 6)]
    assert mmr_rerank(hits, 4) == mmr_rerank(hits, 4)
    assert len(mmr_rerank(hits, 99)) == len(hits)  # k larger than available
    with pytest.raises(ValueError, match="at least 1"):
        mmr_rerank(hits, 0)
