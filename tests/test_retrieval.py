"""Tests for the deterministic TF-IDF retriever and its provenance contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.contracts import CatalogTrack, SourceType
from src.recommender import load_songs
from src.retrieval import RETRIEVAL_FIELDS, TfidfRetriever, build_document_text
from src.service import RecommendationService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "songs.csv"


@pytest.fixture(scope="module")
def catalog() -> tuple[CatalogTrack, ...]:
    """Validate the real catalog once and expose the immutable tracks."""
    return RecommendationService(load_songs(str(CATALOG_PATH))).catalog


@pytest.fixture(scope="module")
def retriever(catalog) -> TfidfRetriever:
    """Build the TF-IDF index once for the module."""
    return TfidfRetriever(catalog)


def _track(**overrides) -> CatalogTrack:
    """Build a valid in-memory track, overriding only the fields under test."""
    base = {
        "id": 1,
        "title": "Base Track",
        "artist": "Base Artist",
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.4,
        "tempo_bpm": 80,
        "valence": 0.6,
        "danceability": 0.5,
        "acousticness": 0.9,
        "description": "A soft tape-warmed lofi loop for a quiet study session.",
        "tags": ("lofi", "tape warmth", "soft beats"),
        "contexts": ("studying", "reading"),
        "instruments": ("rhodes", "soft drums"),
        "instrumental": True,
        "explicit": False,
        "era": "2020s",
    }
    base.update(overrides)
    return CatalogTrack.model_validate(base)


def test_document_uses_the_declared_descriptor_fields():
    track = _track()
    document = build_document_text(track)

    assert "tape warmth" in document      # from tags
    assert "studying" in document          # from contexts
    assert "rhodes" in document            # from instruments
    assert "quiet study session" in document  # from description
    assert track.title not in document     # identity fields are excluded


def test_context_query_retrieves_the_relevant_genre(retriever):
    result = retriever.search("late-night study beats to focus", k=5)

    assert len(result.hits) == 5
    assert all(hit.track.genre == "lofi" for hit in result.hits[:3])
    assert all(hit.score > 0.0 for hit in result.hits)
    # Similarity must be ordered, most relevant first.
    scores = [hit.score for hit in result.hits]
    assert scores == sorted(scores, reverse=True)


def test_distinctive_terms_pull_the_matching_genre(retriever):
    assert retriever.search("gospel organ", k=1).hits[0].track.genre == "soul"
    assert retriever.search("dance floor club groove", k=1).hits[0].track.genre == "house"


def test_every_hit_carries_provenance(retriever):
    result = retriever.search("late-night study beats", k=3)

    assert result.index_fingerprint == retriever.index_fingerprint
    for hit in result.hits:
        assert hit.source_type is SourceType.CATALOG
        assert hit.source_id == f"catalog:{hit.track.id}"
        assert hit.content_hash
        assert hit.fields_used == RETRIEVAL_FIELDS
        assert 0.0 < hit.score <= 1.0
        assert hit.matched_terms  # something justified the match


def test_instrumental_only_filter_excludes_vocal_tracks(retriever):
    result = retriever.search("late-night study", k=10, instrumental_only=True)

    assert result.filters_applied == ("instrumental_only",)
    assert result.hits
    assert all(hit.track.instrumental for hit in result.hits)


def test_exclude_explicit_filter_removes_explicit_tracks(retriever):
    result = retriever.search("dance floor club groove", k=10, exclude_explicit=True)

    assert result.filters_applied == ("exclude_explicit",)
    assert all(not hit.track.explicit for hit in result.hits)


def test_retrieval_is_deterministic(retriever):
    first = retriever.search("rainy day melancholy piano", k=5)
    second = retriever.search("rainy day melancholy piano", k=5)

    assert [hit.track.id for hit in first.hits] == [hit.track.id for hit in second.hits]


def test_ties_break_on_ascending_track_id():
    shared_text = {
        "description": "Identical descriptive text used for a tie-break check here.",
        "tags": ("lofi", "calm", "warm"),
        "contexts": ("studying", "reading"),
        "instruments": ("piano", "pads"),
    }
    tracks = [
        _track(id=5, title="Track Five", artist="Artist Five", **shared_text),
        _track(id=3, title="Track Three", artist="Artist Three", **shared_text),
    ]
    result = TfidfRetriever(tracks).search("lofi calm", k=2)

    assert [hit.track.id for hit in result.hits] == [3, 5]
    assert result.hits[0].score == pytest.approx(result.hits[1].score)


@pytest.mark.parametrize("query", ["", "   ", "zzz qqq xyzzy"])
def test_no_signal_query_returns_no_hits(retriever, query):
    result = retriever.search(query, k=5)
    assert result.hits == ()


def test_k_limit_is_respected(retriever):
    assert len(retriever.search("beats", k=3).hits) <= 3
    with pytest.raises(ValueError, match="at least 1"):
        retriever.search("beats", k=0)


def test_fingerprint_tracks_catalog_content(catalog):
    original = TfidfRetriever(catalog)
    changed_first = catalog[0].model_copy(
        update={"description": catalog[0].description + " Now with an extra retrieval clue."}
    )
    mutated = TfidfRetriever((changed_first,) + catalog[1:])

    assert original.index_fingerprint != mutated.index_fingerprint


def test_empty_catalog_is_rejected():
    with pytest.raises(ValueError, match="at least one track"):
        TfidfRetriever(())
