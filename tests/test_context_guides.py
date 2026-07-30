"""Tests for curated context guides and guide-driven query expansion."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.contracts import CatalogTrack, ContextGuide, SourceType
from src.recommender import load_songs
from src.retrieval import TfidfRetriever, load_context_guides
from src.service import RecommendationService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "songs.csv"
GUIDES_DIR = PROJECT_ROOT / "data" / "context_guides"


@pytest.fixture(scope="module")
def catalog() -> tuple[CatalogTrack, ...]:
    return RecommendationService(load_songs(str(CATALOG_PATH))).catalog


@pytest.fixture(scope="module")
def guides() -> list[ContextGuide]:
    return load_context_guides(str(GUIDES_DIR))


@pytest.fixture(scope="module")
def retriever(catalog, guides) -> TfidfRetriever:
    return TfidfRetriever(catalog, guides)


def test_guides_load_and_validate(guides):
    assert len(guides) >= 6
    ids = [guide.guide_id for guide in guides]
    assert ids == sorted(ids)  # deterministic order
    assert "studying-and-focus" in ids
    for guide in guides:
        assert guide.title
        assert len(guide.body.split()) >= 20


def test_guide_expansion_rescues_a_bridge_query(retriever):
    # "concentrate" appears in no track, only in the Studying guide.
    without = retriever.search("music to concentrate", k=5, use_guides=False)
    with_guides = retriever.search("music to concentrate", k=5, use_guides=True)

    assert without.hits == ()                       # before: nothing
    assert with_guides.hits                          # after: real results
    assert with_guides.hits[0].track.genre == "lofi"
    assert with_guides.expanded_query_terms          # query was expanded


def test_expansion_evidence_has_provenance(retriever):
    result = retriever.search("i need to relax and unwind", k=5)

    assert result.guides_used
    guide = result.guides_used[0]
    assert guide.source_type is SourceType.CONTEXT_GUIDE
    assert guide.source_id.startswith("context_guide:")
    assert guide.content_hash
    assert 0.0 < guide.score <= 1.0
    assert guide.expansion_terms
    # Every result-level expansion term is contributed by some fired guide.
    contributed = {term for g in result.guides_used for term in g.expansion_terms}
    assert set(result.expanded_query_terms) <= contributed


def test_dominance_threshold_keeps_expansion_focused(retriever):
    # A weak, off-topic guide must not ride along on a clear query.
    result = retriever.search("music to concentrate", k=5)
    assert len(result.guides_used) == 1
    assert result.guides_used[0].title == "Studying and Focus"


def test_disabling_guides_reproduces_track_only_behavior(retriever):
    result = retriever.search("late-night study beats to focus", k=5, use_guides=False)
    assert result.guides_used == ()
    assert result.expanded_query_terms == ()
    assert result.hits  # this query has catalog words, so it stands on its own


def test_no_matching_guide_is_a_no_op(retriever):
    # Out-of-vocabulary query matches no guide and no track.
    result = retriever.search("zzz qqq xyzzy", k=5)
    assert result.guides_used == ()
    assert result.expanded_query_terms == ()
    assert result.hits == ()


def test_retriever_without_guides_ignores_use_guides(catalog):
    plain = TfidfRetriever(catalog)  # no guides supplied
    result = plain.search("music to concentrate", k=5, use_guides=True)
    assert result.guides_used == ()
    assert result.hits == ()  # no bridge available, so no rescue


def test_guide_expansion_is_deterministic(retriever):
    first = retriever.search("something upbeat for a celebration", k=5)
    second = retriever.search("something upbeat for a celebration", k=5)
    assert [hit.track.id for hit in first.hits] == [hit.track.id for hit in second.hits]
    assert first.expanded_query_terms == second.expanded_query_terms


def test_context_guide_contract_rejects_thin_body():
    with pytest.raises(ValidationError):
        ContextGuide.model_validate({"guide_id": "x", "title": "X", "body": "too short"})


def test_loader_rejects_guide_without_heading(tmp_path):
    (tmp_path / "broken.md").write_text("no heading here\njust text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must start with a '# Title'"):
        load_context_guides(str(tmp_path))


def test_loader_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no context guides"):
        load_context_guides(str(tmp_path))


def test_fingerprint_includes_guide_content(catalog, guides):
    with_guides = TfidfRetriever(catalog, guides)
    without_guides = TfidfRetriever(catalog)
    # Guides are part of the index, so their presence changes the fingerprint.
    assert with_guides.index_fingerprint != without_guides.index_fingerprint

    edited = list(guides)
    edited[0] = edited[0].model_copy(update={"body": guides[0].body + " Extra curated detail."})
    mutated = TfidfRetriever(catalog, edited)
    assert mutated.index_fingerprint != with_guides.index_fingerprint


def test_fingerprint_includes_expansion_settings(catalog, guides, monkeypatch):
    baseline = TfidfRetriever(catalog, guides).index_fingerprint
    monkeypatch.setattr("src.retrieval.GUIDE_SCORE_RATIO", 0.9)
    retuned = TfidfRetriever(catalog, guides).index_fingerprint
    assert retuned != baseline
