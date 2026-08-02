"""Tests for the unified RankedCandidate mapping (no re-ranking, None != 0)."""

from __future__ import annotations

from src.contracts import CatalogTrack, RetrievalHit, SourceType
from src.scoring import (
    FUSION_HYBRID,
    FUSION_LEXICAL,
    FUSION_SEMANTIC,
    FUSION_STRUCTURED,
    candidate_from_hit,
    candidates_from_hits,
)


def _track(track_id: int) -> CatalogTrack:
    return CatalogTrack.model_validate(
        {
            "id": track_id, "title": f"Track {track_id}", "artist": "A", "genre": "lofi",
            "mood": "chill", "energy": 0.5, "tempo_bpm": 100, "valence": 0.5,
            "danceability": 0.5, "acousticness": 0.5,
            "description": "A placeholder track used only for scoring-logic tests.",
            "tags": ("one", "two"), "contexts": ("alpha", "beta"), "instruments": ("piano",),
            "instrumental": True, "explicit": False, "era": "2020s",
        }
    )


def _hit(track_id, score, *, semantic=None, lexical=None, matched=()):
    return RetrievalHit(
        source_type=SourceType.CATALOG, source_id=f"catalog:{track_id}", content_hash="h",
        fields_used=("genre",), score=score, matched_terms=matched,
        semantic_score=semantic, lexical_score=lexical, track=_track(track_id),
    )


def test_hybrid_hit_names_both_signals():
    c = candidate_from_hit(_hit(1, 0.52, semantic=0.6, lexical=0.4, matched=("study", "calm")))
    assert c.components.semantic == 0.6 and c.components.lexical == 0.4
    assert c.components.fused == 0.52  # fused is unchanged — no re-ranking
    assert c.components.available_signals == ("semantic", "lexical")
    assert c.components.fusion_version == FUSION_HYBRID
    # categorical/numeric/personalization are "not evaluated" on the retrieval path
    assert c.components.categorical is None and c.components.numeric is None
    assert c.components.reasons == ("matched: study", "matched: calm")


def test_lexical_only_and_semantic_only_fusion_labels():
    lex = candidate_from_hit(_hit(1, 0.8, lexical=0.8))
    assert lex.components.semantic is None and lex.components.available_signals == ("lexical",)
    assert lex.components.fusion_version == FUSION_LEXICAL
    sem = candidate_from_hit(_hit(2, 0.7, semantic=0.7))
    assert sem.components.lexical is None and sem.components.available_signals == ("semantic",)
    assert sem.components.fusion_version == FUSION_SEMANTIC


def test_none_is_not_zero():
    # A signal that ran and found nothing is 0.0 and IS available; a signal that
    # never ran is None and is NOT available. The distinction must survive mapping.
    c = candidate_from_hit(_hit(1, 0.3, semantic=0.5, lexical=0.0))
    assert c.components.lexical == 0.0
    assert "lexical" in c.components.available_signals
    c2 = candidate_from_hit(_hit(2, 0.5, semantic=0.5))  # lexical never evaluated
    assert c2.components.lexical is None
    assert "lexical" not in c2.components.available_signals


def test_explicit_fusion_version_survives_candidate_mapping():
    hit = _hit(1, 0.7, semantic=0.8, lexical=0.4).model_copy(
        update={
            "structured_score": 0.9,
            "fusion_version": "percentile:text=0.25,structured=0.75;v1",
        }
    )
    candidate = candidate_from_hit(hit)
    assert candidate.components.fusion_version == hit.fusion_version
    assert candidate.components.fusion_version != FUSION_STRUCTURED


def test_candidates_from_hits_preserves_order():
    hits = [_hit(3, 0.9, lexical=0.9), _hit(1, 0.5, lexical=0.5), _hit(2, 0.7, lexical=0.7)]
    assert [c.track.id for c in candidates_from_hits(hits)] == [3, 1, 2]
