"""Map heterogeneous retrieval hits into one unified ``RankedCandidate`` view.

The retrievers each speak their own dialect — TF-IDF sets a lexical score,
embeddings a semantic score, the hybrid both. This module translates any
:class:`~src.contracts.RetrievalHit` into a single
:class:`~src.contracts.RankedCandidate` so downstream code (the evaluator, the
UI, the event log) reads one shape. It never re-ranks and never changes a score;
it only re-describes what a retriever already decided.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.contracts import RankedCandidate, RetrievalHit, ScoreComponents

# Fusion identifiers name exactly how ``fused`` was produced, so a score stays
# reproducible and comparable. They mirror today's retriever behavior; the
# structured leg will add its own identifiers rather than overload these.
FUSION_HYBRID = "weighted-sum:sem=0.6,lex=0.4;v1"
FUSION_LEXICAL = "lexical-only;v1"
FUSION_SEMANTIC = "semantic-only;v1"
FUSION_UNSCORED = "unscored;v1"

MAX_REASON_TERMS = 5


def _fusion_version(hit: RetrievalHit) -> str:
    """Name the fusion that produced this hit's score, inferred from its signals."""
    has_semantic = hit.semantic_score is not None
    has_lexical = hit.lexical_score is not None
    if has_semantic and has_lexical:
        return FUSION_HYBRID
    if has_lexical:
        return FUSION_LEXICAL
    if has_semantic:
        return FUSION_SEMANTIC
    return FUSION_UNSCORED


def candidate_from_hit(hit: RetrievalHit) -> RankedCandidate:
    """Describe one retrieval hit as a unified ranked candidate (no re-ranking).

    ``categorical`` / ``numeric`` / ``personalization`` stay ``None`` here — *not
    evaluated* on the retrieval-only path, which is distinct from a real ``0.0``.
    The structured-preference leg is what fills them in.
    """
    available = tuple(
        name
        for name, value in (("semantic", hit.semantic_score), ("lexical", hit.lexical_score))
        if value is not None
    )
    components = ScoreComponents(
        semantic=hit.semantic_score,
        lexical=hit.lexical_score,
        fused=hit.score,
        available_signals=available,
        fusion_version=_fusion_version(hit),
        reasons=tuple(f"matched: {term}" for term in hit.matched_terms[:MAX_REASON_TERMS]),
    )
    return RankedCandidate(
        track=hit.track,
        components=components,
        source_type=hit.source_type,
        content_hash=hit.content_hash,
    )


def candidates_from_hits(hits: Sequence[RetrievalHit]) -> tuple[RankedCandidate, ...]:
    """Map an ordered sequence of hits into candidates, preserving order."""
    return tuple(candidate_from_hit(hit) for hit in hits)
