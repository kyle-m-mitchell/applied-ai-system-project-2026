"""Deterministic MMR diversity re-ranking.

Maximal Marginal Relevance trades a little relevance for variety, so the top-k
isn't five near-identical tracks. Relevance leads; a cheap, deterministic
track-to-track similarity (built from the genre families the scorer already
defines) supplies the diversity penalty. No vectors or randomness needed.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.contracts import CatalogTrack, RetrievalHit
from src.recommender import GENRE_TO_FAMILY


def _similarity(a: CatalogTrack, b: CatalogTrack) -> float:
    """Cheap 0-1 track similarity: same genre+mood > same genre > same family."""
    if a.genre == b.genre and a.mood == b.mood:
        return 1.0
    if a.genre == b.genre:
        return 0.7
    if GENRE_TO_FAMILY.get(a.genre) == GENRE_TO_FAMILY.get(b.genre):
        return 0.4
    return 0.0


def mmr_rerank(
    hits: Sequence[RetrievalHit],
    k: int,
    *,
    lambda_: float = 0.7,
    relevance_floor: float = 0.5,
) -> tuple[RetrievalHit, ...]:
    """Re-rank ``hits`` by MMR, returning up to ``k`` diverse-but-relevant hits.

    Each step picks the hit maximizing ``lambda_ * relevance - (1 - lambda_) *
    max_similarity_to_already_selected``. The first pick is pure relevance, so
    the most relevant track always leads; ties break by lower track id.

    A ``relevance_floor`` (fraction of the top hit's score) gates eligibility, so
    a weak candidate can never displace a stronger one purely for variety — if the
    strong candidates are all similar, we return fewer rather than pad with weak,
    off-topic filler.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not hits:
        return ()

    floor = max(hit.score for hit in hits) * relevance_floor
    remaining = [hit for hit in hits if hit.score >= floor]
    selected: list[RetrievalHit] = []
    while remaining and len(selected) < k:
        best: RetrievalHit | None = None
        best_key: tuple[float, int] | None = None
        for hit in remaining:
            penalty = max(
                (_similarity(hit.track, chosen.track) for chosen in selected),
                default=0.0,
            )
            mmr = lambda_ * hit.score - (1.0 - lambda_) * penalty
            key = (mmr, -hit.track.id)  # higher MMR, then lower id
            if best_key is None or key > best_key:
                best_key = key
                best = hit
        assert best is not None
        selected.append(best)
        remaining.remove(best)
    return tuple(selected)
