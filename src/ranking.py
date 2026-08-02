"""Deterministic MMR diversity re-ranking.

Maximal Marginal Relevance trades a little relevance for variety, so the top-k
isn't five near-identical tracks. Relevance leads; a cheap, deterministic
track-to-track similarity (built from the genre families the scorer already
defines) supplies the diversity penalty. No vectors or randomness needed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from src.contracts import CatalogTrack, RetrievalHit
from src.features import same_family
from src.recommender import GENRE_TO_FAMILY, MOOD_TO_FAMILY

Similarity = Callable[[CatalogTrack, CatalogTrack], float]


def _similarity(a: CatalogTrack, b: CatalogTrack) -> float:
    """Cheap 0-1 track similarity: same genre+mood > same genre > same family.

    The default when the listener has *not* fixed a genre: diversity spreads
    results across genres. Genres outside the known families both look up to
    ``None``, and ``None == None`` is True — which once let MMR treat two
    *unrelated* unknown genres as siblings (0.4) and suppress one for the other.
    ``same_family`` carries the shared guard, so this and the legacy scorer can
    never disagree about what "same family" means.
    """
    if a.genre == b.genre:
        return 1.0 if a.mood == b.mood else 0.7
    if same_family(a.genre, b.genre, GENRE_TO_FAMILY):
        return 0.4
    return 0.0


def mood_similarity(a: CatalogTrack, b: CatalogTrack) -> float:
    """Diversity by *mood*, ignoring genre — for when a genre was explicitly asked.

    If a listener says "jazz", spreading the results across genres would violate
    the request. Diversity should instead vary the mood *within* the genre, so we
    measure similarity by mood alone and let same-genre tracks coexist freely.
    """
    if a.mood == b.mood:
        return 1.0
    if same_family(a.mood, b.mood, MOOD_TO_FAMILY):
        return 0.4
    return 0.0


def mmr_rerank(
    hits: Sequence[RetrievalHit],
    k: int,
    *,
    lambda_: float = 0.7,
    relevance_floor: float = 0.5,
    similarity: Similarity | None = None,
) -> tuple[RetrievalHit, ...]:
    """Re-rank ``hits`` by MMR, returning up to ``k`` diverse-but-relevant hits.

    Each step picks the hit maximizing ``lambda_ * relevance - (1 - lambda_) *
    max_similarity_to_already_selected``. The first pick is pure relevance, so
    the most relevant track always leads; ties break by lower track id.

    A ``relevance_floor`` (fraction of the top hit's score) gates eligibility, so
    a weak candidate can never displace a stronger one purely for variety — if the
    strong candidates are all similar, we return fewer rather than pad with weak,
    off-topic filler.

    ``similarity`` selects what "diverse" means. The default spreads across
    genres; pass :func:`mood_similarity` when the listener fixed a genre, so
    diversity varies the mood within it instead of overriding the request.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not hits:
        return ()

    sim = similarity if similarity is not None else _similarity
    floor = max(hit.score for hit in hits) * relevance_floor
    remaining = [hit for hit in hits if hit.score >= floor]
    selected: list[RetrievalHit] = []
    while remaining and len(selected) < k:
        best: RetrievalHit | None = None
        best_key: tuple[float, int] | None = None
        for hit in remaining:
            penalty = max(
                (sim(hit.track, chosen.track) for chosen in selected),
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
