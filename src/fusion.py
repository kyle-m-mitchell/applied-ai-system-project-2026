"""Rank/percentile fusion of the text and structured legs.

The text leg (semantic + lexical) and the structured leg (genre/mood/numeric)
score on **different scales** — a cosine similarity is not a strength-weighted
relevance — so blending their raw numbers with a weighted sum would let one leg's
units silently dominate. Instead we convert each leg's scores to **percentile
ranks** (unit-free, order-preserving) and blend those. The weights then mean what
they look like they mean: "how much should each leg's *ordering* count."

When the intent carries no structured signal, fusion is a no-op — the text
ordering (today's 0.6/0.4 hybrid) is returned unchanged. That is the reproduction
guarantee: a pure free-text query ranks exactly as it did before this leg existed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.contracts import MusicIntent, RetrievalHit
from src.structured import structured_relevance

# Cross-leg blend, calibrated against the report card (a weight sweep from 0.4 to
# 0.8 structured): 0.6 structured maximized genre satisfaction (0.863) with no
# regression across required/holdout/planned/development; beyond that it dipped.
# When a listener names a structured wish, honoring it should lead — text still
# anchors relevance and carries queries with no structured signal at all. Tunable
# so the harness can re-calibrate as cases and data grow.
W_TEXT = 0.4
W_STRUCTURED = 0.6


def fusion_version(w_text: float, w_structured: float) -> str:
    """Return the exact, audit-safe identifier for one configured blend."""
    return f"percentile:text={w_text:g},structured={w_structured:g};v1"


FUSION_VERSION = fusion_version(W_TEXT, W_STRUCTURED)


def percentile_ranks(scores: Mapping[int, float]) -> dict[int, float]:
    """Map raw scores to percentile ranks in ``[0, 1]`` (ties share the mean rank).

    Only order matters, so scores on different scales become comparable without a
    raw weighted sum. The lowest score maps to 0.0, the highest to 1.0.
    """
    n = len(scores)
    if n == 0:
        return {}
    if n == 1:
        return {key: 1.0 for key in scores}
    ordered = sorted(scores.items(), key=lambda item: item[1])
    result: dict[int, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
            j += 1  # group equal scores
        percentile = ((i + j) / 2) / (n - 1)
        for position in range(i, j + 1):
            result[ordered[position][0]] = percentile
        i = j + 1
    return result


def fuse_pool(
    intent: MusicIntent,
    hits: Sequence[RetrievalHit],
    *,
    w_text: float = W_TEXT,
    w_structured: float = W_STRUCTURED,
) -> tuple[RetrievalHit, ...]:
    """Re-rank a text pool by fusing text and structured percentile ranks.

    Each returned hit carries its ``structured_score`` and the fused value in
    ``score`` (both in ``[0, 1]``). If the structured leg did not run for any hit
    — no genre, mood, or numeric goal — the hits are returned unchanged.
    """
    if not hits:
        return tuple(hits)

    structured = {hit.track.id: structured_relevance(intent, hit.track) for hit in hits}
    if all(relevance is None for relevance, _ in structured.values()):
        return tuple(hits)  # no structured signal: preserve the text ordering exactly

    text_scores = {hit.track.id: hit.score for hit in hits}
    struct_scores = {
        track_id: (relevance if relevance is not None else 0.0)
        for track_id, (relevance, _reasons) in structured.items()
    }
    text_pct = percentile_ranks(text_scores)
    struct_pct = percentile_ranks(struct_scores)

    denom = w_text + w_structured
    version = fusion_version(w_text, w_structured)
    fused = {
        track_id: (w_text * text_pct[track_id] + w_structured * struct_pct[track_id]) / denom
        for track_id in text_scores
    }
    reordered = sorted(hits, key=lambda hit: (-fused[hit.track.id], hit.track.id))
    return tuple(
        hit.model_copy(
            update={
                "score": fused[hit.track.id],
                "structured_score": structured[hit.track.id][0],
                "structured_reasons": structured[hit.track.id][1],
                "fusion_version": version,
            }
        )
        for hit in reordered
    )
