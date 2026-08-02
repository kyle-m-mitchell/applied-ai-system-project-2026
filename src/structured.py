"""The structured-preference scoring leg.

Turns a parsed :class:`~src.contracts.MusicIntent` into a per-track relevance in
``[0, 1]``, scoring the listener's *structured* wishes — genre, mood, and
directional numeric goals — directly against the catalog, rather than hoping the
text retriever infers them. It is deterministic, offline, and built entirely on
the shared :mod:`src.features` utilities, so "same family" and "closeness" mean
exactly what they mean everywhere else.

Two rules the reviewer insisted on live here:

* **Direction, not a fabricated target.** "High energy" is ``prefer_high`` — we
  reward higher values — never an invented ``energy == 0.85``.
* **Soft, not hard.** A goal only *reorders*; the hard constraints
  (instrumental-only, clean) are applied as filters elsewhere, before scoring.

The relevance is a strength-weighted average of the components that are actually
present, so it stays in ``[0, 1]`` and returns ``None`` when the intent carries no
structured signal at all (**not evaluated**, distinct from a real ``0.0``).
"""

from __future__ import annotations

from src.contracts import CatalogTrack, FeatureGoal, FeatureRelation, MusicIntent
from src.features import categorical_score, normalize_unit, numeric_closeness
from src.recommender import GENRE_TO_FAMILY, MOOD_TO_FAMILY

# Internal weights: how much each present component pulls the *structured* order.
# Genre dominates (as in the legacy scorer), mood next, each numeric goal a gentle
# nudge. These only order *within* the structured leg; the cross-leg blend is
# handled by rank fusion, so exact values here are non-critical.
W_GENRE = 4.0
W_MOOD = 1.5
W_GOAL = 0.75

# Tempo lives on a BPM scale; everything else is already 0-1. The range is stated
# explicitly (no silent domain assumptions) so conversion is honest.
TEMPO_MIN_BPM = 50.0
TEMPO_MAX_BPM = 200.0

# A goal "counts as evidence" in a reason line once it is at least this satisfied.
_REASON_THRESHOLD = 0.6


def _to_unit(feature: str, value: float) -> float:
    """Put a feature's raw value on the 0-1 scale the relations compare on."""
    if feature == "tempo_bpm":
        return normalize_unit(value, TEMPO_MIN_BPM, TEMPO_MAX_BPM)
    return value  # already 0-1 (validated on the catalog)


def goal_score(goal: FeatureGoal, track: CatalogTrack) -> float:
    """Score one directional goal against a track, in ``[0, 1]``.

    All comparisons happen on the 0-1 scale (tempo normalized first), so a BPM
    goal and an energy goal are measured the same honest way.
    """
    value = _to_unit(goal.feature, getattr(track, goal.feature))
    relation = goal.relation

    if relation is FeatureRelation.PREFER_HIGH:
        return value
    if relation is FeatureRelation.PREFER_LOW:
        return 1.0 - value

    if relation in (FeatureRelation.NEAR, FeatureRelation.AT_LEAST, FeatureRelation.AT_MOST):
        target = _to_unit(goal.feature, goal.target)  # target is non-None (validated)
        if relation is FeatureRelation.NEAR:
            return numeric_closeness(target, value)  # type: ignore[return-value]
        if relation is FeatureRelation.AT_LEAST:
            return 1.0 if value >= target else max(0.0, 1.0 - (target - value))
        return 1.0 if value <= target else max(0.0, 1.0 - (value - target))  # AT_MOST

    # RANGE: full credit inside [low, high], a linear ramp just outside.
    low = _to_unit(goal.feature, goal.low)   # low/high are non-None (validated)
    high = _to_unit(goal.feature, goal.high)
    if low <= value <= high:
        return 1.0
    distance = (low - value) if value < low else (value - high)
    return max(0.0, 1.0 - distance)


def structured_relevance(
    intent: MusicIntent, track: CatalogTrack
) -> tuple[float | None, tuple[str, ...]]:
    """Return this track's structured relevance in ``[0, 1]`` and its reasons.

    ``None`` means the intent carried no structured signal (genre, mood, or a
    numeric goal), so the structured leg did not run — the text leg stands alone.
    """
    total = 0.0
    achieved = 0.0
    reasons: list[str] = []

    if intent.genre:
        score = categorical_score(intent.genre, track.genre, GENRE_TO_FAMILY)
        total += W_GENRE
        achieved += W_GENRE * score
        if score >= 1.0:
            reasons.append(f"genre {track.genre}")
        elif score > 0.0:
            reasons.append(f"genre ~{track.genre}")

    if intent.mood:
        score = categorical_score(intent.mood, track.mood, MOOD_TO_FAMILY)
        total += W_MOOD
        achieved += W_MOOD * score
        if score >= 1.0:
            reasons.append(f"mood {track.mood}")
        elif score > 0.0:
            reasons.append(f"mood ~{track.mood}")

    for goal in intent.feature_goals:
        weight = W_GOAL * goal.strength
        score = goal_score(goal, track)
        total += weight
        achieved += weight * score
        if score >= _REASON_THRESHOLD:
            reasons.append(f"{goal.feature} {goal.relation.value}")

    if total == 0.0:
        return None, ()
    return achieved / total, tuple(reasons)
