"""Session-only taste learning: turn feedback taps into a bounded re-rank.

A listener's ``👍 more like this`` / ``👎 fewer like this`` / ``⚑ didn't fit`` taps
accumulate into an immutable :class:`~src.contracts.SessionPreference`. It only ever
*nudges* ranking (a small bounded weight), never overrides a named intent, and it is
passed to the companion per call — never stored on the shared engine — so two
sessions cannot influence each other. Every helper returns a *new* preference, so
the UI's snapshot/undo history reverses a tap exactly.

The signal is **hybrid** (as chosen): a generalizing *feature/genre affinity* plus a
*pull toward the specific liked exemplars*. Both are built only from a track's real
audio features and genre — never fabricated — reusing the same closeness the trusted
scorer uses elsewhere.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.contracts import CatalogTrack, FeatureVector, RetrievalHit, SessionPreference

# 0-1 audio features that can carry a directional taste bias. Tempo is excluded:
# "I liked this" is a poor way to infer a BPM target.
BIAS_FEATURES: tuple[str, ...] = (
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
)

LIKE_STEP = 0.34          # ~3 aligned taps saturate one axis
MAX_EXEMPLARS = 8         # keep the most recent liked fingerprints
EXEMPLAR_WEIGHT = 1.0     # weight of exemplar similarity vs each biased axis
SESSION_WEIGHT = 0.25     # bounded influence on the final blend (intent still leads)
SUPPRESS_FACTOR = 0.35    # a fit-missed track is demoted, never removed


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _direction(value: float) -> float:
    """Map a 0-1 feature value to a [-1, 1] "how high" signal (0.5 → neutral)."""
    return (value - 0.5) * 2.0


def feature_vector(track: CatalogTrack) -> FeatureVector:
    """Capture a liked track's real audio fingerprint (unknowns stay ``None``)."""
    return FeatureVector(
        energy=track.energy,
        valence=track.valence,
        danceability=track.danceability,
        acousticness=track.acousticness,
        instrumentalness=track.instrumentalness,
        genre=track.genre,
    )


def _nudged_bias(bias: dict[str, float], track: CatalogTrack, step: float) -> dict[str, float]:
    updated = dict(bias)
    for feature in BIAS_FEATURES:
        value = getattr(track, feature, None)
        if value is None:
            continue  # unknown feature contributes nothing — Unknown is not 0
        updated[feature] = _clamp(updated.get(feature, 0.0) + step * _direction(value))
        if updated[feature] == 0.0:
            updated.pop(feature, None)
    return updated


def apply_like(pref: SessionPreference, track: CatalogTrack) -> SessionPreference:
    """Learn toward this track: nudge affinity up and add it as an exemplar."""
    genre_bias = dict(pref.genre_bias)
    if track.genre:
        genre_bias[track.genre] = _clamp(genre_bias.get(track.genre, 0.0) + LIKE_STEP)
    exemplars = (feature_vector(track),) + tuple(
        vector for vector in pref.exemplars if vector != feature_vector(track)
    )
    return pref.model_copy(
        update={
            "feature_bias": _nudged_bias(pref.feature_bias, track, LIKE_STEP),
            "genre_bias": {g: b for g, b in genre_bias.items() if b != 0.0},
            "exemplars": exemplars[:MAX_EXEMPLARS],
            # A like clears any earlier soft suppression of the same track.
            "suppressed_ids": tuple(i for i in pref.suppressed_ids if i != track.id),
        }
    )


def apply_suggest_less(pref: SessionPreference, track: CatalogTrack) -> SessionPreference:
    """Learn away from this track's character (a reversible negative nudge)."""
    genre_bias = dict(pref.genre_bias)
    if track.genre:
        genre_bias[track.genre] = _clamp(genre_bias.get(track.genre, 0.0) - LIKE_STEP)
    return pref.model_copy(
        update={
            "feature_bias": _nudged_bias(pref.feature_bias, track, -LIKE_STEP),
            "genre_bias": {g: b for g, b in genre_bias.items() if b != 0.0},
        }
    )


def apply_fit_missed(pref: SessionPreference, track: CatalogTrack) -> SessionPreference:
    """Softly demote one track for this session (a "not right now", not a ban)."""
    return pref.model_copy(
        update={"suppressed_ids": tuple(dict.fromkeys((*pref.suppressed_ids, track.id)))}
    )


def clear_learning(pref: SessionPreference) -> SessionPreference:
    """Drop accumulated taste while keeping the enabled/disabled choice."""
    return SessionPreference(enabled=pref.enabled)


def _similarity(track: CatalogTrack, exemplar: FeatureVector) -> float:
    """Closeness of a track to a liked exemplar over shared features, in [0, 1]."""
    closeness = [
        1.0 - abs(getattr(track, feature) - getattr(exemplar, feature))
        for feature in BIAS_FEATURES
        if getattr(track, feature, None) is not None
        and getattr(exemplar, feature, None) is not None
    ]
    base = sum(closeness) / len(closeness) if closeness else 0.5
    if track.genre and exemplar.genre and track.genre == exemplar.genre:
        base = min(1.0, base + 0.1)  # small same-genre agreement bonus
    return base


def session_signal(track: CatalogTrack, pref: SessionPreference) -> float:
    """This track's session-taste match in [0, 1] (0.5 = neutral / no signal)."""
    if not pref.is_active:
        return 0.5
    numerator = 0.0
    denominator = 0.0
    for feature in BIAS_FEATURES:
        bias = pref.feature_bias.get(feature, 0.0)
        value = getattr(track, feature, None)
        if value is None or bias == 0.0:
            continue
        aligned = value if bias > 0.0 else (1.0 - value)  # reward preferred direction
        numerator += abs(bias) * aligned
        denominator += abs(bias)
    genre_bias = pref.genre_bias.get(track.genre, 0.0) if track.genre else 0.0
    if genre_bias != 0.0:
        numerator += abs(genre_bias) * (1.0 if genre_bias > 0.0 else 0.0)
        denominator += abs(genre_bias)
    if pref.exemplars:
        numerator += EXEMPLAR_WEIGHT * max(_similarity(track, ex) for ex in pref.exemplars)
        denominator += EXEMPLAR_WEIGHT
    return numerator / denominator if denominator > 0.0 else 0.5


def session_ranked_hits(
    hits: Sequence[RetrievalHit],
    pref: SessionPreference | None,
    *,
    weight: float = SESSION_WEIGHT,
) -> tuple[RetrievalHit, ...]:
    """Re-rank a candidate pool by a bounded blend of relevance and session taste.

    The blended value replaces ``score`` so downstream MMR and display honor it.
    When there is no active preference the pool is returned unchanged, so a session
    with no feedback ranks exactly as it does today.
    """
    if pref is None or not pref.is_active or not hits:
        return tuple(hits)
    suppressed = set(pref.suppressed_ids)

    def blended(hit: RetrievalHit) -> float:
        base = (1.0 - weight) * hit.score + weight * session_signal(hit.track, pref)
        if hit.track.id in suppressed:
            base *= SUPPRESS_FACTOR
        return max(0.0, min(1.0, base))

    scored = [(hit.model_copy(update={"score": blended(hit)}), hit.track.id) for hit in hits]
    scored.sort(key=lambda pair: (-pair[0].score, pair[1]))
    return tuple(hit for hit, _ in scored)
