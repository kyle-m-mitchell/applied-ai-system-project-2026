"""Deterministic intent parser: guarded text -> MusicIntent.

Rule-based and offline, so it is reproducible and needs no provider. It reuses
the genre/mood vocabulary already defined for the scorer, detects hard-filter
phrases, extracts directional numeric preferences (energy/valence/danceability/
acousticness/tempo) as controlled-cue :class:`FeatureGoal`s, and asks one
clarifying question when it recognizes nothing to search on. A Gemini
structured-intent parser can later implement the same ``parse`` shape.

Cues are *directions*, not fabricated targets ("calm" -> energy ``prefer_low``,
never ``energy == 0.2``), and each carries a controlled ``cue_id`` (``energy_low_v1``)
so a parsed preference is reproducible and auditable rather than free text.
"""

from __future__ import annotations

import re

from src.contracts import FeatureGoal, FeatureRelation, MusicIntent
from src.recommender import GENRE_TO_FAMILY, MOOD_TO_FAMILY


# Longest phrases first so multi-word genres/moods win over a shorter substring
# (e.g. "indie pop" before "pop").
_GENRES: tuple[str, ...] = tuple(sorted(GENRE_TO_FAMILY, key=len, reverse=True))
_MOODS: tuple[str, ...] = tuple(sorted(MOOD_TO_FAMILY, key=len, reverse=True))

_INSTRUMENTAL_CUES = ("instrumental", "no vocals", "no lyrics", "without vocals", "no singing")
_CLEAN_CUES = (
    "clean", "no explicit", "family friendly", "family-friendly", "no swearing",
    "kid friendly", "kid-friendly", "radio edit",
)

# Directional numeric cues: (cue_id, feature, relation, trigger phrases).
# Ordered by feature; each cue_id appears at most once in a parse.
_PREFERENCE_CUES: tuple[tuple[str, str, FeatureRelation, tuple[str, ...]], ...] = (
    ("energy_high_v1", "energy", FeatureRelation.PREFER_HIGH,
     ("high energy", "high-energy", "energetic", "upbeat", "intense", "hype",
      "more energetic", "more energy", "workout", "pumping", "banger",
      "fast paced", "fast-paced", "driving")),
    ("energy_low_v1", "energy", FeatureRelation.PREFER_LOW,
     ("low energy", "low-energy", "calm", "mellow", "sleepy", "gentle", "soft",
      "calmer", "less energy", "soothing", "downtempo", "laid back",
      "laid-back", "relaxing")),
    ("acoustic_high_v1", "acousticness", FeatureRelation.PREFER_HIGH,
     ("acoustic", "more acoustic", "unplugged")),
    ("acoustic_low_v1", "acousticness", FeatureRelation.PREFER_LOW,
     ("more electronic", "less acoustic")),
    ("valence_high_v1", "valence", FeatureRelation.PREFER_HIGH,
     ("happy", "happier", "brighter", "cheerful", "uplifting", "feel good",
      "feel-good", "joyful", "sunny")),
    ("valence_low_v1", "valence", FeatureRelation.PREFER_LOW,
     ("sad", "moodier", "darker", "melancholy", "melancholic", "somber",
      "gloomy", "downbeat", "wistful")),
    ("dance_high_v1", "danceability", FeatureRelation.PREFER_HIGH,
     ("danceable", "more danceable", "more movement", "to dance", "groovy",
      "party", "club")),
    ("dance_low_v1", "danceability", FeatureRelation.PREFER_LOW,
     ("less danceable", "less movement")),
)

_TEMPO_MIN_BPM, _TEMPO_MAX_BPM = 50.0, 200.0
_TEMPO_RANGE = re.compile(r"between\s+(\d{2,3})\s+and\s+(\d{2,3})\s*bpm")
_TEMPO_ATLEAST = re.compile(r"(?:at least|over|above|faster than|min)\s+(\d{2,3})\s*bpm")
_TEMPO_ATMOST = re.compile(r"(?:at most|under|below|slower than|max)\s+(\d{2,3})\s*bpm")
_TEMPO_NEAR = re.compile(r"(?:around|about|near|roughly|~)?\s*(\d{2,3})\s*bpm")

_TOKEN = re.compile(r"[a-z0-9&]+")
REDACTED_TOKEN = "[redacted]"  # kept out of the searchable-token count

# These words carry useful activity context even though they also imply a typed
# preference. Keep them in the neutral query when a preference is replaced; the
# rebuilt query appends the *current* canonical preference beside that context.
_CONTEXT_ANCHORS = {"workout", "party", "club", "to dance"}


def _has_phrase(haystack: str, phrase: str) -> bool:
    """Match a whole-word phrase, allowing '&' (for "r&b") at the boundaries."""
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", haystack) is not None


def _clamp_bpm(value: float) -> float:
    return min(_TEMPO_MAX_BPM, max(_TEMPO_MIN_BPM, value))


def _remove_phrase(text: str, phrase: str) -> str:
    return re.sub(
        rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
        " ",
        text,
        flags=re.IGNORECASE,
    )


def neutral_query_text(text: str, intent: MusicIntent) -> str:
    """Remove the controlled facets represented by ``intent`` from ``text``.

    Refinement must not keep stale words such as ``jazz`` or ``happy`` after the
    corresponding typed facet becomes ``rock`` or ``moody``. We remove only the
    facets this parsed intent actually owns, preserving unrelated activity and
    situational context for retrieval.
    """
    neutral = text
    phrases: list[str] = []
    if intent.genre:
        phrases.append(intent.genre)
    if intent.mood:
        phrases.append(intent.mood)
    if intent.instrumental_only:
        phrases.extend(_INSTRUMENTAL_CUES)
    if intent.exclude_explicit:
        phrases.extend(_CLEAN_CUES)

    goal_keys = {(goal.feature, goal.relation) for goal in intent.feature_goals}
    for _cue_id, feature, relation, triggers in _PREFERENCE_CUES:
        if (feature, relation) in goal_keys:
            phrases.extend(
                trigger for trigger in triggers if trigger not in _CONTEXT_ANCHORS
            )
    if any(goal.feature == "tempo_bpm" for goal in intent.feature_goals):
        for pattern in (_TEMPO_RANGE, _TEMPO_ATLEAST, _TEMPO_ATMOST, _TEMPO_NEAR):
            neutral = pattern.sub(" ", neutral.lower())

    for phrase in sorted(set(phrases), key=len, reverse=True):
        neutral = _remove_phrase(neutral, phrase)
    neutral = re.sub(r"\b(?:and|with)\b(?=\s*(?:[,;]|$))", " ", neutral, flags=re.I)
    neutral = re.sub(r"\s+", " ", neutral)
    return neutral.strip(" ,;:-")


def _canonical_goal_text(goal: FeatureGoal) -> str:
    if goal.relation is FeatureRelation.PREFER_HIGH:
        return {
            "energy": "high energy",
            "acousticness": "acoustic",
            "valence": "bright",
            "danceability": "danceable",
            "tempo_bpm": "fast tempo",
        }[goal.feature]
    if goal.relation is FeatureRelation.PREFER_LOW:
        return {
            "energy": "low energy",
            "acousticness": "less acoustic",
            "valence": "moody",
            "danceability": "less danceable",
            "tempo_bpm": "slow tempo",
        }[goal.feature]
    if goal.relation is FeatureRelation.RANGE:
        unit = " bpm" if goal.feature == "tempo_bpm" else ""
        return f"{goal.feature} between {goal.low:g} and {goal.high:g}{unit}"
    relation = {
        FeatureRelation.NEAR: "around",
        FeatureRelation.AT_LEAST: "at least",
        FeatureRelation.AT_MOST: "at most",
    }[goal.relation]
    unit = " bpm" if goal.feature == "tempo_bpm" else ""
    return f"{goal.feature} {relation} {goal.target:g}{unit}"


def rebuild_retrieval_query(intent: MusicIntent, *source_texts: str) -> str:
    """Build one current, contradiction-free query from neutral context + facets."""
    neutral_parts: list[str] = []
    for source in source_texts:
        part = neutral_query_text(source, intent)
        if part and part.casefold() not in {item.casefold() for item in neutral_parts}:
            neutral_parts.append(part)

    facets: list[str] = []
    if intent.genre:
        facets.append(intent.genre)
    if intent.mood:
        facets.append(intent.mood)
    if intent.instrumental_only:
        facets.append("instrumental")
    if intent.exclude_explicit:
        facets.append("clean")
    facets.extend(_canonical_goal_text(goal) for goal in intent.feature_goals)
    return " ".join(neutral_parts + facets).strip()


def _tempo_goal(lowered: str) -> FeatureGoal | None:
    """Parse a single tempo goal, most specific relation first."""
    match = _TEMPO_RANGE.search(lowered)
    if match:
        low, high = sorted((float(match.group(1)), float(match.group(2))))
        return FeatureGoal(
            feature="tempo_bpm", relation=FeatureRelation.RANGE,
            low=_clamp_bpm(low), high=_clamp_bpm(high), cue_id="tempo_range_v1",
        )
    for pattern, relation, cue_id in (
        (_TEMPO_ATLEAST, FeatureRelation.AT_LEAST, "tempo_atleast_v1"),
        (_TEMPO_ATMOST, FeatureRelation.AT_MOST, "tempo_atmost_v1"),
        (_TEMPO_NEAR, FeatureRelation.NEAR, "tempo_near_v1"),
    ):
        match = pattern.search(lowered)
        if match:
            return FeatureGoal(
                feature="tempo_bpm", relation=relation,
                target=_clamp_bpm(float(match.group(1))), cue_id=cue_id,
            )
    return None


def _feature_goals(lowered: str) -> tuple[FeatureGoal, ...]:
    """Extract all directional numeric goals present in the text (deduped by cue)."""
    goals: list[FeatureGoal] = []
    for cue_id, feature, relation, phrases in _PREFERENCE_CUES:
        matched = {phrase for phrase in phrases if _has_phrase(lowered, phrase)}

        # A bare positive word is a substring phrase inside these explicit
        # negatives ("less acoustic" contains "acoustic"). Do not manufacture a
        # conflict unless a separate positive phrase is also present.
        if cue_id == "acoustic_high_v1" and matched == {"acoustic"}:
            if _has_phrase(lowered, "less acoustic"):
                matched.clear()
        if cue_id == "dance_high_v1" and matched == {"danceable"}:
            if _has_phrase(lowered, "less danceable"):
                matched.clear()

        if matched:
            goals.append(FeatureGoal(feature=feature, relation=relation, cue_id=cue_id))
    tempo = _tempo_goal(lowered)
    if tempo is not None:
        goals.append(tempo)
    return tuple(goals)


class IntentParser:
    """Turn a sanitized query into a typed :class:`MusicIntent`."""

    def parse(self, sanitized_query: str, *, limit: int = 5) -> MusicIntent:
        text = sanitized_query.strip()
        lowered = text.lower()

        instrumental_only = any(_has_phrase(lowered, cue) for cue in _INSTRUMENTAL_CUES)
        exclude_explicit = any(_has_phrase(lowered, cue) for cue in _CLEAN_CUES)
        genre = next((g for g in _GENRES if _has_phrase(lowered, g)), None)
        mood = next((m for m in _MOODS if _has_phrase(lowered, m)), None)
        feature_goals = _feature_goals(lowered)

        goal_features = [goal.feature for goal in feature_goals]
        conflicts = sorted(
            {feature for feature in goal_features if goal_features.count(feature) > 1}
        )
        if conflicts:
            friendly = ", ".join(conflicts).replace("danceability", "movement")
            return MusicIntent(
                query=text,
                genre=genre,
                mood=mood,
                instrumental_only=instrumental_only,
                exclude_explicit=exclude_explicit,
                limit=limit,
                needs_clarification=True,
                clarification=(
                    f"I heard conflicting {friendly} directions. Which one should lead?"
                ),
            )

        recognized = bool(
            genre or mood or instrumental_only or exclude_explicit or feature_goals
        )
        searchable = _TOKEN.findall(lowered.replace(REDACTED_TOKEN, " "))

        # Clarify only when nothing was recognized and there is too little to
        # search on; otherwise let the retriever work on the free text.
        if not recognized and len(searchable) < 2:
            return MusicIntent(
                query=text,
                limit=limit,
                needs_clarification=True,
                clarification="What are you in the mood for — a genre, a vibe, or an activity?",
            )

        return MusicIntent(
            query=text,
            genre=genre,
            mood=mood,
            instrumental_only=instrumental_only,
            exclude_explicit=exclude_explicit,
            feature_goals=feature_goals,
            limit=limit,
        )
