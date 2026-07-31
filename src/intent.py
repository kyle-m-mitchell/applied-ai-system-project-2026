"""Deterministic intent parser: guarded text -> MusicIntent.

Rule-based and offline, so it is reproducible and needs no provider. It reuses
the genre/mood vocabulary already defined for the scorer, detects hard-filter
phrases, and asks one clarifying question when it recognizes nothing to search on.
A Gemini structured-intent parser can later implement the same ``parse`` shape.
"""

from __future__ import annotations

import re

from src.contracts import MusicIntent
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

_TOKEN = re.compile(r"[a-z0-9&]+")
REDACTED_TOKEN = "[redacted]"  # kept out of the searchable-token count


def _has_phrase(haystack: str, phrase: str) -> bool:
    """Match a whole-word phrase, allowing '&' (for "r&b") at the boundaries."""
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", haystack) is not None


class IntentParser:
    """Turn a sanitized query into a typed :class:`MusicIntent`."""

    def parse(self, sanitized_query: str, *, limit: int = 5) -> MusicIntent:
        text = sanitized_query.strip()
        lowered = text.lower()

        instrumental_only = any(_has_phrase(lowered, cue) for cue in _INSTRUMENTAL_CUES)
        exclude_explicit = any(_has_phrase(lowered, cue) for cue in _CLEAN_CUES)
        genre = next((g for g in _GENRES if _has_phrase(lowered, g)), None)
        mood = next((m for m in _MOODS if _has_phrase(lowered, m)), None)

        recognized = bool(genre or mood or instrumental_only or exclude_explicit)
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
            limit=limit,
        )
