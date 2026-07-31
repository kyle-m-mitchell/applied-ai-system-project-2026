"""Natural-language music companion: the public front door for typed queries.

It wires the input/privacy guard, the deterministic intent parser, and the
retrievers into one bounded flow that returns a validated ``CompanionResponse``.
This is where a real sentence finally reaches retrieval through the public path
(the demo and tests exercised the retrievers directly). Sensitive input is kept
local — never sent to the provider — and every outcome is one of a small,
predictable set of actions. Phase 5's Cadence persona will render the ``message``
more warmly; here it stays plain and honest.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.contracts import (
    CatalogTrack,
    CompanionAction,
    CompanionResponse,
    ContextGuide,
    GuardCategory,
    MusicIntent,
    OperatingMode,
)
from src.guard import InputGuard
from src.intent import IntentParser
from src.retrieval import Retriever, TfidfRetriever


SAFE_RESPONSE_MESSAGE = (
    "I can only help with music, and it sounds like you may be going through "
    "something serious. I'm not able to help with that — but please consider "
    "reaching out to someone you trust or your local emergency services."
)


class MusicCompanion:
    """Bounded natural-language entry point over the validated catalog."""

    def __init__(
        self,
        tracks: Sequence[CatalogTrack],
        guides: Sequence[ContextGuide] = (),
        *,
        default_retriever: Retriever | None = None,
        local_retriever: Retriever | None = None,
    ) -> None:
        self._guard = InputGuard()
        self._parser = IntentParser()
        self._local = (
            local_retriever if local_retriever is not None else TfidfRetriever(tracks, guides)
        )
        self._default = default_retriever if default_retriever is not None else self._local

    def respond(self, text: str, *, limit: int = 5) -> CompanionResponse:
        """Guard, parse, and answer one natural-language query."""
        verdict = self._guard.inspect(text)

        if verdict.category is GuardCategory.HIGH_RISK:
            return CompanionResponse(
                action=CompanionAction.SAFE_RESPONSE, message=SAFE_RESPONSE_MESSAGE
            )
        if verdict.category in (GuardCategory.EMPTY, GuardCategory.TOO_LONG):
            return CompanionResponse(
                action=CompanionAction.CLARIFY,
                message="Tell me in a few words what you'd like to hear.",
            )

        intent = self._parser.parse(verdict.sanitized_query, limit=limit)
        if intent.needs_clarification:
            return CompanionResponse(
                action=CompanionAction.CLARIFY,
                message=intent.clarification or "What would you like to hear?",
                intent=intent,
            )

        # Sensitive queries must never reach the provider — use the local retriever.
        retriever = self._local if verdict.category is GuardCategory.SENSITIVE else self._default
        result = retriever.search(
            intent.query,
            k=intent.limit,
            instrumental_only=intent.instrumental_only,
            exclude_explicit=intent.exclude_explicit,
        )

        if not result.hits:
            return CompanionResponse(
                action=CompanionAction.NO_MATCH,
                message=(
                    "I couldn't find a good match for that. "
                    "Try naming a genre, a mood, or an activity."
                ),
                retrieval=result,
                intent=intent,
            )

        action = (
            CompanionAction.DEGRADED
            if result.operating_mode is OperatingMode.DEGRADED
            else CompanionAction.RECOMMEND
        )
        return CompanionResponse(
            action=action,
            message=self._recommend_message(intent, verdict.category),
            retrieval=result,
            intent=intent,
        )

    @staticmethod
    def _recommend_message(intent: MusicIntent, category: GuardCategory) -> str:
        filters = []
        if intent.instrumental_only:
            filters.append("instrumental only")
        if intent.exclude_explicit:
            filters.append("clean")
        message = "Here's what I found"
        if filters:
            message += f" ({', '.join(filters)})"
        message += "."
        if category is GuardCategory.SENSITIVE:
            message += " I kept your request local and ignored the personal details."
        elif category is GuardCategory.INJECTION:
            message += " I ignored an instruction embedded in your message."
        return message
