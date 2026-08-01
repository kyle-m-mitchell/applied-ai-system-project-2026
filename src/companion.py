"""Natural-language music companion: the bounded agent behind Cadence.

It wires the input/privacy guard, the deterministic intent parser, retrieval,
MMR diversity, the grounding evaluator, and Cadence's voice into one bounded flow
that returns a validated ``CompanionResponse`` with a privacy-safe ``AgentTrace``.
Every outcome is one of a small, allowlisted set of actions. Sensitive input is
kept entirely local — it reaches neither the retrieval provider nor the language
provider — and the trace records categories and ids, never raw sensitive text.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.contracts import (
    AgentTrace,
    CatalogTrack,
    CompanionAction,
    CompanionResponse,
    ContextGuide,
    EvaluationReport,
    GuardCategory,
    MusicIntent,
    OperatingMode,
    VoiceSource,
)
from src.evaluator import GroundingEvaluator
from src.generation import TextGenerator
from src.guard import InputGuard
from src.intent import IntentParser
from src.ranking import mmr_rerank
from src.retrieval import Retriever, TfidfRetriever
from src.voice import CadenceVoice


SAFE_RESPONSE_MESSAGE = (
    "I can only help with music, and it sounds like you may be going through "
    "something serious. I'm not able to help with that — but please consider "
    "reaching out to someone you trust or your local emergency services."
)


class MusicCompanion:
    """Bounded natural-language agent over the validated catalog."""

    def __init__(
        self,
        tracks: Sequence[CatalogTrack],
        guides: Sequence[ContextGuide] = (),
        *,
        default_retriever: Retriever | None = None,
        local_retriever: Retriever | None = None,
        generator: TextGenerator | None = None,
    ) -> None:
        self._guard = InputGuard()
        self._parser = IntentParser()
        self._evaluator = GroundingEvaluator()
        self._voice = CadenceVoice(self._evaluator)
        self._generator = generator
        self._valid_ids = {track.id for track in tracks}
        self._local = (
            local_retriever if local_retriever is not None else TfidfRetriever(tracks, guides)
        )
        self._default = default_retriever if default_retriever is not None else self._local

    def respond(self, text: str, *, limit: int = 5) -> CompanionResponse:
        """Guard, parse, retrieve, diversify, evaluate, and voice one query."""
        verdict = self._guard.inspect(text)

        if verdict.category is GuardCategory.HIGH_RISK:
            return self._simple(verdict.category, CompanionAction.SAFE_RESPONSE, SAFE_RESPONSE_MESSAGE)
        if verdict.category in (GuardCategory.EMPTY, GuardCategory.TOO_LONG):
            return self._simple(
                verdict.category,
                CompanionAction.CLARIFY,
                "Tell me in a few words what you'd like to hear.",
            )

        intent = self._parser.parse(verdict.sanitized_query, limit=limit)
        if intent.needs_clarification:
            return self._simple(
                verdict.category,
                CompanionAction.CLARIFY,
                intent.clarification or "What would you like to hear?",
                intent=intent,
            )

        # Retrieve a larger pool, then diversify down to the requested count.
        sensitive = verdict.category is GuardCategory.SENSITIVE
        retriever = self._local if sensitive else self._default
        pool = retriever.search(
            intent.query,
            k=max(intent.limit * 4, 12),
            instrumental_only=intent.instrumental_only,
            exclude_explicit=intent.exclude_explicit,
        )
        diversified = mmr_rerank(pool.hits, intent.limit) if pool.hits else ()
        diversity_applied = len(pool.hits) > intent.limit
        result = pool.model_copy(update={"hits": diversified})

        evaluation = self._evaluator.evaluate_result(intent, diversified, self._valid_ids)

        if not diversified or not evaluation.ok:
            return CompanionResponse(
                action=CompanionAction.NO_MATCH,
                message=(
                    "I couldn't find a match I can stand behind for that. "
                    "Try naming a genre, a mood, or an activity."
                ),
                retrieval=result,
                intent=intent,
                trace=AgentTrace(
                    guard_category=verdict.category,
                    intent_summary=self._intent_summary(intent),
                    retrieved_ids=tuple(hit.track.id for hit in diversified),
                    diversity_applied=diversity_applied,
                    evaluation=evaluation,
                    action=CompanionAction.NO_MATCH,
                    fallback_reason=None if diversified else "no candidates",
                ),
            )

        # Sensitive queries never reach the language provider either.
        generator = None if sensitive else self._generator
        voice = self._voice.render(diversified, intent, generator=generator)
        message = self._decorate(voice.message, verdict.category)

        action = (
            CompanionAction.DEGRADED
            if result.operating_mode is OperatingMode.DEGRADED
            else CompanionAction.RECOMMEND
        )
        return CompanionResponse(
            action=action,
            message=message,
            retrieval=result,
            intent=intent,
            trace=AgentTrace(
                guard_category=verdict.category,
                intent_summary=self._intent_summary(intent),
                retrieved_ids=tuple(hit.track.id for hit in diversified),
                diversity_applied=diversity_applied,
                evaluation=evaluation,
                text_evaluation=voice.text_evaluation,
                action=action,
                voice_source=voice.source,
                voice_model=voice.model,
                fallback_reason=voice.fallback_reason,
            ),
        )

    def _simple(
        self,
        category: GuardCategory,
        action: CompanionAction,
        message: str,
        *,
        intent: MusicIntent | None = None,
    ) -> CompanionResponse:
        """Build a no-retrieval response (safe/clarify) with a minimal trace."""
        return CompanionResponse(
            action=action,
            message=message,
            intent=intent,
            trace=AgentTrace(
                guard_category=category,
                intent_summary=self._intent_summary(intent) if intent else "",
                action=action,
                evaluation=EvaluationReport(ok=True),
                voice_source=VoiceSource.TEMPLATE,
            ),
        )

    @staticmethod
    def _intent_summary(intent: MusicIntent) -> str:
        """A privacy-safe summary (no free-text query) for the trace."""
        return (
            f"genre={intent.genre}, mood={intent.mood}, "
            f"instrumental_only={intent.instrumental_only}, clean={intent.exclude_explicit}"
        )

    @staticmethod
    def _decorate(message: str, category: GuardCategory) -> str:
        if category is GuardCategory.SENSITIVE:
            return message + "\n(I kept this local and ignored the personal details.)"
        if category is GuardCategory.INJECTION:
            return message + "\n(I ignored an instruction embedded in your message.)"
        return message
