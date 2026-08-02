"""Natural-language music companion: the bounded agent behind Cadence.

It wires the input/privacy guard, the deterministic intent parser, retrieval,
MMR diversity, the grounding evaluator, and Cadence's voice into one bounded flow
that returns a validated ``CompanionResponse`` with a privacy-safe ``AgentTrace``.
Every outcome is one of a small, allowlisted set of actions. Sensitive input is
kept entirely local — it reaches neither the retrieval provider nor the language
provider — and the trace records categories and ids, never raw sensitive text.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Sequence
from uuid import uuid4

from src.contracts import (
    AgentTrace,
    CatalogDescriptor,
    CatalogTrack,
    CompanionAction,
    CompanionResponse,
    CompanionTurn,
    ContextGuide,
    EmbeddingSource,
    EvaluationReport,
    ExecutionPolicy,
    GuardCategory,
    MusicIntent,
    OperatingMode,
    PipelineReceipt,
    RetrievalHit,
    SignalComparison,
    SignalRow,
    TrackRef,
    VoiceSource,
)
from src.evaluator import GroundingEvaluator
from src.fusion import fuse_pool
from src.generation import TextGenerator
from src.guard import InputGuard
from src.intent import IntentParser
from src.observability import EventSink, NullEventSink, build_event
from src.ranking import diversity_parameters, mmr_rerank, mood_similarity
from src.refine import combine_guard_categories, merge_intents
from src.retrieval import Retriever, TfidfRetriever
from src.scoring import candidates_from_hits
from src.voice import CadenceVoice


SAFE_RESPONSE_MESSAGE = (
    "I can only help with music, and it sounds like you may be going through "
    "something serious. I'm not able to help with that — but please consider "
    "reaching out to someone you trust or your local emergency services."
)

# Only parser/UI-owned identifiers may appear verbatim in a privacy-safe trace.
# A public MusicIntent may carry an extension cue, but user-authored text must
# never hitch a ride through ``cue_id`` into JSONL or the developer drawer.
TRACE_CUE_IDS = frozenset(
    {
        "energy_low_v1",
        "energy_high_v1",
        "acoustic_high_v1",
        "acoustic_low_v1",
        "valence_high_v1",
        "valence_low_v1",
        "dance_high_v1",
        "dance_low_v1",
        "tempo_near_v1",
        "tempo_range_v1",
        "tempo_atleast_v1",
        "tempo_atmost_v1",
        "tempo_near_ui_v1",
        "ui_danceability_low_v1",
        "ui_acousticness_low_v1",
        "instrumentalness_prefer_high_v1",
        "instrumentalness_prefer_low_v1",
        "mood_upbeat_energy_prefer_high_v1",
        "mood_upbeat_valence_prefer_high_v1",
        "mood_calm_energy_prefer_low_v1",
        "mood_calm_valence_prefer_high_v1",
        "mood_intense_energy_prefer_high_v1",
        "mood_intense_valence_prefer_low_v1",
        "mood_somber_energy_prefer_low_v1",
        "mood_somber_valence_prefer_low_v1",
    }
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
        event_sink: EventSink | None = None,
        config_fingerprint: str | None = None,
        catalog_descriptor: CatalogDescriptor | None = None,
        valid_ids: Collection[int] | None = None,
        valid_genres: Collection[str] | None = None,
        valid_moods: Collection[str] | None = None,
        catalog_artifact_source: str | None = None,
        catalog_warnings: Sequence[str] = (),
    ) -> None:
        self._guard = InputGuard()
        self._catalog_descriptor = catalog_descriptor
        self._catalog_artifact_source = catalog_artifact_source
        self._catalog_warnings = tuple(catalog_warnings)
        self._catalog_id = (
            catalog_descriptor.catalog_id
            if catalog_descriptor is not None
            else (tracks[0].catalog_id if tracks else "fictional")
        )
        is_fma = self._catalog_id == "fma"
        supports_instrumentalness = bool(
            catalog_descriptor
            and "instrumentalness"
            in catalog_descriptor.capabilities.supported_features
        )
        self._parser = IntentParser(
            experimental_mood_axes=is_fma,
            soft_instrumentalness=is_fma and supports_instrumentalness,
        )
        self._evaluator = GroundingEvaluator()
        self._voice = CadenceVoice(self._evaluator)
        self._generator = generator
        self._valid_ids = (
            set(valid_ids) if valid_ids is not None else {track.id for track in tracks}
        )
        self._valid_genres = (
            set(valid_genres)
            if valid_genres is not None
            else {track.genre for track in tracks if track.genre is not None}
        )
        self._valid_moods = (
            set(valid_moods)
            if valid_moods is not None
            else {track.mood for track in tracks if track.mood is not None}
        )
        self._local = (
            local_retriever if local_retriever is not None else TfidfRetriever(tracks, guides)
        )
        self._default = default_retriever if default_retriever is not None else self._local
        self._events = event_sink if event_sink is not None else NullEventSink()
        self._config_fingerprint = config_fingerprint

    @property
    def catalog_descriptor(self) -> CatalogDescriptor | None:
        """Describe the active catalog artifact without exposing mutable storage."""
        return self._catalog_descriptor

    @property
    def catalog_id(self) -> str:
        return self._catalog_id

    @property
    def catalog_artifact_source(self) -> str | None:
        return self._catalog_artifact_source

    @property
    def catalog_warnings(self) -> tuple[str, ...]:
        return self._catalog_warnings

    def respond(
        self,
        text: str,
        *,
        limit: int = 5,
        policy: ExecutionPolicy | None = None,
    ) -> CompanionResponse:
        """Guard, parse, retrieve, diversify, evaluate, and voice one query.

        A thin public wrapper: it times the turn, mints an ephemeral request id,
        and emits one privacy-safe receipt to the event sink. The decision logic
        lives in ``_respond``; emission never alters or blocks the response.
        """
        return self.respond_detailed(text, limit=limit, policy=policy).response

    def respond_detailed(
        self,
        text: str,
        *,
        limit: int = 5,
        policy: ExecutionPolicy | None = None,
    ) -> CompanionTurn:
        """Return the response plus request-local diagnostics for a trusted UI."""
        selected_policy = policy or ExecutionPolicy()
        return self._recorded(
            lambda: self._respond(text, limit=limit, policy=selected_policy),
            selected_policy,
        )

    def respond_with_intent(
        self,
        intent: MusicIntent,
        *,
        category: GuardCategory,
        policy: ExecutionPolicy | None = None,
    ) -> CompanionResponse:
        """Re-run a typed intent while re-inspecting its query for safety.

        ``category`` is deliberately required: it carries sticky sensitivity
        from the guarded turn that produced the intent.  The query is inspected
        again, so constructing a ``MusicIntent`` manually cannot bypass the guard.
        """
        return self.respond_with_intent_detailed(
            intent, category=category, policy=policy
        ).response

    def respond_with_intent_detailed(
        self,
        intent: MusicIntent,
        *,
        category: GuardCategory,
        policy: ExecutionPolicy | None = None,
    ) -> CompanionTurn:
        selected_policy = policy or ExecutionPolicy()
        return self._recorded(
            lambda: self._respond_with_intent(
                intent, category=category, policy=selected_policy
            ),
            selected_policy,
        )

    def refine(
        self,
        base_intent: MusicIntent,
        text: str,
        *,
        base_category: GuardCategory,
        policy: ExecutionPolicy | None = None,
    ) -> CompanionResponse:
        """Guard a follow-up, merge its parsed facets, and run one new turn."""
        return self.refine_detailed(
            base_intent,
            text,
            base_category=base_category,
            policy=policy,
        ).response

    def refine_detailed(
        self,
        base_intent: MusicIntent,
        text: str,
        *,
        base_category: GuardCategory,
        policy: ExecutionPolicy | None = None,
    ) -> CompanionTurn:
        selected_policy = policy or ExecutionPolicy()
        return self._recorded(
            lambda: self._refine(
                base_intent,
                text,
                base_category=base_category,
                policy=selected_policy,
            ),
            selected_policy,
        )

    def _recorded(
        self,
        operation: Callable[
            [], tuple[CompanionResponse, tuple[int, ...], SignalComparison | None]
        ],
        policy: ExecutionPolicy,
    ) -> CompanionTurn:
        """Time one submitted transaction, return its receipt, and emit once."""
        request_id = uuid4().hex
        start = time.perf_counter()
        response, candidate_ids, comparison = operation()
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        receipt = self._receipt(
            request_id, response, candidate_ids, latency_ms, policy
        )
        self._emit(request_id, response, candidate_ids, latency_ms, policy)
        return CompanionTurn(response=response, receipt=receipt, comparison=comparison)

    def _emit(
        self,
        request_id: str,
        response: CompanionResponse,
        candidate_ids: tuple[int, ...],
        latency_ms: float,
        policy: ExecutionPolicy,
    ) -> None:
        """Record one receipt, best-effort. Logging must never break a response."""
        if isinstance(self._events, NullEventSink):
            return  # no work when observability is off
        try:
            candidates = (
                candidates_from_hits(response.retrieval.hits) if response.retrieval else ()
            )
            self._events.record(
                build_event(
                    request_id=request_id,
                    response=response,
                    candidates=candidates,
                    candidate_ids=candidate_ids,
                    candidate_refs=tuple(
                        TrackRef(catalog_id=self._catalog_id, track_id=track_id)
                        for track_id in candidate_ids
                    ),
                    latency_ms=latency_ms,
                    config_fingerprint=self._config_fingerprint,
                    force_local=receipt_force_local(response, policy),
                    diversity=policy.diversity,
                )
            )
        except Exception:  # noqa: BLE001 - observability is best-effort
            pass

    def _respond(
        self, text: str, *, limit: int = 5, policy: ExecutionPolicy
    ) -> tuple[CompanionResponse, tuple[int, ...], SignalComparison | None]:
        """Run the bounded flow, returning the response and the candidate-pool ids."""
        verdict = self._guard.inspect(text)

        if verdict.category is GuardCategory.HIGH_RISK:
            return (
                self._simple(verdict.category, CompanionAction.SAFE_RESPONSE, SAFE_RESPONSE_MESSAGE),
                (),
                None,
            )
        if verdict.category in (GuardCategory.EMPTY, GuardCategory.TOO_LONG):
            return (
                self._simple(
                    verdict.category,
                    CompanionAction.CLARIFY,
                    "Tell me in a few words what you'd like to hear.",
                ),
                (),
                None,
            )

        intent = self._parser.parse(verdict.sanitized_query, limit=limit)
        if intent.needs_clarification:
            return (
                self._simple(
                    verdict.category,
                    CompanionAction.CLARIFY,
                    intent.clarification or "What would you like to hear?",
                    intent=intent,
                ),
                (),
                None,
            )

        return self._retrieve_and_voice(intent, verdict.category, policy=policy)

    def _respond_with_intent(
        self,
        intent: MusicIntent,
        *,
        category: GuardCategory,
        policy: ExecutionPolicy,
    ) -> tuple[CompanionResponse, tuple[int, ...], SignalComparison | None]:
        """Validate the trusted-intent boundary before executing it."""
        verdict = self._guard.inspect(intent.query)
        combined = combine_guard_categories(category, verdict.category)
        if combined is GuardCategory.HIGH_RISK:
            return (
                self._simple(combined, CompanionAction.SAFE_RESPONSE, SAFE_RESPONSE_MESSAGE),
                (),
                None,
            )
        if verdict.category in (GuardCategory.EMPTY, GuardCategory.TOO_LONG):
            return (
                self._simple(
                    combined,
                    CompanionAction.CLARIFY,
                    "Tell me in a few words what you'd like to hear.",
                    intent=intent,
                ),
                (),
                None,
            )
        guarded_intent = intent.model_copy(
            update={"query": verdict.sanitized_query}
        )
        if guarded_intent.needs_clarification:
            return (
                self._simple(
                    combined,
                    CompanionAction.CLARIFY,
                    guarded_intent.clarification or "What would you like to hear?",
                    intent=guarded_intent,
                ),
                (),
                None,
            )
        return self._retrieve_and_voice(guarded_intent, combined, policy=policy)

    def _refine(
        self,
        base_intent: MusicIntent,
        text: str,
        *,
        base_category: GuardCategory,
        policy: ExecutionPolicy,
    ) -> tuple[CompanionResponse, tuple[int, ...], SignalComparison | None]:
        """Guard and parse one free-text refinement without mutating the base."""
        base_verdict = self._guard.inspect(base_intent.query)
        follow_verdict = self._guard.inspect(text)
        combined = combine_guard_categories(base_category, base_verdict.category)
        combined = combine_guard_categories(combined, follow_verdict.category)
        if combined is GuardCategory.HIGH_RISK:
            return (
                self._simple(combined, CompanionAction.SAFE_RESPONSE, SAFE_RESPONSE_MESSAGE),
                (),
                None,
            )
        if follow_verdict.category in (GuardCategory.EMPTY, GuardCategory.TOO_LONG):
            return (
                self._simple(
                    combined,
                    CompanionAction.CLARIFY,
                    "Give me one musical change, like ‘calmer’ or ‘more acoustic.’",
                    intent=base_intent,
                ),
                (),
                None,
            )
        follow_intent = self._parser.parse(
            follow_verdict.sanitized_query, limit=base_intent.limit
        )
        if follow_intent.needs_clarification:
            return (
                self._simple(
                    combined,
                    CompanionAction.CLARIFY,
                    "I couldn't map that to a musical change yet. Try ‘calmer,’ "
                    "‘more acoustic,’ ‘brighter,’ or a BPM.",
                    intent=base_intent,
                ),
                (),
                None,
            )
        if not any(
            (
                follow_intent.genre,
                follow_intent.mood,
                follow_intent.instrumental_only,
                follow_intent.exclude_explicit,
                follow_intent.feature_goals,
            )
        ):
            return (
                self._simple(
                    combined,
                    CompanionAction.CLARIFY,
                    "I couldn't map that to a supported musical change yet. Try "
                    "‘calmer,’ ‘more acoustic,’ ‘brighter,’ ‘more movement,’ "
                    "‘instrumental,’ ‘clean,’ or a BPM.",
                    intent=base_intent,
                ),
                (),
                None,
            )
        sanitized_base = base_intent.model_copy(
            update={"query": base_verdict.sanitized_query}
        )
        merged = merge_intents(sanitized_base, follow_intent)
        merged_verdict = self._guard.inspect(merged.query)
        combined = combine_guard_categories(combined, merged_verdict.category)
        if merged_verdict.category is GuardCategory.TOO_LONG:
            return (
                self._simple(
                    combined,
                    CompanionAction.CLARIFY,
                    "That refinement thread is getting long. Start a fresh mix with the key idea.",
                    intent=base_intent,
                ),
                (),
                None,
            )
        merged = merged.model_copy(update={"query": merged_verdict.sanitized_query})
        return self._retrieve_and_voice(merged, combined, policy=policy)

    def _retrieve_and_voice(
        self,
        intent: MusicIntent,
        category: GuardCategory,
        *,
        policy: ExecutionPolicy,
    ) -> tuple[CompanionResponse, tuple[int, ...], SignalComparison | None]:
        """Run the shared post-guard pipeline for text and controlled refinements."""
        unsupported = self._unsupported_capability(intent)
        if unsupported is not None:
            return (
                self._simple(
                    category,
                    CompanionAction.CLARIFY,
                    unsupported,
                    intent=intent,
                ),
                (),
                None,
            )

        # Retrieve a pool, fuse in structured preferences (when present), then
        # diversify down to the requested count. A larger pool when structured
        # signal exists gives the structured leg room to lift a well-matched track
        # from just outside the text top-k; without it, the path is today's exactly.
        sensitive = category is GuardCategory.SENSITIVE
        force_local = sensitive or policy.force_local
        retriever = self._local if force_local else self._default
        structured_active = bool(intent.genre or intent.mood or intent.feature_goals)
        pool_k = max(intent.limit * 8, 40) if structured_active else max(intent.limit * 4, 12)
        intent_search = getattr(retriever, "search_with_intent", None)
        intent_aware = callable(intent_search)
        if intent_aware:
            pool = intent_search(intent, k=pool_k)
        else:
            pool = retriever.search(
                intent.query,
                k=pool_k,
                instrumental_only=intent.instrumental_only,
                exclude_explicit=intent.exclude_explicit,
            )
        candidate_ids = tuple(hit.track.id for hit in pool.hits)
        # Keep the pre-fusion pool so the developer view can show, honestly, how
        # the same candidates would rank under text alone vs structured vs fused.
        text_hits = pool.hits
        if structured_active and pool.hits and not intent_aware:
            pool = pool.model_copy(update={"hits": fuse_pool(intent, pool.hits)})
        comparison = _build_comparison(text_hits, pool.hits, structured_active)
        # When a genre is explicitly requested, diversify by mood *within* it
        # rather than dragging in other genres against the listener's wish.
        similarity = mood_similarity if intent.genre else None
        lambda_, relevance_floor = diversity_parameters(policy.diversity)
        diversified = (
            mmr_rerank(
                pool.hits,
                intent.limit,
                lambda_=lambda_,
                relevance_floor=relevance_floor,
                similarity=similarity,
            )
            if pool.hits
            else ()
        )
        # MMR runs for every non-empty pool and can reorder even when the pool is
        # smaller than the requested limit. Report what actually happened, not a
        # size-based proxy that can make the trace contradict the visible order.
        diversity_applied = diversified != tuple(pool.hits[: intent.limit])
        result = pool.model_copy(update={"hits": diversified})

        evaluation = self._evaluator.evaluate_result(intent, diversified, self._valid_ids)

        if not diversified or not evaluation.ok:
            # The trace and request-local receipt retain the attempted/rejected
            # candidate IDs for diagnosis. The public result must not carry hits
            # that failed the output evaluator: NO_MATCH means exactly zero
            # publishable recommendations.
            empty_result = result.model_copy(update={"hits": ()})
            return (
                CompanionResponse(
                    action=CompanionAction.NO_MATCH,
                    message=(
                        "I couldn't find a match I can stand behind for that. "
                        "Try naming a genre, a mood, or an activity."
                    ),
                    retrieval=empty_result,
                    intent=intent,
                    trace=AgentTrace(
                        guard_category=category,
                        intent_summary=self._intent_summary(intent),
                        retrieved_ids=tuple(hit.track.id for hit in diversified),
                        retrieved_refs=tuple(hit.track.ref for hit in diversified),
                        diversity_applied=diversity_applied,
                        evaluation=evaluation,
                        action=CompanionAction.NO_MATCH,
                        network_used=(
                            result.embedding_source is EmbeddingSource.LIVE
                        ),
                        fallback_reason=None if diversified else "no candidates",
                    ),
                ),
                candidate_ids,
                comparison,
            )

        # Sensitive queries never reach the language provider either.
        provider_failed = (
            result.operating_mode is OperatingMode.DEGRADED
            and result.embedding_source is EmbeddingSource.LIVE
        )
        # One failed provider leg is enough for this interactive turn. Do not
        # immediately spend a second retry budget on optional microcopy selection.
        generator = None if force_local or provider_failed else self._generator
        voice = self._voice.render(diversified, intent, generator=generator)
        message = self._decorate(voice.message, category)
        intro_message = self._decorate(voice.framing, category)

        action = (
            CompanionAction.DEGRADED
            if result.operating_mode is OperatingMode.DEGRADED
            else CompanionAction.RECOMMEND
        )
        return (
            CompanionResponse(
                action=action,
                message=message,
                intro_message=intro_message,
                retrieval=result,
                intent=intent,
                trace=AgentTrace(
                    guard_category=category,
                    intent_summary=self._intent_summary(intent),
                    retrieved_ids=tuple(hit.track.id for hit in diversified),
                    retrieved_refs=tuple(hit.track.ref for hit in diversified),
                    diversity_applied=diversity_applied,
                    evaluation=evaluation,
                    text_evaluation=voice.text_evaluation,
                    action=action,
                    voice_source=voice.source,
                    voice_model=voice.model,
                    network_used=(
                        result.embedding_source is EmbeddingSource.LIVE
                    )
                    or voice.network_used,
                    fallback_reason=voice.fallback_reason,
                ),
            ),
            candidate_ids,
            comparison,
        )

    def _receipt(
        self,
        request_id: str,
        response: CompanionResponse,
        candidate_ids: tuple[int, ...],
        latency_ms: float,
        policy: ExecutionPolicy,
    ) -> PipelineReceipt:
        retrieval = response.retrieval
        trace = response.trace
        return PipelineReceipt(
            request_id=request_id,
            latency_ms=latency_ms,
            candidate_ids=candidate_ids,
            candidate_refs=tuple(
                TrackRef(catalog_id=self._catalog_id, track_id=track_id)
                for track_id in candidate_ids
            ),
            final_ids=(
                tuple(hit.track.id for hit in retrieval.hits) if retrieval else ()
            ),
            final_refs=(
                tuple(hit.track.ref for hit in retrieval.hits) if retrieval else ()
            ),
            guard_category=(trace.guard_category if trace else GuardCategory.OK),
            action=response.action,
            force_local=receipt_force_local(response, policy),
            diversity=policy.diversity,
            embedding_source=(retrieval.embedding_source if retrieval else None),
            network_used=(trace.network_used if trace else False),
            operating_mode=(retrieval.operating_mode if retrieval else None),
            voice_source=(trace.voice_source if trace else None),
            index_fingerprint=(retrieval.index_fingerprint if retrieval else None),
            config_fingerprint=self._config_fingerprint,
        )

    def _unsupported_capability(self, intent: MusicIntent) -> str | None:
        """Clarify hard constraints this catalog cannot prove, before retrieval."""
        descriptor = self._catalog_descriptor
        if descriptor is None:
            return None  # legacy fictional construction supports both booleans
        capabilities = descriptor.capabilities
        missing_clean = (
            intent.exclude_explicit
            and not capabilities.supports_filter("exclude_explicit")
        )
        missing_instrumental = (
            intent.instrumental_only
            and not capabilities.supports_filter("instrumental_only")
        )
        if not missing_clean and not missing_instrumental:
            return None
        if missing_clean and missing_instrumental:
            unknown = "clean lyrics or instrumental-only status"
            requirement = "those requirements"
        elif missing_clean:
            unknown = "clean lyrics"
            requirement = "that requirement"
        else:
            unknown = "instrumental-only status"
            requirement = "that requirement"
        return (
            f"I can’t verify {unknown} in this catalog, and I’d rather not guess. "
            f"Remove {requirement} or switch to the fictional catalog."
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
            intro_message=message,
            intent=intent,
            trace=AgentTrace(
                guard_category=category,
                intent_summary=self._intent_summary(intent) if intent else "",
                action=action,
                evaluation=EvaluationReport(ok=True),
                voice_source=VoiceSource.TEMPLATE,
            ),
        )

    def _intent_summary(self, intent: MusicIntent) -> str:
        """A privacy-safe summary (no free-text query) for the trace.

        Known feature goals appear as controlled cue ids. Unknown extension cues
        collapse to their typed feature/relation so arbitrary text can never leak
        through a public manually-constructed intent.
        """
        genre = intent.genre if intent.genre in self._valid_genres else None
        mood = intent.mood if intent.mood in self._valid_moods else None
        goals = ",".join(
            goal.cue_id
            if goal.cue_id in TRACE_CUE_IDS
            else f"{goal.feature}:{goal.relation.value}"
            for goal in intent.feature_goals
        )
        return (
            f"genre={genre}, mood={mood}, "
            f"instrumental_only={intent.instrumental_only}, clean={intent.exclude_explicit}, "
            f"goals=[{goals}]"
        )

    @staticmethod
    def _decorate(message: str, category: GuardCategory) -> str:
        if category is GuardCategory.SENSITIVE:
            return message + "\n(I kept this local and ignored the personal details.)"
        if category is GuardCategory.INJECTION:
            return message + "\n(I ignored an instruction embedded in your message.)"
        return message


def _build_comparison(
    text_hits: Sequence[RetrievalHit],
    fused_hits: Sequence[RetrievalHit],
    structured_active: bool,
) -> SignalComparison | None:
    """Assemble a developer-only per-leg view of the candidate pool.

    ``text_hits`` are the pool before fusion (``score`` is the text leg's
    semantic+lexical value); ``fused_hits`` carry the fused score and structured
    relevance. When the structured leg did not run, ``fused`` equals ``text`` and
    the structured column is ``None`` — shown honestly rather than faked.
    """
    if not text_hits:
        return None
    text_by_id = {
        hit.track.id: (
            hit.lexical_score
            if hit.lexical_score is not None
            else (
                hit.semantic_score
                if hit.semantic_score is not None
                else hit.score
            )
        )
        for hit in text_hits
    }
    rows = tuple(
        SignalRow(
            track_id=hit.track.id,
            track_ref=hit.track.ref,
            title=hit.track.title,
            text=text_by_id.get(hit.track.id, hit.score),
            structured=hit.structured_score if structured_active else None,
            fused=hit.score,
        )
        for hit in fused_hits
    )
    return SignalComparison(structured_active=structured_active, rows=rows)


def receipt_force_local(
    response: CompanionResponse, policy: ExecutionPolicy
) -> bool:
    """Whether local-only execution was effectively enforced for this turn."""
    trace = response.trace
    return policy.force_local or (
        trace is not None and trace.guard_category is GuardCategory.SENSITIVE
    )
