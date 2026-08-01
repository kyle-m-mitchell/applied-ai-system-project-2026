"""Cadence — the companion's voice.

Cadence is a warm, observant *fictional* DJ: concise, tasteful, and honest. Her
personality is a presentation layer only; the song facts always come from the
validated evidence, never from the model. The deterministic renderer is the
reproducible baseline and fallback; an optional generator writes only the warm
*framing*, which must pass the grounding check or be discarded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.contracts import EvaluationReport, MusicIntent, RetrievalHit, VoiceSource
from src.evaluator import GroundingEvaluator
from src.generation import FewShot, TextGenerator


@dataclass(frozen=True)
class VoiceResult:
    """The outcome of rendering — message plus how it was produced."""

    message: str
    source: VoiceSource
    model: str | None = None
    fallback_reason: str | None = None
    text_evaluation: EvaluationReport | None = None


# The voice card: the system instruction that shapes Cadence's framing sentence.
# It mirrors the product vision's may/must-not rules (see docs/CADENCE_VOICE.md).
CADENCE_SYSTEM = (
    "You are Cadence, a warm, observant fictional radio DJ for a music app. "
    "You will be given a listener's request and a short list of real tracks the "
    "app has already chosen. Write ONE friendly, concise framing sentence (about "
    "25 words) about why this set suits the request. "
    "Rules: do NOT name any song or artist and do NOT use quotation marks — the "
    "app lists the tracks itself. Never claim to have listened to a track, to have "
    "feelings, or to be human. Never invent songs, artists, or facts. Stay on "
    "music."
)

CADENCE_FEW_SHOT: FewShot = (
    (
        "Request: late-night study focus\nTracks: 3 calm lofi tracks",
        "For late-night focus, here's a calm, low-key set that stays out of your "
        "way so your attention stays on the work.",
    ),
    (
        "Request: something for a rainy, reflective evening\nTracks: 3 slow blues and soul tracks",
        "For a rainy, reflective evening, these lean slow and warm — good company "
        "for sitting with the mood rather than shaking it off.",
    ),
)


class CadenceVoice:
    """Render a recommendation set in Cadence's grounded voice."""

    def __init__(self, evaluator: GroundingEvaluator | None = None) -> None:
        self._evaluator = evaluator if evaluator is not None else GroundingEvaluator()

    def render(
        self,
        hits: Sequence[RetrievalHit],
        intent: MusicIntent,
        *,
        generator: TextGenerator | None = None,
    ) -> VoiceResult:
        """Render a track set in Cadence's voice, falling back safely."""
        track_block = self._render_tracks(hits)

        if generator is None:
            return VoiceResult(self._template(intent, track_block), VoiceSource.TEMPLATE)

        try:
            framing = generator.generate(
                CADENCE_SYSTEM, CADENCE_FEW_SHOT, self._evidence_packet(hits, intent)
            ).strip()
        except Exception:  # noqa: BLE001 - provider failed; fall back honestly
            return VoiceResult(
                self._template(intent, track_block),
                VoiceSource.TEMPLATE,
                fallback_reason="generation failed",
            )

        allowed = [hit.track.title for hit in hits] + [hit.track.artist for hit in hits]
        report = self._evaluator.check_grounded_text(framing, allowed)
        if framing and report.ok:
            return VoiceResult(
                f"{framing}\n{track_block}",
                VoiceSource.GENERATED,
                model=generator.model_id,
                text_evaluation=report,
            )
        return VoiceResult(
            self._template(intent, track_block),
            VoiceSource.TEMPLATE,
            fallback_reason="generated text failed grounding",
            text_evaluation=report,
        )

    def _template(self, intent: MusicIntent, track_block: str) -> str:
        bits = []
        if intent.instrumental_only:
            bits.append("instrumental")
        if intent.exclude_explicit:
            bits.append("clean")
        tag = f" ({', '.join(bits)})" if bits else ""
        return f"Here are a few picks{tag} for that:\n{track_block}"

    @staticmethod
    def _render_tracks(hits: Sequence[RetrievalHit]) -> str:
        lines = []
        for rank, hit in enumerate(hits, start=1):
            track = hit.track
            if hit.semantic_score is not None:
                why = " — a close match in feel"
            elif hit.matched_terms:
                why = " — " + ", ".join(hit.matched_terms[:3])
            else:
                why = ""
            lines.append(
                f"{rank}. {track.title} — {track.artist} [{track.genre} · {track.mood}]{why}"
            )
        return "\n".join(lines)

    @staticmethod
    def _evidence_packet(hits: Sequence[RetrievalHit], intent: MusicIntent) -> str:
        lines = [f"Request: {intent.query}"]
        filters = []
        if intent.instrumental_only:
            filters.append("instrumental only")
        if intent.exclude_explicit:
            filters.append("clean")
        if filters:
            lines.append("Constraints: " + ", ".join(filters))
        genres = ", ".join(sorted({hit.track.genre for hit in hits}))
        lines.append(f"Tracks: {len(hits)} tracks ({genres})")
        return "\n".join(lines)
