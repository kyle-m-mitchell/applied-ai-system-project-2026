"""Cadence — the companion's voice.

Cadence is a warm, observant *fictional* DJ: concise, tasteful, and honest. Her
personality is a presentation layer only; the song facts always come from the
validated evidence, never from the model. The deterministic renderer is the
reproducible baseline and fallback; an optional model selects one application-owned
framing line, which must pass the exact allowlist guard or be discarded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.contracts import EvaluationReport, MusicIntent, RetrievalHit, VoiceSource
from src.evaluator import APPROVED_FRAMINGS, GroundingEvaluator
from src.generation import FewShot, TextGenerator


@dataclass(frozen=True)
class VoiceResult:
    """The outcome of rendering — message plus how it was produced."""

    message: str
    source: VoiceSource
    framing: str
    model: str | None = None
    network_used: bool = False
    fallback_reason: str | None = None
    text_evaluation: EvaluationReport | None = None


# The voice card: the system instruction that shapes Cadence's framing sentence.
# It mirrors the product vision's may/must-not rules (see docs/CADENCE_VOICE.md).
CADENCE_SYSTEM = (
    "You are Cadence, a warm, observant fictional radio DJ for a music app. "
    "You will be given a listener's guarded request after the app has already "
    "chosen and validated a set. Select exactly ONE approved Cadence line from "
    "the list below. Copy it character-for-character with no label, explanation, "
    "or extra text. You are choosing tone, not writing facts.\nApproved lines:\n- "
    + "\n- ".join(APPROVED_FRAMINGS)
)

CADENCE_FEW_SHOT: FewShot = (
    (
        "Request: late-night study focus\nTracks: 3 calm lofi tracks",
        "Here's a thoughtfully chosen set for the moment you described.",
    ),
    (
        "Request: something for a rainy, reflective evening\nTracks: 3 slow blues and soul tracks",
        "I found a few picks worth meeting right where you are.",
    ),
)


class CadenceVoice:
    """Render a recommendation set in Cadence's grounded voice."""

    def __init__(self, evaluator: GroundingEvaluator | None = None) -> None:
        self._evaluator = evaluator if evaluator is not None else GroundingEvaluator()

    EXPLORATORY_FRAMING = (
        "I wasn't sure exactly what you meant, so here's a varied starting "
        "point — shape it below."
    )

    def render(
        self,
        hits: Sequence[RetrievalHit],
        intent: MusicIntent,
        *,
        generator: TextGenerator | None = None,
        exploratory: bool = False,
    ) -> VoiceResult:
        """Render a track set in Cadence's voice, falling back safely."""
        track_block = self._render_tracks(hits)
        template_framing = self._template_framing(intent)

        # A best-effort starting set is framed honestly and deterministically; the
        # provider only knows "we matched your request" lines, which would overclaim.
        if exploratory:
            return VoiceResult(
                f"{self.EXPLORATORY_FRAMING}\n{track_block}",
                VoiceSource.TEMPLATE,
                framing=self.EXPLORATORY_FRAMING,
            )

        if generator is None:
            return VoiceResult(
                f"{template_framing}\n{track_block}",
                VoiceSource.TEMPLATE,
                framing=template_framing,
            )

        try:
            framing = generator.generate(
                CADENCE_SYSTEM, CADENCE_FEW_SHOT, self._evidence_packet(hits, intent)
            ).strip()
        except Exception:  # noqa: BLE001 - provider failed; fall back honestly
            return VoiceResult(
                f"{template_framing}\n{track_block}",
                VoiceSource.TEMPLATE,
                framing=template_framing,
                network_used=generator.is_remote,
                fallback_reason="generation failed",
            )

        allowed = [hit.track.title for hit in hits] + [hit.track.artist for hit in hits]
        report = self._evaluator.check_grounded_text(framing, allowed)
        if framing and report.ok:
            return VoiceResult(
                f"{framing}\n{track_block}",
                VoiceSource.GENERATED,
                framing=framing,
                model=generator.model_id,
                network_used=generator.is_remote,
                text_evaluation=report,
            )
        return VoiceResult(
            f"{template_framing}\n{track_block}",
            VoiceSource.TEMPLATE,
            framing=template_framing,
            network_used=generator.is_remote,
            fallback_reason="generated text failed grounding",
            text_evaluation=report,
        )

    @staticmethod
    def _template_framing(intent: MusicIntent) -> str:
        bits = []
        if intent.instrumental_only:
            bits.append("instrumental")
        if intent.exclude_explicit:
            bits.append("clean")
        tag = f" ({', '.join(bits)})" if bits else ""
        return f"Here are a few picks{tag} for that:"

    @staticmethod
    def _render_tracks(hits: Sequence[RetrievalHit]) -> str:
        lines = []
        for rank, hit in enumerate(hits, start=1):
            track = hit.track
            if hit.semantic_score is not None and hit.semantic_score > 0.0:
                why = " — a close match in feel"
            elif hit.matched_terms:
                why = " — " + ", ".join(hit.matched_terms[:3])
            else:
                why = ""
            details: list[str] = []
            if track.genre:
                details.append(track.genre)
            if track.mood:
                details.append(track.mood)
            elif track.mood_profile and track.mood_profile.label:
                details.append(f"{track.mood_profile.label.value} (experimental)")
            evidence = f" [{' · '.join(details)}]" if details else ""
            lines.append(f"{rank}. {track.title} — {track.artist}{evidence}{why}")
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
        lines.append(
            f"Tracks: {len(hits)} app-validated tracks. Do not describe their properties."
        )
        return "\n".join(lines)
