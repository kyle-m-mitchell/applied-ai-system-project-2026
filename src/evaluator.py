"""Grounding evaluator: the guardrail before the companion speaks.

It checks a result is trustworthy — real ids, no duplicates, hard constraints
held, evidence present, within the requested count — and that a rendered message
mentions only tracks that were actually retrieved. This is what lets an optional
generated voice be trusted: anything it invents is caught here and the system
falls back to the deterministic voice.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

from src.contracts import EvaluationReport, MusicIntent, RetrievalHit


# Song titles are conventionally double-quoted; contractions use apostrophes, so
# matching only double quotes avoids false positives on "don't", "it's", etc.
_QUOTED = re.compile(r"[\"“”]([^\"“”]{2,})[\"“”]")


class GroundingEvaluator:
    """Validate retrieval results and any generated text against the evidence."""

    def evaluate_result(
        self,
        intent: MusicIntent,
        hits: Sequence[RetrievalHit],
        valid_ids: Collection[int],
    ) -> EvaluationReport:
        """Confirm the hits are real, unique, constraint-respecting, and evidenced."""
        failures: list[str] = []
        ids = [hit.track.id for hit in hits]

        unknown = sorted({track_id for track_id in ids if track_id not in valid_ids})
        if unknown:
            failures.append(f"unknown track ids: {unknown}")
        if len(ids) != len(set(ids)):
            failures.append("duplicate track ids")
        if intent.instrumental_only and any(not hit.track.instrumental for hit in hits):
            failures.append("instrumental-only constraint violated")
        if intent.exclude_explicit and any(hit.track.explicit for hit in hits):
            failures.append("clean constraint violated")
        if any(not hit.matched_terms and hit.semantic_score is None for hit in hits):
            failures.append("a hit carries no evidence")
        if len(hits) > intent.limit:
            failures.append("more hits than requested")

        return EvaluationReport(ok=not failures, failures=tuple(failures))

    def check_grounded_text(
        self,
        text: str,
        allowed_titles: Collection[str],
    ) -> EvaluationReport:
        """Reject a message that quotes a song title not in the evidence packet."""
        if not text or not text.strip():
            return EvaluationReport(ok=False, failures=("empty message",))
        allowed = {title.casefold() for title in allowed_titles}
        failures = tuple(
            f'mentions a track not in the evidence: "{quoted}"'
            for quoted in _QUOTED.findall(text)
            if quoted.casefold() not in allowed
        )
        return EvaluationReport(ok=not failures, failures=failures)
