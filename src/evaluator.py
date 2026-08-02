"""Grounding evaluator: the guardrail before the companion speaks.

``evaluate_result`` is the strong check: it confirms a result is trustworthy —
real ids, no duplicates, hard constraints held, evidence present, within the
requested count.

``check_grounded_text`` is a narrower, defense-in-depth check on *generated
framing only*. It does not parse arbitrary claims; it flags a **quoted** title
that is not in the evidence. That is sufficient here because the authoritative
track list is always rendered deterministically by the system, so a generator
can never change *which* tracks are recommended — only its framing prose is at
risk, and a quoted invented title is the most likely way that shows up.
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
        if any(not self._has_evidence(hit) for hit in hits):
            failures.append("a hit carries no evidence")
        if len(hits) > intent.limit:
            failures.append("more hits than requested")

        return EvaluationReport(ok=not failures, failures=tuple(failures))

    @staticmethod
    def _has_evidence(hit: RetrievalHit) -> bool:
        """A hit is grounded if any leg justifies it: matched terms, a semantic
        similarity, or a positive structured relevance. A structured ``0.0`` is
        "evaluated, no match" and is *not* evidence; ``None`` is "not evaluated"."""
        return (
            bool(hit.matched_terms)
            or hit.semantic_score is not None
            or (hit.structured_score is not None and hit.structured_score > 0.0)
        )

    def check_grounded_text(
        self,
        text: str,
        allowed_titles: Collection[str],
    ) -> EvaluationReport:
        """Reject framing that quotes a title/name not in the evidence.

        Narrow by design (see the module docstring): it catches quoted names, not
        every conceivable claim — the track facts themselves are never the
        generator's to produce.
        """
        if not text or not text.strip():
            return EvaluationReport(ok=False, failures=("empty message",))
        allowed = {title.casefold() for title in allowed_titles}
        failures = tuple(
            f'mentions a track not in the evidence: "{quoted}"'
            for quoted in _QUOTED.findall(text)
            if quoted.casefold() not in allowed
        )
        return EvaluationReport(ok=not failures, failures=failures)
