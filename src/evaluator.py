"""Deterministic grounding and presentation guardrails.

``evaluate_result`` checks the authoritative recommendation payload: real IDs,
no duplicates, hard constraints, evidence, and count.

``check_grounded_text`` checks optional model-written *framing*. The model never
chooses tracks, and it is not allowed to narrate track facts: verified song facts
belong to the deterministic cards. Prose that describes the music, violates
Cadence's contract, leaks markup, makes persona claims, or adds calls to action is
discarded and the deterministic template is shown instead.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

from src.contracts import EvaluationReport, MusicIntent, RetrievalHit


_URL_OR_MARKUP = re.compile(
    r"(?:https?://|www\.|<[^>]*>|\[[^\]]+\]\([^)]*\)|`)", re.IGNORECASE
)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_QUOTES = re.compile(r"[\"“”]")
_TERMINALS = re.compile(r"[.!?]+")
_FORBIDDEN_PERSONA = (
    re.compile(r"\b(?:i am|i'm)\s+(?:an?\s+)?(?:human|person|doctor|therapist|ai|model)\b", re.I),
    re.compile(r"\bi\s+(?:have\s+)?(?:heard|listened(?:\s+to)?|experienced|remember|feel|love|hate)\b", re.I),
    re.compile(r"\b(?:diagnos(?:e|is)|medical advice|treatment plan|crisis counselor)\b", re.I),
    re.compile(r"\bas\s+(?:an?\s+)?human\b", re.I),
)
_TITLE_LIKE = re.compile(
    r"\b(?:[A-Z][a-z0-9'’-]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z0-9'’-]+|[A-Z]{2,})){1,}\b"
)
_TRACK_DIRECTIVE = re.compile(
    r"\b(?:try|play|hear|listen to|start with)\b", re.IGNORECASE
)
_UNSAFE_FRAMING = re.compile(
    r"\b(?:password|passcode|credit card|social security|api key|private key|"
    r"send me|enter your|stop taking|medication|medical advice|cures? (?:depression|anxiety)|"
    r"hurt yourself|kill yourself|end your life|suicide|go hurt|visit this site)\b",
    re.IGNORECASE,
)
# Cadence's optional model selects only the social bridge around a result. It may
# not describe the result itself. This deliberately conservative boundary is
# easier to verify than pretending a keyword checker can fact-check arbitrary
# prose. Facts such as tempo, genre, vocals, or energy are rendered from validated
# catalog fields elsewhere in the application.
_TRACK_FACT_LANGUAGE = re.compile(
    r"\b(?:"
    r"slow(?:er)?|fast(?:er)?|up[ -]?tempo|down[ -]?tempo|tempo|bpm|"
    r"acoustic|unplugged|electronic|synthetic|organic|"
    r"instrumentals?|vocals?|lyrics?|lyrical|sung|wordless|"
    r"genres?|jazz|rock|pop|hip[ -]?hop|classical|country|blues|reggae|"
    r"folk|metal|punk|soul|funk|r\s*&\s*b|lo[ -]?fi|ambient|edm|"
    r"danceable|dancefloor|rhythmic|groov(?:e|y)|bouncy|"
    r"energetic|energy|high[ -]?energy|low[ -]?energy|"
    r"calm|mellow|soft|loud|quiet|hushed|intense|driving|punchy|gentle|"
    r"soothing|relaxing|dreamy|reflective|bright|dark|upbeat|moody|"
    r"melanchol(?:y|ic)|happy|sad|warm|cool|"
    r"clean|explicit|eras?|decades?"
    r")\b",
    re.IGNORECASE,
)
MAX_FRAMING_CHARS = 280
MAX_FRAMING_WORDS = 45

# The language model is a bounded selector, not an unconstrained copywriter.
# Every publishable line is application-owned, fact-free microcopy. Exact membership
# closes the open-world hole that any keyword denylist would leave (release dates,
# artist nationalities, awards, durations, and infinitely many other claims).
APPROVED_FRAMINGS: tuple[str, ...] = (
    "Here's a thoughtfully chosen set for the moment you described.",
    "I found a few picks worth meeting right where you are.",
    "A fresh set is ready whenever you are.",
    "Let's start here, then shape the next set together.",
    "Consider this a first pass; we can tune it from here.",
    "Here are a few directions worth exploring together.",
    "I pulled together a small set for this particular moment.",
    "Let's give this moment a soundtrack and see where it takes you.",
)


class GroundingEvaluator:
    """Validate retrieval results and model-selected framing against explicit rules."""

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
        if intent.instrumental_only and any(
            hit.track.instrumental is not True for hit in hits
        ):
            failures.append("instrumental-only constraint violated")
        if intent.exclude_explicit and any(
            hit.track.explicit is not False for hit in hits
        ):
            failures.append("clean constraint violated")
        if any(not self._fields_are_present(hit) for hit in hits):
            failures.append("a hit claims unavailable evidence fields")
        if any(not self._has_evidence(hit) for hit in hits):
            failures.append("a hit carries no evidence")
        if len(hits) > intent.limit:
            failures.append("more hits than requested")

        return EvaluationReport(ok=not failures, failures=tuple(failures))

    @staticmethod
    def _fields_are_present(hit: RetrievalHit) -> bool:
        """Reject provenance that names absent or unknown-valued track fields."""
        for field in hit.fields_used:
            if not hasattr(hit.track, field):
                return False
            value = getattr(hit.track, field)
            if value is None:
                return False
            if isinstance(value, (tuple, list, set, dict, str)) and not value:
                return False
        return True

    @staticmethod
    def _has_evidence(hit: RetrievalHit) -> bool:
        """A hit is grounded when at least one ranking leg supplies evidence."""
        return (
            bool(hit.matched_terms)
            or (hit.semantic_score is not None and hit.semantic_score > 0.0)
            or (hit.structured_score is not None and hit.structured_score > 0.0)
        )

    def check_grounded_text(
        self,
        text: str,
        evidence_names: Collection[str],
    ) -> EvaluationReport:
        """Enforce Cadence's small, deterministic generated-framing contract.

        Framing is one short social-bridge sentence with no music descriptions,
        track/artist names, quotation marks, links, markup, control characters,
        human/listening claims, or medical-role language. The validated
        application renders track facts separately.
        """
        stripped = text.strip()
        failures: list[str] = []
        if not stripped:
            return EvaluationReport(ok=False, failures=("empty message",))
        if stripped not in APPROVED_FRAMINGS:
            failures.append("framing is not an approved bounded line")
        if len(stripped) > MAX_FRAMING_CHARS or len(stripped.split()) > MAX_FRAMING_WORDS:
            failures.append("framing exceeds the bounded length")
        if "\n" in stripped or "\r" in stripped or len(_TERMINALS.findall(stripped)) > 1:
            failures.append("framing must be one sentence")
        if _URL_OR_MARKUP.search(stripped):
            failures.append("framing contains a URL or markup")
        if _CONTROL.search(stripped):
            failures.append("framing contains control characters")
        if _QUOTES.search(stripped):
            failures.append("framing contains quotation marks")
        if any(pattern.search(stripped) for pattern in _FORBIDDEN_PERSONA):
            failures.append("framing violates the Cadence persona boundary")
        if _TITLE_LIKE.search(stripped) or _TRACK_DIRECTIVE.search(stripped):
            failures.append("framing may be naming or directing a specific track")
        if _UNSAFE_FRAMING.search(stripped):
            failures.append("framing contains unsafe or credential-seeking language")
        if _TRACK_FACT_LANGUAGE.search(stripped):
            failures.append("framing attempts to describe track facts")

        folded = stripped.casefold()
        mentioned = sorted(
            {
                name
                for name in evidence_names
                if len(name.strip()) >= 3 and name.casefold() in folded
            },
            key=str.casefold,
        )
        if mentioned:
            failures.append("framing names a track or artist reserved for the app")
        return EvaluationReport(ok=not failures, failures=tuple(failures))
