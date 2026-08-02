"""Input and privacy guard for natural-language queries.

The first thing a typed sentence meets. It rejects oversized input; redacts
secrets and direct identifiers so they never reach retrieval, the provider, or
logs; strips prompt-injection directives (user text is data, never instruction);
and routes clear crisis language to a safe, non-clinical response.

Deterministic and offline — the patterns are compiled regexes you can read and
audit. It is a coarse safety net, not a perfect classifier, so it errs toward
redacting or routing locally.
"""

from __future__ import annotations

import re

from src.contracts import GuardCategory, GuardVerdict


MAX_QUERY_CHARS = 400
REDACTION = "[redacted]"

# PII / secrets — matches are replaced with the redaction marker so raw values
# never leave the guard. Over-matching only means extra redaction (the safe side).
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),  # email
    re.compile(r"\b(?:\+?\d[\s().\-]?){9,}\d\b"),  # long phone-like digit run
    re.compile(r"\b(?:\d[ \-]?){13,16}\b"),  # card-like
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-like
    re.compile(r"\b(?:AIza[0-9A-Za-z\-_]{10,}|sk-[0-9A-Za-z\-_]{10,})\b"),  # known key prefixes
    # high-entropy token: 20+ chars containing at least one letter and one digit
    re.compile(r"\b(?=[A-Za-z0-9_\-.]*[A-Za-z])(?=[A-Za-z0-9_\-.]*\d)[A-Za-z0-9_\-.]{20,}\b"),
)

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore (all |any )?(the )?(previous|prior|above)[^.?!]*",
        r"disregard[^.?!]*instructions[^.?!]*",
        r"forget (your|all|the)[^.?!]*instructions[^.?!]*",
        r"you are now[^.?!]*",
        r"reveal[^.?!]*prompt[^.?!]*",
        r"(your |the )?system prompt[^.?!]*",
        r"developer mode[^.?!]*",
    )
)

# Conservative crisis/self-harm cues. A match routes to a gentle, non-clinical
# safe response; a rare false positive is an acceptable cost for caution.
_HIGH_RISK: tuple[str, ...] = (
    "kill myself", "killing myself", "want to die", "end my life", "end it all",
    "suicide", "suicidal", "hurt myself", "harm myself", "self harm", "self-harm",
)


def _redact_pii(text: str) -> tuple[str, bool]:
    """Replace any PII/secret spans with the redaction marker."""
    redacted = text
    found = False
    for pattern in _PII_PATTERNS:
        redacted, count = pattern.subn(REDACTION, redacted)
        found = found or count > 0
    return redacted, found


def _strip_injection(text: str) -> tuple[str, bool]:
    """Remove any prompt-injection directives, keeping the rest as data."""
    stripped = text
    found = False
    for pattern in _INJECTION_PATTERNS:
        stripped, count = pattern.subn(" ", stripped)
        found = found or count > 0
    return stripped, found


def _collapse(text: str) -> str:
    return " ".join(text.split())


class InputGuard:
    """Classify and sanitize one raw query before it reaches the system."""

    def __init__(self, max_chars: int = MAX_QUERY_CHARS) -> None:
        self._max_chars = max_chars

    def inspect(self, text: str) -> GuardVerdict:
        """Return text safe for routed processing—not for persistence or URLs."""
        if not text or not text.strip():
            return GuardVerdict(category=GuardCategory.EMPTY, reason="empty query")
        if len(text) > self._max_chars:
            return GuardVerdict(
                category=GuardCategory.TOO_LONG,
                reason=f"query exceeds {self._max_chars} characters",
            )

        lowered = text.lower()
        if any(phrase in lowered for phrase in _HIGH_RISK):
            return GuardVerdict(
                category=GuardCategory.HIGH_RISK,
                reason="possible crisis language; routed to a safe response",
            )

        redacted, has_pii = _redact_pii(text)
        cleaned, has_injection = _strip_injection(redacted)
        sanitized = _collapse(cleaned)

        # PII takes priority: a query carrying personal data is kept local, and
        # any injection in it is stripped too.
        if has_pii:
            return GuardVerdict(
                category=GuardCategory.SENSITIVE,
                sanitized_query=sanitized,
                reason="redacted personal or secret information; kept local",
            )
        if has_injection:
            return GuardVerdict(
                category=GuardCategory.INJECTION,
                sanitized_query=sanitized,
                reason="ignored an instruction embedded in the query",
            )
        return GuardVerdict(category=GuardCategory.OK, sanitized_query=sanitized)
