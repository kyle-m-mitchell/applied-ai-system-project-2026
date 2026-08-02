"""Privacy-safe observability: a receipt, not a diary.

An :class:`EventSink` records companion *decisions and identifiers* — never the
words a person typed, the prompt sent to a provider, or any persistent user id.
:class:`~src.contracts.CompanionEvent` defines the exact allowlist. Sinks are
best-effort: a logging failure must never change or break a response, so the
companion wraps every ``record`` call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from src.contracts import (
    CompanionEvent,
    CompanionResponse,
    GuardCategory,
    RankedCandidate,
)


class EventSink(ABC):
    """Where companion receipts go. Implementations must be side-effect-safe."""

    @abstractmethod
    def record(self, event: CompanionEvent) -> None:
        """Persist one receipt."""


class NullEventSink(EventSink):
    """Discards events. The default — observability is opt-in."""

    def record(self, event: CompanionEvent) -> None:
        return None


class JsonlEventSink(EventSink):
    """Append one JSON object per line to a file (JSON Lines).

    Each receipt is a self-contained line, so a log can be streamed, tailed, or
    loaded row-by-row without parsing the whole file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def record(self, event: CompanionEvent) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def build_event(
    *,
    request_id: str,
    response: CompanionResponse,
    candidates: Sequence[RankedCandidate],
    candidate_ids: Sequence[int],
    latency_ms: float,
    config_fingerprint: str | None = None,
) -> CompanionEvent:
    """Assemble a privacy-safe receipt from a finished response.

    ``candidates`` describe the *returned* tracks (parallel to ``final_ids``);
    their ``reasons`` — which carry query-derived matched terms — are stripped so
    the receipt keeps only numeric scores and ids. Nothing assembled here contains
    query text, prompts, or a persistent identity.
    """
    trace = response.trace
    retrieval = response.retrieval
    final_ids = tuple(candidate.track.id for candidate in candidates)
    stripped = tuple(
        candidate.components.model_copy(update={"reasons": ()}) for candidate in candidates
    )
    return CompanionEvent(
        request_id=request_id,
        timestamp=_utc_now(),
        guard_category=trace.guard_category if trace else GuardCategory.OK,
        action=response.action,
        intent_summary=trace.intent_summary if trace else "",
        operating_mode=retrieval.operating_mode if retrieval else None,
        voice_source=trace.voice_source if trace else None,
        candidate_ids=tuple(candidate_ids),
        final_ids=final_ids,
        components=stripped,
        fallback_reason=trace.fallback_reason if trace else None,
        latency_ms=latency_ms,
        index_fingerprint=retrieval.index_fingerprint if retrieval else None,
        config_fingerprint=config_fingerprint,
    )
