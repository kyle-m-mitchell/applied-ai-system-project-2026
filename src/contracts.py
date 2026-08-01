"""Validated data contracts shared by every application interface.

Type hints describe what developers intend. Pydantic models also enforce that
intent at runtime, which makes these contracts our first reliability layer.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContractModel(BaseModel):
    """Strict, immutable defaults shared by the project's public contracts."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class OperatingMode(str, Enum):
    """How the recommendation was produced."""

    LOCAL = "local"
    GEMINI = "gemini"
    DEGRADED = "degraded"


class RecommendationRequest(ContractModel):
    """Structured listener preferences accepted by the deterministic scorer.

    This request stays structured-only by design. Natural-language input is
    handled separately by ``MusicCompanion`` (guard → intent parser → retrieval),
    so the trusted scorer path never has to interpret free text.
    """

    PREFERENCE_FIELDS: ClassVar[tuple[str, ...]] = (
        "genre",
        "mood",
        "energy",
        "acousticness",
        "valence",
        "danceability",
        "tempo_bpm",
    )
    NUMERIC_FIELDS: ClassVar[tuple[str, ...]] = (
        "energy",
        "acousticness",
        "valence",
        "danceability",
        "tempo_bpm",
    )

    genre: str | None = Field(default=None, min_length=1, max_length=80)
    mood: str | None = Field(default=None, min_length=1, max_length=80)
    energy: float | None = Field(default=None, ge=0.0, le=1.0)
    acousticness: float | None = Field(default=None, ge=0.0, le=1.0)
    valence: float | None = Field(default=None, ge=0.0, le=1.0)
    danceability: float | None = Field(default=None, ge=0.0, le=1.0)
    tempo_bpm: float | None = Field(default=None, ge=50.0, le=200.0)
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("genre", "mood", mode="after")
    @classmethod
    def normalize_category(cls, value: str | None) -> str | None:
        """Normalize categories so matching is stable across interfaces."""
        return value.lower() if value is not None else None

    @field_validator(*NUMERIC_FIELDS, mode="before")
    @classmethod
    def reject_boolean_numbers(cls, value: object) -> object:
        """Reject booleans instead of silently treating True/False as 1/0."""
        if isinstance(value, bool):
            raise ValueError("boolean values are not valid numeric preferences")
        return value

    @field_validator("limit", mode="before")
    @classmethod
    def reject_boolean_limit(cls, value: object) -> object:
        """Reject True as a request for one recommendation."""
        if isinstance(value, bool):
            raise ValueError("limit must be an integer, not a boolean")
        return value

    @model_validator(mode="after")
    def require_preference(self) -> Self:
        """An all-empty request cannot produce a meaningful recommendation."""
        if not any(getattr(self, field) is not None for field in self.PREFERENCE_FIELDS):
            raise ValueError("provide at least one music preference")
        return self


class CatalogTrack(ContractModel):
    """Validated form of one authoritative catalog record."""

    NUMERIC_FIELDS: ClassVar[tuple[str, ...]] = (
        "energy",
        "tempo_bpm",
        "valence",
        "danceability",
        "acousticness",
    )

    id: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=200)
    artist: str = Field(min_length=1, max_length=200)
    genre: str = Field(min_length=1, max_length=80)
    mood: str = Field(min_length=1, max_length=80)
    energy: float = Field(ge=0.0, le=1.0)
    tempo_bpm: float = Field(ge=50.0, le=200.0)
    valence: float = Field(ge=0.0, le=1.0)
    danceability: float = Field(ge=0.0, le=1.0)
    acousticness: float = Field(ge=0.0, le=1.0)
    description: str = Field(min_length=20, max_length=500)
    tags: tuple[str, ...] = Field(min_length=2, max_length=12)
    contexts: tuple[str, ...] = Field(min_length=2, max_length=12)
    instruments: tuple[str, ...] = Field(min_length=1, max_length=12)
    instrumental: bool
    explicit: bool
    era: str = Field(pattern=r"^(?:19|20)\d0s$")

    @field_validator("genre", "mood", mode="after")
    @classmethod
    def normalize_category(cls, value: str) -> str:
        """Store matching categories in one canonical form."""
        return value.lower()

    @field_validator("tags", "contexts", "instruments", mode="before")
    @classmethod
    def normalize_metadata_values(cls, value: object) -> tuple[str, ...]:
        """Require nonempty, unique metadata terms in a canonical form."""
        if isinstance(value, (str, bytes)) or value is None:
            raise ValueError("metadata collections must be a sequence of strings")

        try:
            raw_values = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(
                "metadata collections must be a sequence of strings"
            ) from exc

        normalized: list[str] = []
        for item in raw_values:
            if not isinstance(item, str):
                raise ValueError("metadata collection items must be strings")
            term = " ".join(item.split()).lower()
            if not term:
                raise ValueError("metadata collection items cannot be empty")
            if len(term) > 80:
                raise ValueError("metadata collection items cannot exceed 80 characters")
            normalized.append(term)

        if len(normalized) != len(set(normalized)):
            raise ValueError("metadata collections cannot contain duplicate values")
        return tuple(normalized)

    @field_validator("instrumental", "explicit", mode="before")
    @classmethod
    def require_real_booleans(cls, value: object) -> object:
        """Reject truthy strings and integers at the validated service boundary."""
        if not isinstance(value, bool):
            raise ValueError("catalog boolean fields must be true booleans")
        return value

    @field_validator("id", *NUMERIC_FIELDS, mode="before")
    @classmethod
    def reject_boolean_numbers(cls, value: object) -> object:
        """Catalog data must not coerce booleans into numeric attributes."""
        if isinstance(value, bool):
            raise ValueError("boolean values are not valid catalog numbers")
        return value


class RecommendationItem(ContractModel):
    """One ranked track and the evidence produced by the local scorer."""

    track: CatalogTrack
    raw_score: float = Field(ge=0.0)
    match_strength: float = Field(ge=0.0, le=1.0)
    reasons: tuple[str, ...] = ()


class RecommendationResult(ContractModel):
    """Validated response returned by ``RecommendationService``."""

    request: RecommendationRequest
    recommendations: tuple[RecommendationItem, ...]
    max_possible_score: float = Field(gt=0.0)
    operating_mode: OperatingMode
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def enforce_result_invariants(self) -> Self:
        """Prevent oversized results and duplicate catalog IDs."""
        if len(self.recommendations) > self.request.limit:
            raise ValueError("recommendation count exceeds the requested limit")

        track_ids = [item.track.id for item in self.recommendations]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("recommendations contain duplicate track IDs")
        return self


class SourceType(str, Enum):
    """Where a retrieved piece of evidence came from.

    ``CONTEXT_GUIDE`` is defined now so the retrieval interface stays stable, but
    it is unused until curated context guides are added as a second source.
    """

    CATALOG = "catalog"
    CONTEXT_GUIDE = "context_guide"


class ContextGuide(ContractModel):
    """One curated, human-written guide about a listening situation.

    Guides are a second retrieval source. They are not recommendable tracks;
    they connect a listener's words to catalog vocabulary and provide grounded
    context for an explanation.
    """

    guide_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=20, max_length=2000)

    def index_text(self) -> str:
        """Return the text indexed for retrieval (title plus body)."""
        return f"{self.title} {self.body}"


class RetrievalHit(ContractModel):
    """One track surfaced by the retriever, with the provenance that justifies it.

    ``score`` is a cosine similarity in ``[0, 1]``, not a probability or a
    calibrated confidence. Every hit records where it came from so a later
    evaluator (and a human) can trace why it was retrieved.
    """

    source_type: SourceType
    source_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    fields_used: tuple[str, ...] = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    matched_terms: tuple[str, ...] = ()
    semantic_score: float | None = Field(default=None, ge=0.0, le=1.0)
    lexical_score: float | None = Field(default=None, ge=0.0, le=1.0)
    track: CatalogTrack


class GuideEvidence(ContractModel):
    """A context guide that informed a retrieval, kept as cited evidence.

    ``expansion_terms`` are the catalog-vocabulary terms this guide contributed
    to the track query, which is how a guide improves retrieval without ever
    being recommended itself.
    """

    source_type: SourceType
    source_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    title: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    matched_terms: tuple[str, ...] = ()
    expansion_terms: tuple[str, ...] = ()


class RetrievalResult(ContractModel):
    """Validated response returned by a ``Retriever``.

    ``index_fingerprint`` identifies the exact index the hits came from, so a
    result can be tied back to a specific catalog content hash and retrieval
    method version. ``guides_used`` and ``expanded_query_terms`` record how a
    second source (context guides) shaped this result; both are empty when no
    guide fired.
    """

    query: str
    hits: tuple[RetrievalHit, ...]
    index_fingerprint: str = Field(min_length=1)
    filters_applied: tuple[str, ...] = ()
    guides_used: tuple[GuideEvidence, ...] = ()
    expanded_query_terms: tuple[str, ...] = ()
    operating_mode: OperatingMode = OperatingMode.LOCAL


class GuardCategory(str, Enum):
    """How the input/privacy guard classified a raw natural-language query."""

    OK = "ok"
    EMPTY = "empty"
    TOO_LONG = "too_long"
    SENSITIVE = "sensitive"  # contained PII or a secret (now redacted)
    INJECTION = "injection"  # contained a prompt-injection directive (now stripped)
    HIGH_RISK = "high_risk"  # crisis/self-harm; routed to a safe response


class GuardVerdict(ContractModel):
    """The guard's decision about one raw query.

    ``sanitized_query`` is always safe to retrieve on and to log: PII/secret
    spans are replaced with ``[redacted]`` and injection directives are stripped,
    so raw sensitive text never reaches retrieval, the provider, or logs.
    """

    category: GuardCategory
    sanitized_query: str = ""
    reason: str = ""


class MusicIntent(ContractModel):
    """Structured intent parsed from a guarded query.

    ``query`` is the sanitized text handed to the retriever; the categorical and
    filter fields are extracted deterministically from recognizable music words.
    """

    query: str = ""
    genre: str | None = Field(default=None, max_length=80)
    mood: str | None = Field(default=None, max_length=80)
    instrumental_only: bool = False
    exclude_explicit: bool = False
    limit: int = Field(default=5, ge=1, le=20)
    needs_clarification: bool = False
    clarification: str | None = None
    source: str = "rules"


class CompanionAction(str, Enum):
    """The bounded set of actions the companion may take for one query."""

    RECOMMEND = "recommend"
    CLARIFY = "clarify"
    NO_MATCH = "no_match"
    SAFE_RESPONSE = "safe_response"
    DEGRADED = "degraded"


class VoiceSource(str, Enum):
    """Which renderer produced the companion's message."""

    TEMPLATE = "template"  # deterministic, reproducible, always available
    GENERATED = "generated"  # produced by a text generator (fake or provider)


class EvaluationReport(ContractModel):
    """The grounding evaluator's verdict on a result or a rendered message."""

    ok: bool
    failures: tuple[str, ...] = ()


class AgentTrace(ContractModel):
    """A structured, privacy-safe record of one bounded-agent turn.

    It captures categories, ids, and decisions — never raw sensitive text — so a
    reviewer can see how the companion reached its answer. ``evaluation`` is the
    result check; ``text_evaluation`` is the grounding check on generated text (if
    a generator ran); ``voice_model`` names the generator that produced the voice.
    """

    guard_category: GuardCategory
    intent_summary: str = ""
    retrieved_ids: tuple[int, ...] = ()
    diversity_applied: bool = False
    evaluation: EvaluationReport = EvaluationReport(ok=True)
    text_evaluation: EvaluationReport | None = None
    action: CompanionAction
    voice_source: VoiceSource = VoiceSource.TEMPLATE
    voice_model: str | None = None
    fallback_reason: str | None = None


class CompanionResponse(ContractModel):
    """Validated response from the natural-language companion.

    ``retrieval`` reuses the retriever's own result (hits, provenance, operating
    mode, guide evidence); it is ``None`` for clarify/safe/empty outcomes.
    ``trace`` is the bounded-agent record of how the answer was produced.
    """

    action: CompanionAction
    message: str
    retrieval: RetrievalResult | None = None
    intent: MusicIntent | None = None
    trace: AgentTrace | None = None

    @model_validator(mode="after")
    def enforce_action_invariants(self) -> Self:
        """Keep action and payload consistent (e.g. recommend must have hits)."""
        if self.action in (CompanionAction.RECOMMEND, CompanionAction.DEGRADED):
            if self.retrieval is None or not self.retrieval.hits:
                raise ValueError(f"{self.action.value} response must include hits")
        if self.action is CompanionAction.NO_MATCH and self.retrieval is None:
            raise ValueError("no_match response must include the (empty) retrieval result")
        if self.action in (CompanionAction.CLARIFY, CompanionAction.SAFE_RESPONSE):
            if self.retrieval is not None:
                raise ValueError(f"{self.action.value} response must not include retrieval")
        return self
