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
    """Structured listener preferences accepted by the current recommender.

    Natural-language ``query`` input will be added with the intent-parsing
    feature. Accepting it before the application can interpret it would create
    a misleading API.
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
    track: CatalogTrack


class RetrievalResult(ContractModel):
    """Validated response returned by a ``Retriever``.

    ``index_fingerprint`` identifies the exact index the hits came from, so a
    result can be tied back to a specific catalog content hash and retrieval
    method version.
    """

    query: str
    hits: tuple[RetrievalHit, ...]
    index_fingerprint: str = Field(min_length=1)
    filters_applied: tuple[str, ...] = ()
