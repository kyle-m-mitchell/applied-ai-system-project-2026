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

    @field_validator("genre", "mood", mode="after")
    @classmethod
    def normalize_category(cls, value: str) -> str:
        """Store matching categories in one canonical form."""
        return value.lower()

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
