"""Shared application service for validated music recommendations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.contracts import (
    CatalogTrack,
    OperatingMode,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResult,
)
from src.recommender import WEIGHTS, recommend_songs


REQUEST_TO_SCORER_FIELD = {
    "genre": "genre",
    "mood": "mood",
    "energy": "energy",
    "acousticness": "acousticness",
    "valence": "valence",
    "danceability": "danceability",
    "tempo_bpm": "tempo",
}


class RecommendationService:
    """Validate input and expose one reusable recommendation entry point.

    The service delegates ranking to the original ``recommend_songs`` function.
    It adds contracts and normalization without reimplementing the scoring or
    sorting algorithm, so existing behavior remains our trusted baseline.
    """

    def __init__(self, catalog: Sequence[Mapping[str, Any]]) -> None:
        if not catalog:
            raise ValueError("catalog must contain at least one track")

        validated_catalog = tuple(
            CatalogTrack.model_validate(dict(track)) for track in catalog
        )
        track_ids = [track.id for track in validated_catalog]
        duplicate_ids = sorted(
            track_id for track_id in set(track_ids) if track_ids.count(track_id) > 1
        )
        if duplicate_ids:
            formatted_ids = ", ".join(str(track_id) for track_id in duplicate_ids)
            raise ValueError(f"catalog contains duplicate track IDs: {formatted_ids}")

        self._catalog = validated_catalog
        self._tracks_by_id = {track.id: track for track in validated_catalog}

    @property
    def catalog(self) -> tuple[CatalogTrack, ...]:
        """Return the immutable, validated catalog."""
        return self._catalog

    def recommend(
        self,
        request: RecommendationRequest,
    ) -> RecommendationResult:
        """Return validated local recommendations for one validated request."""
        if not isinstance(request, RecommendationRequest):
            request = RecommendationRequest.model_validate(request)

        preferences = self._legacy_preferences(request)
        max_possible_score = self._active_max_score(request)

        # Convert immutable models into fresh dictionaries so legacy code cannot
        # mutate the service's validated catalog.
        legacy_catalog = [track.model_dump() for track in self._catalog]
        ranked = recommend_songs(preferences, legacy_catalog, k=request.limit)

        recommendations = tuple(
            RecommendationItem(
                track=self._tracks_by_id[song["id"]],
                raw_score=raw_score,
                match_strength=self._match_strength(raw_score, max_possible_score),
                reasons=self._split_reasons(explanation),
            )
            for song, raw_score, explanation in ranked
        )

        warnings: tuple[str, ...] = ()
        if len(recommendations) < request.limit:
            warnings = (
                f"requested {request.limit} tracks, but the catalog contains "
                f"only {len(recommendations)}",
            )

        return RecommendationResult(
            request=request,
            recommendations=recommendations,
            max_possible_score=max_possible_score,
            operating_mode=OperatingMode.LOCAL,
            warnings=warnings,
        )

    @staticmethod
    def _legacy_preferences(request: RecommendationRequest) -> dict[str, Any]:
        """Adapt public request names to the original scorer's dictionary API."""
        preferences: dict[str, Any] = {}
        for request_field, scorer_field in REQUEST_TO_SCORER_FIELD.items():
            value = getattr(request, request_field)
            if value is not None:
                preferences[scorer_field] = value
        return preferences

    @staticmethod
    def _active_max_score(request: RecommendationRequest) -> float:
        """Return the best possible score for the preferences actually supplied."""
        return sum(
            WEIGHTS[scorer_field]
            for request_field, scorer_field in REQUEST_TO_SCORER_FIELD.items()
            if getattr(request, request_field) is not None
        )

    @staticmethod
    def _match_strength(raw_score: float, max_possible_score: float) -> float:
        """Normalize a request-specific score without calling it probability."""
        normalized = raw_score / max_possible_score
        return min(1.0, max(0.0, normalized))

    @staticmethod
    def _split_reasons(explanation: str) -> tuple[str, ...]:
        """Turn the legacy display string into structured reason values."""
        return tuple(
            reason.strip()
            for reason in explanation.split(";")
            if reason.strip()
        )
