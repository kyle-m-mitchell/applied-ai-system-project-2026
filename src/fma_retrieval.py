"""Existing-retriever-compatible adapter over the lazy FMA SQLite store."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable, Sequence

from src.contracts import (
    CatalogTrack,
    MusicIntent,
    OperatingMode,
    RetrievalHit,
    RetrievalResult,
    SourceType,
)
from src.etl.integrity import sha256_file
from src.fma_store import CatalogStoreHit, FmaCatalogStore
from src.retrieval import Retriever


TEXT_POOL_SIZE = 200
STRUCTURED_POOL_SIZE = 200
RRF_K = 60
W_TEXT = 0.4
W_STRUCTURED = 0.6
METHOD_ID = "fma-sqlite-fts5-structured-rrf-v1"
FUSION_VERSION = f"rrf:text={W_TEXT:g},structured={W_STRUCTURED:g},k={RRF_K};fma-v1"
_TERM = re.compile(r"[^\W_]+", re.UNICODE)


def _rank_map(hits: Sequence[CatalogStoreHit]) -> dict[int, int]:
    return {hit.track_id: rank for rank, hit in enumerate(hits, start=1)}


def _indexed_fields(track: CatalogTrack) -> tuple[str, ...]:
    fields: list[str] = []
    for name in (
        "title",
        "artist",
        "genres",
        "tags",
        "album_tags",
        "artist_tags",
        "track_information",
        "album_information",
        "artist_biography",
    ):
        value = getattr(track, name)
        if value is None or value == () or value == "":
            continue
        fields.append(name)
    # Deterministic feature terms in the FTS document are derived solely from
    # these evidenced values. Naming the values, rather than a hidden generated
    # string, keeps provenance inspectable.
    fields.extend(
        name
        for name in (
            "energy", "valence", "acousticness", "danceability",
            "tempo_bpm", "instrumentalness", "mood_profile",
        )
        if getattr(track, name) is not None
    )
    return tuple(fields)


def _content_hash(track: CatalogTrack, fields: Sequence[str]) -> str:
    payload: dict[str, object] = {}
    for field in fields:
        value = getattr(track, field)
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        payload[field] = value
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=list
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _matched_terms(query: str, track: CatalogTrack) -> tuple[str, ...]:
    query_terms = {
        match.group(0).casefold() for match in _TERM.finditer(query) if len(match.group(0)) >= 2
    }
    values: list[str] = [track.title, track.artist]
    for name in (
        "genres", "tags", "album_tags", "artist_tags", "track_information",
        "album_information", "artist_biography",
    ):
        value = getattr(track, name)
        if isinstance(value, tuple):
            values.extend(value)
        elif isinstance(value, str):
            values.append(value)
    document_terms = {
        match.group(0).casefold()
        for match in _TERM.finditer(" ".join(values))
        if len(match.group(0)) >= 2
    }
    return tuple(sorted(query_terms & document_terms))


class FmaRetriever(Retriever):
    """Lazy FTS/structured retriever with independent top-200 candidate legs."""

    def __init__(self, store: FmaCatalogStore) -> None:
        self.store = store
        artifact_hash = (
            store.manifest.artifact_sha256
            if store.manifest is not None
            else sha256_file(store.database_path)
        )
        self._fingerprint = hashlib.sha256(
            f"{METHOD_ID}:{artifact_hash}".encode("ascii")
        ).hexdigest()

    @property
    def index_fingerprint(self) -> str:
        return self._fingerprint

    def _empty(
        self, query: str, *, instrumental_only: bool, exclude_explicit: bool
    ) -> RetrievalResult:
        filters: list[str] = []
        if instrumental_only:
            filters.append("instrumental_only:unsupported_unknown")
        if exclude_explicit:
            filters.append("exclude_explicit:unsupported_unknown")
        return RetrievalResult(
            query=query,
            hits=(),
            index_fingerprint=self._fingerprint,
            filters_applied=tuple(filters),
            operating_mode=OperatingMode.LOCAL,
        )

    @staticmethod
    def _rrf(
        text_hits: Sequence[CatalogStoreHit], structured_hits: Sequence[CatalogStoreHit]
    ) -> tuple[tuple[int, float], ...]:
        text_rank = _rank_map(text_hits)
        structured_rank = _rank_map(structured_hits)
        active_text = bool(text_hits)
        active_structured = bool(structured_hits)
        active_weight = (W_TEXT if active_text else 0.0) + (
            W_STRUCTURED if active_structured else 0.0
        )
        if active_weight == 0.0:
            return ()
        maximum = active_weight / (RRF_K + 1)
        scores: dict[int, float] = {}
        for track_id in set(text_rank) | set(structured_rank):
            score = 0.0
            if track_id in text_rank:
                score += W_TEXT / (RRF_K + text_rank[track_id])
            if track_id in structured_rank:
                score += W_STRUCTURED / (RRF_K + structured_rank[track_id])
            scores[track_id] = min(1.0, score / maximum)
        return tuple(sorted(scores.items(), key=lambda pair: (-pair[1], pair[0])))

    def _result(
        self,
        query: str,
        ranked: Sequence[tuple[int, float]],
        *,
        text_hits: Sequence[CatalogStoreHit],
        structured_hits: Sequence[CatalogStoreHit],
        k: int,
        fusion: bool,
    ) -> RetrievalResult:
        chosen = tuple(track_id for track_id, _score in ranked[:k])
        tracks = self.store.get_contract_tracks(chosen)
        by_id = {track.id: track for track in tracks}
        text_scores = {hit.track_id: hit.score for hit in text_hits}
        structured_scores = {hit.track_id: hit.score for hit in structured_hits}
        scores = dict(ranked)
        hits: list[RetrievalHit] = []
        for track_id in chosen:
            track = by_id.get(track_id)
            if track is None:
                continue
            fields = _indexed_fields(track)
            hits.append(
                RetrievalHit(
                    source_type=SourceType.CATALOG,
                    source_id=track.ref.source_id,
                    content_hash=_content_hash(track, fields),
                    fields_used=fields,
                    score=scores[track_id],
                    matched_terms=_matched_terms(query, track),
                    lexical_score=text_scores.get(track_id),
                    structured_score=structured_scores.get(track_id),
                    structured_reasons=tuple(
                        next(
                            (hit.reasons for hit in structured_hits if hit.track_id == track_id),
                            (),
                        )
                    ),
                    fusion_version=FUSION_VERSION if fusion else None,
                    track=track,
                )
            )
        return RetrievalResult(
            query=query,
            hits=tuple(hits),
            index_fingerprint=self._fingerprint,
            operating_mode=OperatingMode.LOCAL,
        )

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        instrumental_only: bool = False,
        exclude_explicit: bool = False,
        use_guides: bool = False,
    ) -> RetrievalResult:
        """Protocol-compatible text-only search; FMA context guides stay disabled."""
        if k < 1:
            raise ValueError("k must be at least 1")
        if instrumental_only or exclude_explicit:
            return self._empty(
                query,
                instrumental_only=instrumental_only,
                exclude_explicit=exclude_explicit,
            )
        text_hits = self.store.text_search(query, limit=max(TEXT_POOL_SIZE, k))
        ranked = tuple((hit.track_id, hit.score) for hit in text_hits)
        return self._result(
            query,
            ranked,
            text_hits=text_hits,
            structured_hits=(),
            k=k,
            fusion=False,
        )

    def sample_diverse(self, k: int = 5) -> RetrievalResult:
        """A deterministic genre-spread starting set for an unmatched request."""
        if k < 1:
            raise ValueError("k must be at least 1")
        sample = self.store.sample_diverse(k)
        ranked = tuple((hit.track_id, hit.score) for hit in sample)
        return self._result(
            "", ranked, text_hits=(), structured_hits=(), k=k, fusion=False
        )

    def search_with_intent(self, intent: MusicIntent, *, k: int | None = None) -> RetrievalResult:
        """Union independent FTS/structured pools and fuse their ranks with RRF."""
        limit = intent.limit if k is None else k
        if limit < 1:
            raise ValueError("k must be at least 1")
        if intent.instrumental_only or intent.exclude_explicit:
            return self._empty(
                intent.query,
                instrumental_only=intent.instrumental_only,
                exclude_explicit=intent.exclude_explicit,
            )
        text_hits = self.store.text_search(intent.query, limit=TEXT_POOL_SIZE)
        structured_hits = self.store.structured_search(
            genre=intent.genre,
            goals=intent.feature_goals,
            limit=STRUCTURED_POOL_SIZE,
        )
        ranked = self._rrf(text_hits, structured_hits)
        return self._result(
            intent.query,
            ranked,
            text_hits=text_hits,
            structured_hits=structured_hits,
            k=limit,
            fusion=bool(text_hits and structured_hits),
        )
