"""Deterministic, offline text retrieval over the validated catalog.

This module is the *retrieval* half of RAG. It does not call a language model.
It turns each track's descriptive text into a TF-IDF vector and ranks tracks by
cosine similarity to a query, so requests expressed in words ("late-night study
beats") can find relevant tracks that the numeric scorer alone cannot.

Everything here is pure Python (standard library only). The math is written out
rather than hidden inside a dependency so it can be read, tested, and trusted.
TF-IDF is *lexical*: it matches word forms, not meaning, so "studying" will not
match "study". That gap is intentional groundwork for a later provider-embedding
retriever that will share the :class:`Retriever` interface defined below.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence

from src.contracts import CatalogTrack, RetrievalHit, RetrievalResult, SourceType


# Descriptor fields joined into one retrieval document per track. Title and
# artist are identity, not description, and the ``description`` text already
# embeds them, so they are deliberately excluded.
RETRIEVAL_FIELDS: tuple[str, ...] = (
    "genre",
    "mood",
    "era",
    "description",
    "tags",
    "contexts",
    "instruments",
)

# Bumped whenever the document construction or scoring math changes, so a cached
# index built by an older method can be detected via the index fingerprint.
METHOD_ID = "tfidf-v1"

# A small, inspectable stop list. IDF already down-weights common words; this
# just removes the most frequent function words before they reach the vectors.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
        "to", "with", "your", "you",
    }
)

MIN_TOKEN_LENGTH = 2
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric terms, dropping stopwords."""
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if len(token) >= MIN_TOKEN_LENGTH and token not in STOPWORDS
    ]


def build_document_text(track: CatalogTrack) -> str:
    """Assemble one canonical retrieval document from a track's descriptor fields."""
    parts: list[str] = []
    for field in RETRIEVAL_FIELDS:
        value = getattr(track, field)
        if isinstance(value, tuple):
            parts.append(" ".join(value))
        else:
            parts.append(str(value))
    return " ".join(parts)


def _l2_normalize(vector: dict[str, float]) -> dict[str, float]:
    """Scale a vector to unit length so cosine similarity is a plain dot product."""
    norm = math.sqrt(sum(weight * weight for weight in vector.values()))
    if norm == 0.0:
        return {}
    return {term: weight / norm for term, weight in vector.items()}


def _cosine(query: dict[str, float], document: dict[str, float]) -> float:
    """Cosine similarity of two already L2-normalized sparse vectors."""
    if not query or not document:
        return 0.0
    # Iterate the smaller vector; missing terms contribute zero.
    smaller, larger = (query, document) if len(query) <= len(document) else (document, query)
    similarity = sum(weight * larger.get(term, 0.0) for term, weight in smaller.items())
    # Guard against tiny floating-point overshoot above 1.0.
    return min(1.0, max(0.0, similarity))


class Retriever(ABC):
    """Interface every retrieval strategy shares.

    A future ``GeminiEmbeddingRetriever`` will implement the same ``search`` so
    the agent and evaluator can treat local and provider modes identically.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        k: int = 5,
        instrumental_only: bool = False,
        exclude_explicit: bool = False,
    ) -> RetrievalResult:
        """Return the top ``k`` tracks relevant to ``query`` with provenance."""


class TfidfRetriever(Retriever):
    """In-memory TF-IDF retriever over a validated catalog.

    The index is built once at construction. Hard filters (instrumental-only,
    clean-only) are applied *before* ranking, so a high similarity can never
    override a hard constraint.
    """

    def __init__(self, tracks: Sequence[CatalogTrack]) -> None:
        if not tracks:
            raise ValueError("retriever requires at least one track")

        self._tracks = tuple(tracks)
        self._tracks_by_id = {track.id: track for track in self._tracks}

        # 1. One retrieval document per track, plus a content hash for provenance.
        documents = {track.id: build_document_text(track) for track in self._tracks}
        self._content_hashes = {
            track_id: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for track_id, text in documents.items()
        }
        tokenized = {track_id: _tokenize(text) for track_id, text in documents.items()}

        # 2. Inverse document frequency: rare terms carry more weight.
        n_docs = len(self._tracks)
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized.values():
            document_frequency.update(set(tokens))
        self._idf = {
            term: math.log((n_docs + 1) / (df + 1)) + 1.0
            for term, df in document_frequency.items()
        }

        # 3. Per-document normalized TF-IDF vectors.
        self._vectors: dict[int, dict[str, float]] = {}
        for track_id, tokens in tokenized.items():
            term_frequency = Counter(tokens)
            weights = {
                term: count * self._idf[term]
                for term, count in term_frequency.items()
            }
            self._vectors[track_id] = _l2_normalize(weights)

        self._fingerprint = self._compute_fingerprint()

    @property
    def index_fingerprint(self) -> str:
        """Identify this exact index (method + catalog content)."""
        return self._fingerprint

    def _compute_fingerprint(self) -> str:
        """Derive a stable fingerprint from the method id and per-doc hashes."""
        parts = [METHOD_ID]
        parts.extend(
            f"{track_id}:{self._content_hashes[track_id]}"
            for track_id in sorted(self._content_hashes)
        )
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def _query_vector(self, tokens: list[str]) -> dict[str, float]:
        """Build a normalized query vector, ignoring terms not in the index."""
        term_frequency = Counter(tokens)
        weights = {
            term: count * self._idf[term]
            for term, count in term_frequency.items()
            if term in self._idf
        }
        return _l2_normalize(weights)

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        instrumental_only: bool = False,
        exclude_explicit: bool = False,
    ) -> RetrievalResult:
        """Return up to ``k`` positively-scored tracks, most relevant first."""
        if k < 1:
            raise ValueError("k must be at least 1")

        # Hard filters run first: they define the candidate set, not a penalty.
        candidates = self._tracks
        filters_applied: list[str] = []
        if instrumental_only:
            candidates = tuple(track for track in candidates if track.instrumental)
            filters_applied.append("instrumental_only")
        if exclude_explicit:
            candidates = tuple(track for track in candidates if not track.explicit)
            filters_applied.append("exclude_explicit")

        query_vector = self._query_vector(_tokenize(query))

        scored: list[tuple[CatalogTrack, float, tuple[str, ...]]] = []
        for track in candidates:
            document_vector = self._vectors[track.id]
            score = _cosine(query_vector, document_vector)
            if score <= 0.0:
                # No lexical overlap: do not claim relevance we cannot justify.
                continue
            shared = set(query_vector) & set(document_vector)
            # Most distinctive shared terms first (highest IDF), ties alphabetical.
            matched = tuple(sorted(shared, key=lambda term: (-self._idf[term], term)))
            scored.append((track, score, matched))

        # Rank by score, breaking ties by id to match the scorer's stable order.
        scored.sort(key=lambda item: (-item[1], item[0].id))

        hits = tuple(
            RetrievalHit(
                source_type=SourceType.CATALOG,
                source_id=f"catalog:{track.id}",
                content_hash=self._content_hashes[track.id],
                fields_used=RETRIEVAL_FIELDS,
                score=score,
                matched_terms=matched,
                track=track,
            )
            for track, score, matched in scored[:k]
        )

        return RetrievalResult(
            query=query,
            hits=hits,
            index_fingerprint=self._fingerprint,
            filters_applied=tuple(filters_applied),
        )
