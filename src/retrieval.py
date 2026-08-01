"""Deterministic, offline text retrieval over the validated catalog.

This module is the *retrieval* half of RAG. It does not call a language model.
It turns each track's descriptive text into a TF-IDF vector and ranks tracks by
cosine similarity to a query, so requests expressed in words ("late-night study
beats") can find relevant tracks that the numeric scorer alone cannot.

Feature 3b adds a second source: curated context guides. A guide is not a
recommendable track. Instead, when a query matches a guide, the guide's
distinctive catalog-vocabulary terms are folded into the query (query
expansion), which lets a listener's words ("music to concentrate") reach tracks
that never use that exact word. Guides ride along as cited evidence.

Everything here is pure Python (standard library only). The math is written out
rather than hidden inside a dependency so it can be read, tested, and trusted.
TF-IDF is *lexical*: it matches word forms, not meaning, so "studying" will not
match "study". Provider embeddings behind the :class:`Retriever` interface are
the planned next step for closing that gap.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.contracts import (
    CatalogTrack,
    ContextGuide,
    GuideEvidence,
    OperatingMode,
    RetrievalHit,
    RetrievalResult,
    SourceType,
)
from src.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    CachedQueryEmbedder,
    Embedder,
    EmbeddingCache,
    cosine,
    load_embedding_cache,
    load_query_cache,
)


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

# Query-expansion tuning: how many matching guides may contribute, how many
# distinctive terms to take from each, and how dominant a guide must be to count.
# The ratio drops weak, spurious guide matches (a guide scoring far below the
# best one only injects off-topic terms).
MAX_EXPANSION_GUIDES = 2
TERMS_PER_GUIDE = 8
GUIDE_SCORE_RATIO = 0.5

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


def load_context_guides(directory: str) -> list[ContextGuide]:
    """Load curated context guides from a directory of Markdown files.

    Each file's first ``# Heading`` is the title and the remainder is the body.
    The file stem is the guide id. Files are read in sorted order for
    determinism.
    """
    directory_path = Path(directory)
    guides: list[ContextGuide] = []
    for path in sorted(directory_path.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        lines = text.splitlines()
        if not lines or not lines[0].startswith("# "):
            raise ValueError(
                f"context guide {path.name} must start with a '# Title' heading"
            )
        title = lines[0][2:].strip()
        body = "\n".join(lines[1:]).strip()
        guides.append(
            ContextGuide.model_validate(
                {"guide_id": path.stem, "title": title, "body": body}
            )
        )
    if not guides:
        raise ValueError(f"no context guides found in {directory}")
    return guides


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


def _apply_hard_filters(
    tracks: tuple[CatalogTrack, ...],
    instrumental_only: bool,
    exclude_explicit: bool,
) -> tuple[tuple[CatalogTrack, ...], tuple[str, ...]]:
    """Narrow candidates by hard constraints *before* any ranking.

    A hard constraint defines the candidate set; it is never a soft penalty a
    high similarity could override. Shared by every retriever.
    """
    candidates = tracks
    applied: list[str] = []
    if instrumental_only:
        candidates = tuple(track for track in candidates if track.instrumental)
        applied.append("instrumental_only")
    if exclude_explicit:
        candidates = tuple(track for track in candidates if not track.explicit)
        applied.append("exclude_explicit")
    return candidates, tuple(applied)


class _TfidfIndex:
    """A small TF-IDF index over an ``id -> text`` mapping.

    Shared by both sources (tracks and context guides) so the TF-IDF math lives
    in exactly one place.
    """

    def __init__(self, documents: dict[Any, str]) -> None:
        self.content_hashes = {
            key: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for key, text in documents.items()
        }
        tokenized = {key: _tokenize(text) for key, text in documents.items()}

        n_docs = len(documents)
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized.values():
            document_frequency.update(set(tokens))
        self.idf = {
            term: math.log((n_docs + 1) / (df + 1)) + 1.0
            for term, df in document_frequency.items()
        }

        self.vectors: dict[Any, dict[str, float]] = {}
        for key, tokens in tokenized.items():
            term_frequency = Counter(tokens)
            weights = {
                term: count * self.idf[term]
                for term, count in term_frequency.items()
            }
            self.vectors[key] = _l2_normalize(weights)

    def query_vector(self, tokens: list[str]) -> dict[str, float]:
        """Build a normalized query vector, ignoring terms not in this index."""
        term_frequency = Counter(tokens)
        weights = {
            term: count * self.idf[term]
            for term, count in term_frequency.items()
            if term in self.idf
        }
        return _l2_normalize(weights)

    def top_terms(self, key: Any, limit: int) -> tuple[str, ...]:
        """Return a document's most distinctive terms, highest TF-IDF first."""
        ordered = sorted(
            self.vectors[key].items(),
            key=lambda item: (-item[1], item[0]),
        )
        return tuple(term for term, _weight in ordered[:limit])


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
    override a hard constraint. When curated context guides are supplied, a
    matching guide expands the query toward catalog vocabulary and is recorded
    as evidence.
    """

    def __init__(
        self,
        tracks: Sequence[CatalogTrack],
        guides: Sequence[ContextGuide] = (),
    ) -> None:
        if not tracks:
            raise ValueError("retriever requires at least one track")

        self._tracks = tuple(tracks)
        self._tracks_by_id = {track.id: track for track in self._tracks}

        documents = {track.id: build_document_text(track) for track in self._tracks}
        self._track_index = _TfidfIndex(documents)
        self._content_hashes = self._track_index.content_hashes

        self._guides = tuple(guides)
        self._guides_by_id = {guide.guide_id: guide for guide in self._guides}
        self._guide_index: _TfidfIndex | None = (
            _TfidfIndex({guide.guide_id: guide.index_text() for guide in self._guides})
            if self._guides
            else None
        )

        # Computed last: the fingerprint depends on every input that changes
        # retrieval results — both sources' content and the expansion settings.
        self._fingerprint = self._compute_fingerprint()

    @property
    def index_fingerprint(self) -> str:
        """Identify this exact index: method, both sources' content, and settings."""
        return self._fingerprint

    def _compute_fingerprint(self) -> str:
        """Fingerprint everything that determines retrieval output.

        A cache keyed on this must rebuild whenever the method, either source's
        content, or the query-expansion tuning changes.
        """
        parts = [
            METHOD_ID,
            f"expansion:{MAX_EXPANSION_GUIDES},{TERMS_PER_GUIDE},{GUIDE_SCORE_RATIO}",
            "tracks",
        ]
        parts.extend(
            f"{track_id}:{self._content_hashes[track_id]}"
            for track_id in sorted(self._content_hashes)
        )
        parts.append("guides")
        if self._guide_index is not None:
            parts.extend(
                f"{guide_id}:{self._guide_index.content_hashes[guide_id]}"
                for guide_id in sorted(self._guide_index.content_hashes)
            )
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def _expand_with_guides(
        self,
        query_tokens: list[str],
    ) -> tuple[tuple[GuideEvidence, ...], list[str]]:
        """Retrieve matching guides and collect catalog terms they contribute."""
        assert self._guide_index is not None
        guide_query = self._guide_index.query_vector(query_tokens)

        matches = [
            (guide_id, _cosine(guide_query, self._guide_index.vectors[guide_id]))
            for guide_id in self._guides_by_id
        ]
        matches = [(guide_id, score) for guide_id, score in matches if score > 0.0]
        matches.sort(key=lambda item: (-item[1], item[0]))
        if matches:
            # Keep only guides that clearly matched, relative to the best one.
            floor = matches[0][1] * GUIDE_SCORE_RATIO
            matches = [item for item in matches if item[1] >= floor]

        already = set(query_tokens)
        added: list[str] = []
        evidence: list[GuideEvidence] = []
        for guide_id, score in matches[:MAX_EXPANSION_GUIDES]:
            guide = self._guides_by_id[guide_id]
            # A guide only helps track retrieval through terms the catalog uses;
            # bridge words that no track contains are naturally dropped here.
            contributed = [
                term
                for term in self._guide_index.top_terms(guide_id, TERMS_PER_GUIDE)
                if term in self._track_index.idf
                and term not in already
                and term not in added
            ]
            added.extend(contributed)
            matched = tuple(
                sorted(
                    set(query_tokens) & set(self._guide_index.vectors[guide_id]),
                    key=lambda term: (-self._guide_index.idf[term], term),
                )
            )
            evidence.append(
                GuideEvidence(
                    source_type=SourceType.CONTEXT_GUIDE,
                    source_id=f"context_guide:{guide_id}",
                    content_hash=self._guide_index.content_hashes[guide_id],
                    title=guide.title,
                    score=score,
                    matched_terms=matched,
                    expansion_terms=tuple(contributed),
                )
            )
        return tuple(evidence), added

    def _lexical_scores(
        self,
        query: str,
        candidates: tuple[CatalogTrack, ...],
        use_guides: bool,
    ) -> tuple[
        dict[int, float],
        dict[int, tuple[str, ...]],
        tuple[GuideEvidence, ...],
        tuple[str, ...],
    ]:
        """Score candidates lexically, expanding via guides when they help.

        Returns per-track scores and matched terms plus the guide evidence and
        expansion terms, so both ``search`` and the hybrid retriever can reuse it.
        """
        query_tokens = _tokenize(query)

        guides_used: tuple[GuideEvidence, ...] = ()
        expansion_terms: tuple[str, ...] = ()
        search_tokens = query_tokens
        if use_guides and self._guide_index is not None and query_tokens:
            guides_used, added_terms = self._expand_with_guides(query_tokens)
            if added_terms:
                search_tokens = query_tokens + added_terms
                expansion_terms = tuple(added_terms)

        query_vector = self._track_index.query_vector(search_tokens)

        scores: dict[int, float] = {}
        matched_terms: dict[int, tuple[str, ...]] = {}
        for track in candidates:
            document_vector = self._track_index.vectors[track.id]
            score = _cosine(query_vector, document_vector)
            if score <= 0.0:
                # No lexical overlap: do not claim relevance we cannot justify.
                continue
            shared = set(query_vector) & set(document_vector)
            scores[track.id] = score
            # Most distinctive shared terms first (highest IDF), ties alphabetical.
            matched_terms[track.id] = tuple(
                sorted(shared, key=lambda term: (-self._track_index.idf[term], term))
            )
        return scores, matched_terms, guides_used, expansion_terms

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        instrumental_only: bool = False,
        exclude_explicit: bool = False,
        use_guides: bool = True,
    ) -> RetrievalResult:
        """Return up to ``k`` positively-scored tracks, most relevant first."""
        if k < 1:
            raise ValueError("k must be at least 1")

        candidates, filters_applied = _apply_hard_filters(
            self._tracks, instrumental_only, exclude_explicit
        )
        scores, matched_terms, guides_used, expansion_terms = self._lexical_scores(
            query, candidates, use_guides
        )

        # Rank by score, breaking ties by id to match the scorer's stable order.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]

        hits = tuple(
            RetrievalHit(
                source_type=SourceType.CATALOG,
                source_id=f"catalog:{track_id}",
                content_hash=self._content_hashes[track_id],
                fields_used=RETRIEVAL_FIELDS,
                score=score,
                matched_terms=matched_terms[track_id],
                lexical_score=score,
                track=self._tracks_by_id[track_id],
            )
            for track_id, score in ranked
        )

        return RetrievalResult(
            query=query,
            hits=hits,
            index_fingerprint=self._fingerprint,
            filters_applied=filters_applied,
            guides_used=guides_used,
            expanded_query_terms=expansion_terms,
            operating_mode=OperatingMode.LOCAL,
        )


# Default hybrid weights: semantic leads, lexical anchors. Configurable so the
# handbook's fuller 55/35/10 (adding the numeric scorer and session feedback)
# drops in once the intent parser and session memory exist.
W_SEMANTIC = 0.6
W_LEXICAL = 0.4


def embedding_content_hash(
    tracks: Sequence[CatalogTrack],
    model_id: str,
    dimension: int,
) -> str:
    """Identify a catalog embedding index by its track content, model, and size.

    Independent of guides and expansion tuning (which do not affect embeddings).
    Embedding spaces differ by model and dimension, so both belong in the key.
    """
    parts = [f"embed:{model_id}:{dimension}"]
    items = sorted(
        (track.id, hashlib.sha256(build_document_text(track).encode("utf-8")).hexdigest())
        for track in tracks
    )
    parts.extend(f"{track_id}:{digest}" for track_id, digest in items)
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


class EmbeddingRetriever(Retriever):
    """Semantic retrieval over cached track embeddings.

    Track vectors come from a committed cache (built once by
    ``scripts/build_embeddings.py``); only the query is embedded live. If the
    cache is missing or stale, or the embedder is unavailable, it delegates to a
    TF-IDF fallback and labels the result ``DEGRADED`` — never silently.
    """

    def __init__(
        self,
        tracks: Sequence[CatalogTrack],
        embedder: Embedder,
        cache: EmbeddingCache | None,
        *,
        fallback: Retriever | None = None,
    ) -> None:
        if not tracks:
            raise ValueError("retriever requires at least one track")
        self._tracks = tuple(tracks)
        self._tracks_by_id = {track.id: track for track in self._tracks}
        self._content_hashes = {
            track.id: hashlib.sha256(build_document_text(track).encode("utf-8")).hexdigest()
            for track in self._tracks
        }
        self._embedder = embedder
        self._cache = cache
        self._fallback = fallback if fallback is not None else TfidfRetriever(self._tracks)

        model = cache.embedding_model if cache is not None else EMBEDDING_MODEL
        dimension = cache.dimension if cache is not None else EMBEDDING_DIM
        self._expected_hash = embedding_content_hash(self._tracks, model, dimension)
        self._usable = cache is not None and cache.content_hash == self._expected_hash
        self._fingerprint = self._expected_hash

    @property
    def index_fingerprint(self) -> str:
        """Identify this embedding index (model, dimension, catalog content)."""
        return self._fingerprint

    @property
    def usable(self) -> bool:
        """True when a valid, current cache backs semantic search."""
        return self._usable

    def _embed_query(self, query: str) -> tuple[float, ...]:
        """Embed the query once. The single retry lives in the provider adapter's
        ``_post`` (retrying here as well would multiply attempts and delay)."""
        return self._embedder.embed_query(query)

    def _degraded(
        self, query: str, k: int, instrumental_only: bool, exclude_explicit: bool
    ) -> RetrievalResult:
        result = self._fallback.search(
            query, k=k, instrumental_only=instrumental_only, exclude_explicit=exclude_explicit
        )
        return result.model_copy(update={"operating_mode": OperatingMode.DEGRADED})

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        instrumental_only: bool = False,
        exclude_explicit: bool = False,
    ) -> RetrievalResult:
        """Return up to ``k`` tracks by semantic similarity, or a labeled fallback."""
        if k < 1:
            raise ValueError("k must be at least 1")
        if not self._usable:
            return self._degraded(query, k, instrumental_only, exclude_explicit)
        try:
            query_vector = self._embed_query(query)
        except Exception:  # noqa: BLE001 - provider failed; degrade honestly
            return self._degraded(query, k, instrumental_only, exclude_explicit)

        candidates, filters_applied = _apply_hard_filters(
            self._tracks, instrumental_only, exclude_explicit
        )
        scored: list[tuple[int, float]] = []
        for track in candidates:
            vector = self._cache.vectors.get(track.id)
            if vector is None:
                continue
            similarity = cosine(query_vector, vector)
            if similarity <= 0.0:
                continue
            scored.append((track.id, min(1.0, similarity)))
        scored.sort(key=lambda item: (-item[1], item[0]))

        hits = tuple(
            RetrievalHit(
                source_type=SourceType.CATALOG,
                source_id=f"catalog:{track_id}",
                content_hash=self._content_hashes[track_id],
                fields_used=RETRIEVAL_FIELDS,
                score=similarity,
                semantic_score=similarity,
                track=self._tracks_by_id[track_id],
            )
            for track_id, similarity in scored[:k]
        )
        return RetrievalResult(
            query=query,
            hits=hits,
            index_fingerprint=self._fingerprint,
            filters_applied=filters_applied,
            operating_mode=OperatingMode.GEMINI,
        )


class HybridRetriever(Retriever):
    """Blend semantic (embeddings) and lexical (TF-IDF) scores into one ranking.

    Dense retrieval understands meaning; sparse retrieval is exact and offline.
    Blending them beats either alone. If embeddings are unavailable it degrades
    to pure TF-IDF and labels the result ``DEGRADED``.
    """

    def __init__(
        self,
        tracks: Sequence[CatalogTrack],
        embedder: Embedder,
        cache: EmbeddingCache | None,
        guides: Sequence[ContextGuide] = (),
        *,
        w_semantic: float = W_SEMANTIC,
        w_lexical: float = W_LEXICAL,
    ) -> None:
        if not tracks:
            raise ValueError("retriever requires at least one track")
        self._tracks = tuple(tracks)
        self._tracks_by_id = {track.id: track for track in self._tracks}
        self._w_semantic = w_semantic
        self._w_lexical = w_lexical
        self._tfidf = TfidfRetriever(self._tracks, guides)
        self._embedding = EmbeddingRetriever(
            self._tracks, embedder, cache, fallback=self._tfidf
        )
        self._content_hashes = self._tfidf._content_hashes
        self._fingerprint = hashlib.sha256(
            (
                f"hybrid:{w_semantic},{w_lexical}"
                f"|{self._tfidf.index_fingerprint}|{self._embedding.index_fingerprint}"
            ).encode("utf-8")
        ).hexdigest()

    @property
    def index_fingerprint(self) -> str:
        """Identify the hybrid: weights plus both sub-index fingerprints."""
        return self._fingerprint

    def _degraded(
        self,
        query: str,
        k: int,
        instrumental_only: bool,
        exclude_explicit: bool,
        use_guides: bool,
    ) -> RetrievalResult:
        result = self._tfidf.search(
            query,
            k=k,
            instrumental_only=instrumental_only,
            exclude_explicit=exclude_explicit,
            use_guides=use_guides,
        )
        return result.model_copy(update={"operating_mode": OperatingMode.DEGRADED})

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        instrumental_only: bool = False,
        exclude_explicit: bool = False,
        use_guides: bool = True,
    ) -> RetrievalResult:
        """Return up to ``k`` tracks ranked by the blended semantic+lexical score."""
        if k < 1:
            raise ValueError("k must be at least 1")
        if not self._embedding.usable:
            return self._degraded(query, k, instrumental_only, exclude_explicit, use_guides)
        try:
            query_vector = self._embedding._embed_query(query)
        except Exception:  # noqa: BLE001 - provider failed; degrade to lexical
            return self._degraded(query, k, instrumental_only, exclude_explicit, use_guides)

        candidates, filters_applied = _apply_hard_filters(
            self._tracks, instrumental_only, exclude_explicit
        )
        lexical_scores, matched_terms, guides_used, expansion_terms = self._tfidf._lexical_scores(
            query, candidates, use_guides
        )

        blended: list[tuple[int, float, float, float, tuple[str, ...]]] = []
        for track in candidates:
            vector = self._embedding._cache.vectors.get(track.id)
            semantic = min(1.0, max(0.0, cosine(query_vector, vector))) if vector else 0.0
            lexical = lexical_scores.get(track.id, 0.0)
            score = self._w_semantic * semantic + self._w_lexical * lexical
            if score <= 0.0:
                continue
            blended.append((track.id, score, semantic, lexical, matched_terms.get(track.id, ())))
        blended.sort(key=lambda item: (-item[1], item[0]))

        hits = tuple(
            RetrievalHit(
                source_type=SourceType.CATALOG,
                source_id=f"catalog:{track_id}",
                content_hash=self._content_hashes[track_id],
                fields_used=RETRIEVAL_FIELDS,
                score=min(1.0, score),
                matched_terms=matched,
                semantic_score=semantic,
                lexical_score=lexical,
                track=self._tracks_by_id[track_id],
            )
            for track_id, score, semantic, lexical, matched in blended[:k]
        )
        return RetrievalResult(
            query=query,
            hits=hits,
            index_fingerprint=self._fingerprint,
            filters_applied=filters_applied,
            guides_used=guides_used,
            expanded_query_terms=expansion_terms,
            operating_mode=OperatingMode.GEMINI,
        )


def build_default_retriever(
    tracks: Sequence[CatalogTrack],
    guides: Sequence[ContextGuide] = (),
    *,
    catalog_cache_path: str | None = None,
    query_cache_path: str | None = None,
    live_embedder: Embedder | None = None,
) -> Retriever:
    """Return the best retriever available: hybrid if a usable committed
    embedding cache exists, otherwise the local TF-IDF retriever.

    Cached example-query vectors keep the hybrid reproducible offline; a
    ``live_embedder`` (when a key is present) covers novel queries. Any missing
    or mismatched cache simply falls back to TF-IDF. Shared by the demo, the CLI,
    and the companion so the "which retriever?" logic lives in one place.
    """
    cache = None
    if catalog_cache_path is not None and Path(catalog_cache_path).exists():
        try:
            cache = load_embedding_cache(catalog_cache_path)
        except Exception:  # noqa: BLE001 - a bad cache just falls back to TF-IDF
            cache = None
    if cache is None:
        return TfidfRetriever(tracks, guides)

    embedder: Embedder | None = None
    if query_cache_path is not None and Path(query_cache_path).exists():
        try:
            query_cache = load_query_cache(query_cache_path)
            if query_cache.embedding_model == cache.embedding_model:
                embedder = CachedQueryEmbedder(query_cache, fallback=live_embedder)
        except Exception:  # noqa: BLE001
            embedder = None
    if embedder is None and live_embedder is not None and live_embedder.model_id == cache.embedding_model:
        embedder = live_embedder
    if embedder is None:
        return TfidfRetriever(tracks, guides)
    return HybridRetriever(tracks, embedder, cache, guides)
