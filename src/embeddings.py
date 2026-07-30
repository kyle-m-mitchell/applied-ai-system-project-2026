"""Text embeddings and their cache — the semantic half of Feature 4.

An *embedding* turns text into a dense vector of meaning, so paraphrases and
word-form variants ("studying" vs "study") land near each other — something the
lexical TF-IDF retriever cannot do. This module defines the provider-agnostic
:class:`Embedder` interface, a deterministic offline :class:`FakeEmbedder` for
tests and development, a real :class:`GeminiEmbedder`, and a small on-disk cache.

Reproducibility is the governing concern. A live API is non-portable, so:

* track vectors are computed once and cached to a committed JSON file, keyed on
  the catalog content plus the embedding model and dimension;
* tests and the fallback use the deterministic ``FakeEmbedder`` and never touch
  the network;
* the ``google-genai`` SDK is imported lazily, only inside ``GeminiEmbedder``, so
  the rest of the system runs with the package absent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 768

# Paraphrase-style queries used to build the committed query cache, so the
# semantic demo reproduces offline with no key. Each avoids exact catalog words.
EXAMPLE_QUERIES = (
    "music to concentrate",
    "tunes for cramming before an exam",
    "something to help me unwind before bed",
    "high-energy songs for a hard run",
    "wistful rainy-day music",
)

Vector = tuple[float, ...]

_WORD = re.compile(r"[a-z0-9]+")


def normalize(vector: Sequence[float]) -> Vector:
    """Scale a dense vector to unit length so cosine equals a dot product."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(0.0 for _ in vector)
    return tuple(value / norm for value in vector)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two dense vectors (exact dot product if normalized)."""
    return sum(x * y for x, y in zip(a, b))


class Embedder(ABC):
    """Provider-agnostic text embedder.

    Concrete embedders set ``model_id`` and ``dimension`` and return unit-length
    vectors, so a cosine is a plain dot product.
    """

    model_id: str
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        """Embed catalog documents (indexing side)."""

    @abstractmethod
    def embed_query(self, text: str) -> Vector:
        """Embed one listener query (retrieval side)."""


class FakeEmbedder(Embedder):
    """Deterministic, offline embedder for tests and key-free development.

    It hashes each word into a few dimensions, so identical text yields an
    identical vector and shared words pull vectors together. It captures no real
    semantics (it cannot know "studying" ~ "study"); it exists to exercise the
    caching, blending, and fallback plumbing without a network call.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM, model_id: str = "fake-embedder-v1") -> None:
        self.dimension = dimension
        self.model_id = model_id

    def _embed_one(self, text: str) -> Vector:
        vector = [0.0] * self.dimension
        for word in _WORD.findall(text.lower()):
            digest = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            for slot in range(4):
                index = (digest >> (slot * 17)) % self.dimension
                sign = 1.0 if (digest >> (slot * 7)) & 1 else -1.0
                vector[index] += sign
        return normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        return tuple(self._embed_one(text) for text in texts)

    def embed_query(self, text: str) -> Vector:
        return self._embed_one(text)


class GeminiEmbedder(Embedder):
    """Real embedder backed by Gemini via the ``google-genai`` SDK.

    The SDK and API key are touched lazily so importing this module never
    requires either. Matryoshka vectors truncated to ``dimension`` are
    re-normalized, and the key and prompt text are never logged.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIM,
    ) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set; provide a key via the environment "
                "(a git-ignored .env), never in code."
            )
        self._client = None  # created on first use

    def _client_lazy(self):
        if self._client is None:
            from google import genai  # imported only on the real path

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _embed(self, texts: Sequence[str], intent: str) -> tuple[Vector, ...]:
        from google.genai import types

        client = self._client_lazy()
        # Embedding 2 takes retrieval intent as in-text instruction rather than a
        # task_type field. Verify the exact request shape against the current API
        # docs before relying on live output.
        contents = [f"task: {intent} | {text}" for text in texts]
        response = client.models.embed_content(
            model=self.model_id,
            contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=self.dimension),
        )
        return tuple(normalize(tuple(item.values)) for item in response.embeddings)

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        return self._embed(texts, "search document")

    def embed_query(self, text: str) -> Vector:
        return self._embed([text], "search query")[0]


@dataclass(frozen=True)
class EmbeddingCache:
    """A validated, on-disk set of catalog vectors plus its identity."""

    embedding_model: str
    dimension: int
    content_hash: str
    vectors: dict[int, Vector]


def save_embedding_cache(path: str | Path, cache: EmbeddingCache) -> None:
    """Write an embedding cache as JSON (track vectors are lists of floats)."""
    payload = {
        "embedding_model": cache.embedding_model,
        "dimension": cache.dimension,
        "content_hash": cache.content_hash,
        "vectors": {str(track_id): list(vector) for track_id, vector in cache.vectors.items()},
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_embedding_cache(path: str | Path) -> EmbeddingCache:
    """Load an embedding cache from JSON, restoring integer ids and tuples."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    vectors = {
        int(track_id): tuple(float(value) for value in vector)
        for track_id, vector in payload["vectors"].items()
    }
    return EmbeddingCache(
        embedding_model=payload["embedding_model"],
        dimension=int(payload["dimension"]),
        content_hash=payload["content_hash"],
        vectors=vectors,
    )


@dataclass(frozen=True)
class QueryCache:
    """Cached embeddings for a fixed set of example queries (for offline demos)."""

    embedding_model: str
    dimension: int
    vectors: dict[str, Vector]


def save_query_cache(path: str | Path, cache: QueryCache) -> None:
    """Write a query cache as JSON keyed by the query text."""
    payload = {
        "embedding_model": cache.embedding_model,
        "dimension": cache.dimension,
        "vectors": {text: list(vector) for text, vector in cache.vectors.items()},
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_query_cache(path: str | Path) -> QueryCache:
    """Load a query cache from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    vectors = {
        text: tuple(float(value) for value in vector)
        for text, vector in payload["vectors"].items()
    }
    return QueryCache(
        embedding_model=payload["embedding_model"],
        dimension=int(payload["dimension"]),
        vectors=vectors,
    )


class CachedQueryEmbedder(Embedder):
    """Serve pre-embedded example queries from a cache; delegate anything else.

    Lets the semantic demo run offline for the committed example queries. A novel
    query goes to the live ``fallback`` embedder if one is supplied, otherwise it
    raises and the retriever degrades honestly.
    """

    def __init__(self, cache: QueryCache, fallback: Embedder | None = None) -> None:
        self.model_id = cache.embedding_model
        self.dimension = cache.dimension
        self._vectors = dict(cache.vectors)
        self._fallback = fallback

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        raise NotImplementedError("CachedQueryEmbedder is for queries only")

    def embed_query(self, text: str) -> Vector:
        if text in self._vectors:
            return self._vectors[text]
        if self._fallback is not None:
            return self._fallback.embed_query(text)
        raise RuntimeError("query is not cached and no live embedder is available")
