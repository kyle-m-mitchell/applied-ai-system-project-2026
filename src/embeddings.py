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
* the real embedder calls the Gemini REST API with the Python standard library
  (``urllib``) plus ``certifi`` for TLS certificate verification, so there is no
  SDK to compile and it runs on any supported Python.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import ssl
import time
import urllib.error
import urllib.request
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


def _make_ssl_context() -> ssl.SSLContext:
    """Build a TLS context, preferring certifi's CA bundle when available.

    Some Python builds (notably the python.org macOS installers) ship without a
    usable system CA bundle, so a plain default context fails verification.
    ``certifi`` is a pure-Python bundle (no compilation), used only if present.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 - fall back to the system default context
        return ssl.create_default_context()


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
    """Real embedder calling the Gemini REST API with the standard library.

    No third-party SDK: a plain HTTPS POST to the embeddings endpoint. Matryoshka
    vectors truncated to ``dimension`` are re-normalized, and the key is sent in a
    header (never a URL) and never logged.

    Embedding 2 takes retrieval intent as an in-text instruction rather than a
    task-type field. Model id, endpoint, and payload shape drift over time; verify
    them against the current API docs before relying on live output. Any failure
    raises, and the retriever degrades to TF-IDF.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIM,
        *,
        timeout: float = 30.0,
        request_delay: float = 0.0,
        max_retries: int = 5,
    ) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self._timeout = timeout
        self._request_delay = request_delay  # gentle throttle to stay under RPM
        self._max_retries = max_retries
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set; provide a key via the environment "
                "(a git-ignored .env), never in code."
            )
        self._ssl_context = _make_ssl_context()

    def _post(self, method: str, payload: dict) -> dict:
        """POST JSON to a model method, retrying on rate limits with backoff."""
        url = f"{self.BASE_URL}/models/{self.model_id}:{method}"
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(self._max_retries + 1):
            request = urllib.request.Request(url, data=body, method="POST")
            request.add_header("Content-Type", "application/json")
            request.add_header("x-goog-api-key", self._api_key)  # key in header, not URL
            try:
                with urllib.request.urlopen(
                    request, timeout=self._timeout, context=self._ssl_context
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:  # never leak the key or request text
                retriable = exc.code in (429, 500, 503)
                if retriable and attempt < self._max_retries:
                    time.sleep(min(2**attempt, 30))  # 1, 2, 4, 8, 16 seconds
                    continue
                raise RuntimeError(f"Gemini embedding request failed: HTTP {exc.code}") from exc
        raise RuntimeError("Gemini embedding request failed after retries")

    def _embed_one(self, text: str) -> Vector:
        # The model exposes single embedContent (not sync batch), so we call once
        # per text. Raw text is embedded; task-type optimization is a later refinement.
        if self._request_delay:
            time.sleep(self._request_delay)
        payload = {
            "model": f"models/{self.model_id}",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.dimension,
        }
        data = self._post("embedContent", payload)
        return normalize(tuple(data["embedding"]["values"]))

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        return tuple(self._embed_one(text) for text in texts)

    def embed_query(self, text: str) -> Vector:
        return self._embed_one(text)


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
