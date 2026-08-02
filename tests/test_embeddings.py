"""Tests for semantic embeddings, the embedding/hybrid retrievers, and fallback.

All tests run fully offline with the deterministic ``FakeEmbedder`` — no network,
no API key. The fake captures no real semantics, so these tests assert on the
plumbing (ranking by cosine, blend math, caching, provenance, and honest
fallback), not on semantic quality (which the real Gemini path provides).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.contracts import CatalogTrack, EmbeddingSource, OperatingMode
from src.embeddings import (
    EMBEDDING_MODEL,
    EXAMPLE_QUERIES,
    CachedQueryEmbedder,
    EmbeddingCache,
    FakeEmbedder,
    QueryCache,
    load_embedding_cache,
    load_query_cache,
    save_embedding_cache,
    save_query_cache,
)
from src.recommender import load_songs
from src.retrieval import (
    EmbeddingRetriever,
    HybridRetriever,
    build_default_retriever,
    build_document_text,
    embedding_content_hash,
    load_context_guides,
)
from src.service import RecommendationService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "songs.csv"
GUIDES_DIR = PROJECT_ROOT / "data" / "context_guides"


@pytest.fixture(scope="module")
def catalog() -> tuple[CatalogTrack, ...]:
    return RecommendationService(load_songs(str(CATALOG_PATH))).catalog


@pytest.fixture(scope="module")
def guides():
    return load_context_guides(str(GUIDES_DIR))


@pytest.fixture(scope="module")
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dimension=64)


def _build_cache(tracks, embedder) -> EmbeddingCache:
    vectors = embedder.embed_documents([build_document_text(t) for t in tracks])
    return EmbeddingCache(
        embedding_model=embedder.model_id,
        dimension=embedder.dimension,
        content_hash=embedding_content_hash(tracks, embedder.model_id, embedder.dimension),
        vectors={track.id: vector for track, vector in zip(tracks, vectors)},
    )


@pytest.fixture(scope="module")
def cache(catalog, embedder) -> EmbeddingCache:
    return _build_cache(catalog, embedder)


def test_fake_embedder_is_deterministic_and_unit_length(embedder):
    first = embedder.embed_query("calm lofi study")
    second = embedder.embed_query("calm lofi study")
    assert first == second
    assert len(first) == embedder.dimension
    assert math.isclose(math.sqrt(sum(x * x for x in first)), 1.0, abs_tol=1e-9)


def test_cache_round_trips_through_disk(tmp_path, cache):
    path = tmp_path / "catalog.json"
    save_embedding_cache(path, cache)
    restored = load_embedding_cache(path)

    assert restored.embedding_model == cache.embedding_model
    assert restored.dimension == cache.dimension
    assert restored.content_hash == cache.content_hash
    assert restored.vectors[9] == cache.vectors[9]


def test_embedding_retriever_ranks_nearest_vector_first(catalog, embedder, cache):
    retriever = EmbeddingRetriever(catalog, embedder, cache)
    assert retriever.usable

    target = catalog[8]  # id 9
    result = retriever.search(build_document_text(target), k=3)

    assert result.operating_mode is OperatingMode.GEMINI
    assert result.embedding_source is EmbeddingSource.LOCAL
    assert result.hits[0].track.id == target.id
    assert result.hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert result.hits[0].semantic_score is not None
    assert result.hits[0].lexical_score is None


def test_embedding_retriever_applies_hard_filters(catalog, embedder, cache):
    instrumental = next(track for track in catalog if track.instrumental)
    retriever = EmbeddingRetriever(catalog, embedder, cache)
    result = retriever.search(
        build_document_text(instrumental), k=10, instrumental_only=True
    )

    assert result.filters_applied == ("instrumental_only",)
    assert result.hits
    assert all(hit.track.instrumental for hit in result.hits)


def test_stale_cache_degrades_to_fallback(catalog, embedder, cache):
    stale = EmbeddingCache(cache.embedding_model, cache.dimension, "wrong-hash", cache.vectors)
    retriever = EmbeddingRetriever(catalog, embedder, stale)

    assert not retriever.usable
    result = retriever.search("late night study", k=3)
    assert result.operating_mode is OperatingMode.DEGRADED
    assert result.hits  # the TF-IDF fallback still answered


def test_missing_cache_degrades(catalog, embedder):
    retriever = EmbeddingRetriever(catalog, embedder, None)
    assert not retriever.usable
    assert retriever.search("study", k=1).operating_mode is OperatingMode.DEGRADED


def test_partial_catalog_cache_is_never_treated_as_a_semantic_index(
    catalog, embedder, cache
):
    one_id = catalog[0].id
    partial = EmbeddingCache(
        cache.embedding_model,
        cache.dimension,
        cache.content_hash,
        {one_id: cache.vectors[one_id]},
    )
    retriever = EmbeddingRetriever(catalog, embedder, partial)
    assert not retriever.usable
    result = retriever.search("study", k=3)
    assert result.operating_mode is OperatingMode.DEGRADED


def test_query_vector_with_wrong_runtime_dimension_degrades_instead_of_crashing(
    catalog, cache
):
    class LyingEmbedder(FakeEmbedder):
        def embed_query(self, text):
            return FakeEmbedder(dimension=32).embed_query(text)

    retriever = HybridRetriever(
        catalog,
        LyingEmbedder(dimension=cache.dimension, model_id=cache.embedding_model),
        cache,
    )
    result = retriever.search("study", k=3)
    assert result.operating_mode is OperatingMode.DEGRADED


def test_embedder_error_degrades(catalog, cache):
    class Boom(FakeEmbedder):
        def embed_query(self, text):
            raise RuntimeError("no network")

    retriever = EmbeddingRetriever(catalog, Boom(dimension=64), cache)
    result = retriever.search("study", k=3)
    assert result.operating_mode is OperatingMode.DEGRADED
    assert result.hits


def test_failed_live_attempt_preserves_network_provenance(catalog, cache, guides):
    class LiveBoom(FakeEmbedder):
        def query_source(self, text):
            return EmbeddingSource.LIVE

        def embed_query(self, text):
            raise RuntimeError("provider reached, then failed")

    embedder = LiveBoom(dimension=64)
    semantic = EmbeddingRetriever(catalog, embedder, cache)
    hybrid = HybridRetriever(catalog, embedder, cache, guides)

    for result in (semantic.search("study"), hybrid.search("study")):
        assert result.operating_mode is OperatingMode.DEGRADED
        assert result.embedding_source is EmbeddingSource.LIVE


def test_hybrid_blends_semantic_and_lexical_exactly(catalog, embedder, cache, guides):
    retriever = HybridRetriever(catalog, embedder, cache, guides, w_semantic=0.7, w_lexical=0.3)
    result = retriever.search("late-night study beats to focus", k=5)

    assert result.operating_mode is OperatingMode.GEMINI
    assert result.hits
    for hit in result.hits:
        assert hit.semantic_score is not None
        assert hit.lexical_score is not None
        assert hit.score == pytest.approx(
            0.7 * hit.semantic_score + 0.3 * hit.lexical_score, abs=1e-9
        )
        assert hit.fusion_version == "weighted-sum:sem=0.7,lex=0.3;v1"


def test_hybrid_degrades_when_embeddings_unavailable(catalog, embedder, guides):
    retriever = HybridRetriever(catalog, embedder, None, guides)  # no cache
    result = retriever.search("music to concentrate", k=3)

    assert result.operating_mode is OperatingMode.DEGRADED
    assert result.hits  # TF-IDF + guide expansion still rescue the bridge query


def test_provider_free_cache_miss_is_expected_local_operation(
    catalog, embedder, cache, guides
):
    retriever = HybridRetriever(
        catalog,
        CachedQueryEmbedder(
            QueryCache(embedder.model_id, embedder.dimension, {})
        ),
        cache,
        guides,
        fallback_mode=OperatingMode.LOCAL,
    )
    result = retriever.search("uncached study query", k=3)
    assert result.operating_mode is OperatingMode.LOCAL
    assert result.embedding_source is EmbeddingSource.LOCAL


def test_embedding_content_hash_is_sensitive(catalog):
    base = embedding_content_hash(catalog, EMBEDDING_MODEL, 768)
    assert base != embedding_content_hash(catalog, EMBEDDING_MODEL, 256)  # dimension
    assert base != embedding_content_hash(catalog, "other-model", 768)     # model
    changed = (
        catalog[0].model_copy(update={"description": catalog[0].description + " More detail."}),
    ) + tuple(catalog[1:])
    assert base != embedding_content_hash(changed, EMBEDDING_MODEL, 768)   # content


def test_hybrid_fingerprint_includes_weights(catalog, embedder, cache, guides):
    a = HybridRetriever(catalog, embedder, cache, guides, w_semantic=0.6, w_lexical=0.4)
    b = HybridRetriever(catalog, embedder, cache, guides, w_semantic=0.7, w_lexical=0.3)
    assert a.index_fingerprint != b.index_fingerprint


def test_hybrid_retrieval_is_deterministic(catalog, embedder, cache, guides):
    retriever = HybridRetriever(catalog, embedder, cache, guides)
    first = retriever.search("rainy day melancholy piano", k=5)
    second = retriever.search("rainy day melancholy piano", k=5)
    assert [h.track.id for h in first.hits] == [h.track.id for h in second.hits]


def test_query_cache_serves_cached_queries_and_falls_back(tmp_path, embedder):
    query_cache = QueryCache(
        embedder.model_id,
        embedder.dimension,
        {query: embedder.embed_query(query) for query in EXAMPLE_QUERIES},
    )
    path = tmp_path / "queries.json"
    save_query_cache(path, query_cache)
    restored = load_query_cache(path)

    cached_only = CachedQueryEmbedder(restored)
    assert cached_only.query_source(EXAMPLE_QUERIES[0]) is EmbeddingSource.CACHE
    assert cached_only.embed_query(EXAMPLE_QUERIES[0]) == query_cache.vectors[EXAMPLE_QUERIES[0]]
    with pytest.raises(RuntimeError):
        cached_only.embed_query("an uncached novel query")

    with_fallback = CachedQueryEmbedder(restored, fallback=embedder)
    assert with_fallback.query_source("novel") is EmbeddingSource.LOCAL
    assert len(with_fallback.embed_query("novel")) == embedder.dimension


def test_factory_rejects_same_model_caches_with_mismatched_dimensions(
    tmp_path, catalog, embedder, cache
):
    catalog_path = tmp_path / "catalog.json"
    query_path = tmp_path / "queries.json"
    save_embedding_cache(catalog_path, cache)
    wrong = FakeEmbedder(dimension=32, model_id=cache.embedding_model)
    save_query_cache(
        query_path,
        QueryCache(
            cache.embedding_model,
            wrong.dimension,
            {"music to concentrate": wrong.embed_query("music to concentrate")},
        ),
    )
    retriever = build_default_retriever(
        catalog,
        catalog_cache_path=str(catalog_path),
        query_cache_path=str(query_path),
    )
    result = retriever.search("music to concentrate", k=3)
    assert result.operating_mode is OperatingMode.LOCAL
