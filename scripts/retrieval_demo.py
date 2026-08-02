"""Show local retrieval evolving: lexical, multi-source, and semantic.

This is the before/after evidence for Features 3, 3b, and 4. It does *not* touch
the public ``recommend()`` path or add a natural-language field to the request
contract; it exercises the standalone retrievers directly, the way the
intent-parsing feature will later call them.

The semantic panel reads the committed embedding caches, so it reproduces offline
for the example queries. Without a cache (or a key for a novel query) it degrades
honestly to TF-IDF and says so.

Run from any directory::

    python scripts/retrieval_demo.py "music to concentrate"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.contracts import RecommendationRequest  # noqa: E402
from src.embeddings import (  # noqa: E402
    CachedQueryEmbedder,
    FakeEmbedder,
    load_embedding_cache,
    load_query_cache,
)
from src.recommender import load_songs  # noqa: E402
from src.retrieval import HybridRetriever, TfidfRetriever, load_context_guides  # noqa: E402
from src.service import RecommendationService  # noqa: E402


DEFAULT_QUERY = "music to concentrate"
CATALOG_PATH = REPO_ROOT / "data" / "songs.csv"
GUIDES_DIR = REPO_ROOT / "data" / "context_guides"
CATALOG_CACHE = REPO_ROOT / "data" / "embeddings" / "catalog.json"
QUERY_CACHE = REPO_ROOT / "data" / "embeddings" / "queries.json"
RULE = "=" * 68


def _load_dotenv() -> None:
    """Populate os.environ from a git-ignored .env at the repo root, if present."""
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for raw in env.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _live_embedder():
    """Return a live Gemini embedder if a key is set, else None."""
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        from src.embeddings import GeminiEmbedder

        return GeminiEmbedder()
    except Exception:  # noqa: BLE001 - any setup failure means "no live embedder"
        return None


def _build_hybrid(tracks, guides) -> HybridRetriever:
    """Wire a hybrid retriever from the committed caches, or one that degrades.

    Semantic search runs only when a query embedder consistent with the cached
    model is available (cached example queries, or a live key). Otherwise the
    hybrid is built with no cache so it degrades honestly to TF-IDF.
    """
    cache = load_embedding_cache(CATALOG_CACHE) if CATALOG_CACHE.exists() else None
    embedder = None
    if cache is not None:
        live = _live_embedder()
        if QUERY_CACHE.exists():
            query_cache = load_query_cache(QUERY_CACHE)
            if query_cache.embedding_model == cache.embedding_model:
                embedder = CachedQueryEmbedder(query_cache, fallback=live)
        if embedder is None and live is not None and live.model_id == cache.embedding_model:
            embedder = live
    if cache is None or embedder is None:
        return HybridRetriever(tracks, FakeEmbedder(), None, guides)
    return HybridRetriever(tracks, embedder, cache, guides)


def show_baseline(service: RecommendationService, query: str) -> None:
    """Illustrate that the numeric scorer has no home for a free-text phrase."""
    print("BEFORE - original numeric scorer")
    print(
        "  The public request accepts only structured preferences and rejects\n"
        "  free text. Placed in the only text slot (genre), the phrase matches no\n"
        "  known label, so ranking falls back to stable ID order:\n"
    )
    result = service.recommend(RecommendationRequest(genre=query, limit=5))
    for item in result.recommendations:
        print(
            f"  #{item.track.id:>3}  {item.track.title[:26]:26} "
            f"[{item.track.genre:9}] match strength {item.match_strength:.2f}"
        )


def show_retrieval(retriever: TfidfRetriever, query: str, *, use_guides: bool, label: str) -> None:
    """Show what TF-IDF retrieval surfaces, with provenance for each hit."""
    result = retriever.search(query, k=5, use_guides=use_guides)
    print(f"{label}  (mode: {result.operating_mode.value}, index {result.index_fingerprint[:12]})")
    for guide in result.guides_used:
        print(
            f"  guide fired: {guide.title!r}  (score {guide.score:.3f})  "
            f"expanded query with: {', '.join(guide.expansion_terms)}"
        )
    if not result.hits:
        print("  (no lexical overlap with the catalog - retriever reports no signal)")
        return
    for hit in result.hits:
        print(
            f"  #{hit.track.id:>3}  {hit.track.title[:26]:26} "
            f"[{hit.track.genre:9}] similarity {hit.score:.3f}"
        )


def show_hybrid(retriever: HybridRetriever, query: str) -> None:
    """Show the semantic + lexical hybrid, with both sub-scores per hit."""
    result = retriever.search(query, k=5)
    mode = result.operating_mode.value
    print(f"PLUS  - semantic + lexical hybrid  (mode: {mode}, index {result.index_fingerprint[:12]})")
    if mode == "degraded":
        print("  (no embedding cache or no key -> degraded to TF-IDF;")
        print("   run `python scripts/build_embeddings.py` with a key to enable semantics)")
    for hit in result.hits:
        semantic = f"{hit.semantic_score:.3f}" if hit.semantic_score is not None else "  -  "
        lexical = f"{hit.lexical_score:.3f}" if hit.lexical_score is not None else "  -  "
        print(
            f"  #{hit.track.id:>3}  {hit.track.title[:26]:26} "
            f"[{hit.track.genre:9}] score {hit.score:.3f}  (sem {semantic} | lex {lexical})"
        )


def main() -> None:
    _load_dotenv()
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY

    service = RecommendationService(load_songs(str(CATALOG_PATH)))
    guides = load_context_guides(str(GUIDES_DIR))
    tfidf = TfidfRetriever(service.catalog, guides)
    hybrid = _build_hybrid(service.catalog, guides)

    print(RULE)
    print(" Features 3 / 3b / 4 - Local retrieval: lexical, multi-source, semantic")
    print(RULE)
    print(f'Query: "{query}"\n')

    show_baseline(service, query)
    print()
    show_retrieval(tfidf, query, use_guides=False, label="THEN  - TF-IDF over the catalog alone")
    print()
    show_retrieval(tfidf, query, use_guides=True, label="NEXT  - + versioned context guides (query expansion)")
    print()
    show_hybrid(hybrid, query)

    print()
    print(
        "Note: TF-IDF is lexical (word forms only); context guides bridge a vague\n"
        "word to catalog vocabulary; embeddings add meaning, so paraphrases match\n"
        "even with no shared words. All three sit behind one Retriever interface,\n"
        "and the hybrid degrades to TF-IDF (labeled) whenever Gemini is unavailable."
    )


if __name__ == "__main__":
    main()
