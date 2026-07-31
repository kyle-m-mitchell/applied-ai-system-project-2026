"""Build and commit the Gemini embedding caches (run once per catalog change).

This is the only script that calls the real embedding API. It writes two files:

* ``data/embeddings/catalog.json`` — one vector per track, keyed by the catalog
  content plus the model and dimension; and
* ``data/embeddings/queries.json`` — vectors for a fixed set of example queries,
  so the semantic demo reproduces offline with no key.

Commit both so anyone can reproduce the exact semantic index without a key.

Usage (with a rotated key in a git-ignored ``.env`` at the repo root)::

    python scripts/build_embeddings.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embeddings import (  # noqa: E402
    EMBEDDING_MODEL,
    EXAMPLE_QUERIES,
    EmbeddingCache,
    GeminiEmbedder,
    QueryCache,
    save_embedding_cache,
    save_query_cache,
)
from src.recommender import load_songs  # noqa: E402
from src.retrieval import build_document_text, embedding_content_hash  # noqa: E402
from src.service import RecommendationService  # noqa: E402


CATALOG_PATH = REPO_ROOT / "data" / "songs.csv"
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"
CATALOG_CACHE = EMBEDDINGS_DIR / "catalog.json"
QUERY_CACHE = EMBEDDINGS_DIR / "queries.json"


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file, if present."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _fail(message: str) -> None:
    print(f"\nError: {message}")
    print("Checklist:")
    print("  - 403/401 -> GEMINI_API_KEY in .env is missing, invalid, or unauthorized.")
    print("  - 429     -> rate limit / quota; progress is saved, just rerun to resume.")
    print(f"  - 404     -> model name may not exist for your key. Current is")
    print(f"               '{EMBEDDING_MODEL}'; set EMBEDDING_MODEL in src/embeddings.py.")
    sys.exit(1)


def main() -> None:
    _load_dotenv(REPO_ROOT / ".env")
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    tracks = RecommendationService(load_songs(str(CATALOG_PATH))).catalog
    try:
        embedder = GeminiEmbedder(request_delay=0.7)  # gentle throttle under the RPM cap
    except RuntimeError as exc:
        _fail(str(exc))
        return
    model, dimension = embedder.model_id, embedder.dimension
    content_hash = embedding_content_hash(tracks, model, dimension)

    # Resume: reuse a matching cache so a rerun only embeds what is missing.
    vectors: dict[int, tuple] = {}
    if CATALOG_CACHE.exists():
        try:
            previous = load_embedding_cache(CATALOG_CACHE)
            if (previous.embedding_model, previous.dimension, previous.content_hash) == (
                model,
                dimension,
                content_hash,
            ):
                vectors = dict(previous.vectors)
        except Exception:  # noqa: BLE001 - a bad cache is simply rebuilt
            vectors = {}

    def _save_catalog() -> None:
        save_embedding_cache(
            CATALOG_CACHE, EmbeddingCache(model, dimension, content_hash, vectors)
        )

    pending = [track for track in tracks if track.id not in vectors]
    print(
        f"Embedding {len(pending)} of {len(tracks)} tracks with {model} @ {dimension}-d "
        f"({len(vectors)} already cached)"
    )
    try:
        for done, track in enumerate(pending, 1):
            vectors[track.id] = embedder.embed_documents([build_document_text(track)])[0]
            if done % 16 == 0 or done == len(pending):
                _save_catalog()
                print(f"  embedded {done}/{len(pending)} (saved)")
    except RuntimeError as exc:
        _save_catalog()
        _fail(f"{exc}  [saved {len(vectors)}/{len(tracks)}; rerun to resume]")
        return
    _save_catalog()
    print(f"Wrote {CATALOG_CACHE.relative_to(REPO_ROOT)} ({len(vectors)} vectors)")

    # Example query cache (small), resumed the same way.
    query_vectors: dict[str, tuple] = {}
    if QUERY_CACHE.exists():
        try:
            previous_q = load_query_cache(QUERY_CACHE)
            if (previous_q.embedding_model, previous_q.dimension) == (model, dimension):
                query_vectors = dict(previous_q.vectors)
        except Exception:  # noqa: BLE001
            query_vectors = {}
    try:
        for query in EXAMPLE_QUERIES:
            if query not in query_vectors:
                query_vectors[query] = embedder.embed_query(query)
    except RuntimeError as exc:
        save_query_cache(QUERY_CACHE, QueryCache(model, dimension, query_vectors))
        _fail(f"{exc}  [query progress saved; rerun to resume]")
        return
    save_query_cache(QUERY_CACHE, QueryCache(model, dimension, query_vectors))
    print(f"Wrote {QUERY_CACHE.relative_to(REPO_ROOT)}")
    print("Done. Commit data/embeddings/ to make the semantic index reproducible.")


if __name__ == "__main__":
    main()
