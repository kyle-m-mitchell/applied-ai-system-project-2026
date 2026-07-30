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
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embeddings import (  # noqa: E402
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


def _embed_in_chunks(embedder, texts, chunk_size=32, pause=0.5):
    """Embed documents in throttled chunks to respect rate limits."""
    vectors = []
    for start in range(0, len(texts), chunk_size):
        vectors.extend(embedder.embed_documents(texts[start : start + chunk_size]))
        done = min(start + chunk_size, len(texts))
        print(f"  embedded {done}/{len(texts)} tracks")
        if done < len(texts):
            time.sleep(pause)
    return vectors


def main() -> None:
    _load_dotenv(REPO_ROOT / ".env")
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    tracks = RecommendationService(load_songs(str(CATALOG_PATH))).catalog
    embedder = GeminiEmbedder()  # raises a clear error if GEMINI_API_KEY is unset
    print(f"Embedding {len(tracks)} tracks with {embedder.model_id} @ {embedder.dimension}-d")

    documents = [build_document_text(track) for track in tracks]
    track_vectors = _embed_in_chunks(embedder, documents)
    catalog_cache = EmbeddingCache(
        embedding_model=embedder.model_id,
        dimension=embedder.dimension,
        content_hash=embedding_content_hash(tracks, embedder.model_id, embedder.dimension),
        vectors={track.id: vector for track, vector in zip(tracks, track_vectors)},
    )
    save_embedding_cache(CATALOG_CACHE, catalog_cache)
    print(f"Wrote {CATALOG_CACHE.relative_to(REPO_ROOT)}")

    print(f"Embedding {len(EXAMPLE_QUERIES)} example queries")
    query_cache = QueryCache(
        embedding_model=embedder.model_id,
        dimension=embedder.dimension,
        vectors={query: embedder.embed_query(query) for query in EXAMPLE_QUERIES},
    )
    save_query_cache(QUERY_CACHE, query_cache)
    print(f"Wrote {QUERY_CACHE.relative_to(REPO_ROOT)}")
    print("Done. Commit data/embeddings/ to make the semantic index reproducible.")


if __name__ == "__main__":
    main()
