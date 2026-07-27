"""Show local TF-IDF retrieval next to the original numeric scorer.

This is Feature 3's before/after evidence. It does *not* touch the public
``recommend()`` path or add a natural-language field to the request contract;
it exercises the standalone :class:`~src.retrieval.TfidfRetriever` directly, the
way the intent-parsing feature will later call it.

Run from any directory::

    python scripts/retrieval_demo.py "late-night study beats to focus"
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.contracts import RecommendationRequest  # noqa: E402
from src.recommender import load_songs  # noqa: E402
from src.retrieval import TfidfRetriever  # noqa: E402
from src.service import RecommendationService  # noqa: E402


DEFAULT_QUERY = "late-night study beats to focus"
CATALOG_PATH = REPO_ROOT / "data" / "songs.csv"
RULE = "=" * 68


def show_baseline(service: RecommendationService, query: str) -> None:
    """Illustrate that the numeric scorer has no home for a free-text phrase."""
    print("BEFORE - original numeric scorer")
    print(
        "  The public request accepts only structured preferences (genre, mood,\n"
        "  numeric features) and rejects free text. A phrase has nowhere to go.\n"
        "  Placed in the only text slot (genre), it matches no known label, so\n"
        "  ranking falls back to stable ID order with zero match strength:\n"
    )
    result = service.recommend(RecommendationRequest(genre=query, limit=5))
    for item in result.recommendations:
        print(
            f"  #{item.track.id:>3}  {item.track.title[:26]:26} "
            f"[{item.track.genre:9}] match strength {item.match_strength:.2f}"
        )


def show_retrieval(retriever: TfidfRetriever, query: str) -> None:
    """Show what TF-IDF retrieval surfaces, with provenance for each hit."""
    result = retriever.search(query, k=5)
    print(f"AFTER - TF-IDF retriever  (mode: local, index {result.index_fingerprint[:12]})")
    if not result.hits:
        print("  (no lexical overlap with the catalog - retriever reports no signal)")
        return
    print()
    for hit in result.hits:
        print(
            f"  #{hit.track.id:>3}  {hit.track.title[:26]:26} "
            f"[{hit.track.genre:9}] similarity {hit.score:.3f}"
        )
        print(
            f"        source={hit.source_id}  matched: {', '.join(hit.matched_terms[:5])}"
        )


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY

    service = RecommendationService(load_songs(str(CATALOG_PATH)))
    retriever = TfidfRetriever(service.catalog)

    print(RULE)
    print(" Feature 3 - Local TF-IDF Retrieval (before / after)")
    print(RULE)
    print(f'Query: "{query}"\n')

    show_baseline(service, query)
    print()
    show_retrieval(retriever, query)

    print()
    print(
        "Note: TF-IDF is lexical, not semantic. It matches word forms, so\n"
        '"studying" would not match the catalog\'s "study". Closing that gap is\n'
        "the job of the provider-embedding retriever planned behind this same\n"
        "Retriever interface."
    )


if __name__ == "__main__":
    main()
