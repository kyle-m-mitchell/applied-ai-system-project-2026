"""Command line runner for the Music Recommender Simulation.

Two entry points share the one validated catalog:

- no argument      -> the original structured-preference scorer (unchanged);
- a quoted phrase  -> the natural-language companion
                      (guard -> intent -> retrieval), e.g.::

      python -m src.main "clean chill beats for studying, no vocals"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.companion import MusicCompanion
from src.contracts import CompanionResponse, RecommendationRequest, RecommendationResult
from src.recommender import load_songs
from src.retrieval import build_default_retriever, load_context_guides
from src.service import RecommendationService


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "data" / "songs.csv"
GUIDES_DIR = REPO_ROOT / "data" / "context_guides"
CATALOG_CACHE = REPO_ROOT / "data" / "embeddings" / "catalog.json"
QUERY_CACHE = REPO_ROOT / "data" / "embeddings" / "queries.json"
DIVIDER = "-" * 64


def format_request(request: RecommendationRequest) -> str:
    """Render supplied preferences as a compact, single-line summary."""
    values = request.model_dump(exclude_none=True, exclude={"limit"})
    return ", ".join(f"{key}={value}" for key, value in values.items())


def print_recommendations(result: RecommendationResult) -> None:
    """Print structured-scorer recommendations in a clean terminal layout."""
    print("\n🎵  Music Recommender — your top picks\n")
    print(f"Taste profile: {format_request(result.request)}")
    print(f"Operating mode: {result.operating_mode.value}")
    print(DIVIDER)
    for rank, item in enumerate(result.recommendations, start=1):
        track = item.track
        print(f"{rank}. {track.title} — {track.artist}  [{track.genre} · {track.mood}]")
        print(
            f"   Raw score: {item.raw_score:.2f}  ·  "
            f"Match strength: {item.match_strength:.0%}"
        )
        print("   Why:")
        for reason in item.reasons:
            print(f"     • {reason}")
        print(DIVIDER)
    for warning in result.warnings:
        print(f"Note: {warning}")


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
    """A live Gemini embedder if a key is present, else None."""
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        from src.embeddings import GeminiEmbedder

        return GeminiEmbedder()
    except Exception:  # noqa: BLE001 - any setup failure means "no live embedder"
        return None


def _build_companion(catalog, guides) -> MusicCompanion:
    default = build_default_retriever(
        catalog,
        guides,
        catalog_cache_path=str(CATALOG_CACHE),
        query_cache_path=str(QUERY_CACHE),
        live_embedder=_live_embedder(),
    )
    return MusicCompanion(catalog, guides, default_retriever=default)


def print_companion_response(query: str, response: CompanionResponse) -> None:
    """Print a natural-language companion response with evidence."""
    print(f'\n🎧  You asked: "{query}"\n')
    print(f"Cadence [{response.action.value}]: {response.message}")
    result = response.retrieval
    if result is None:
        return
    print(f"Operating mode: {result.operating_mode.value}")
    print(DIVIDER)
    for rank, hit in enumerate(result.hits, start=1):
        track = hit.track
        print(f"{rank}. {track.title} — {track.artist}  [{track.genre} · {track.mood}]")
        why = []
        if hit.semantic_score is not None:
            why.append(f"semantic {hit.semantic_score:.2f}")
        if hit.lexical_score is not None:
            why.append(f"lexical {hit.lexical_score:.2f}")
        if hit.matched_terms:
            why.append("matched: " + ", ".join(hit.matched_terms[:4]))
        print(f"   score {hit.score:.3f}" + (("  ·  " + "  ·  ".join(why)) if why else ""))
        print(DIVIDER)
    for guide in result.guides_used:
        print(f"context guide: {guide.title} → added {', '.join(guide.expansion_terms)}")


def run_structured_demo() -> None:
    """The original hard-coded structured-preference run."""
    service = RecommendationService(load_songs(str(CATALOG_PATH)))
    request = RecommendationRequest(
        genre="lofi",
        mood="chill",
        energy=0.40,
        acousticness=0.80,
        valence=0.55,
        danceability=0.40,
        tempo_bpm=78,  # A relaxed study-beat tempo.
        limit=5,
    )
    print_recommendations(service.recommend(request))


def main() -> None:
    if len(sys.argv) > 1:
        _load_dotenv()
        query = " ".join(sys.argv[1:])
        catalog = RecommendationService(load_songs(str(CATALOG_PATH))).catalog
        guides = load_context_guides(str(GUIDES_DIR))
        companion = _build_companion(catalog, guides)
        print_companion_response(query, companion.respond(query))
    else:
        run_structured_demo()


if __name__ == "__main__":
    main()
