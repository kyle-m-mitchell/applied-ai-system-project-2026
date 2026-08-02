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

# Make the app runnable both as a module (`python -m src.main`) and as a script
# (`python src/main.py` or the IDE run button) by ensuring the repo root — which
# holds the importable `src` package — is on the path before the `src` imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.companion import MusicCompanion  # noqa: E402
from src.contracts import (  # noqa: E402
    CompanionResponse,
    RecommendationRequest,
    RecommendationResult,
    ExecutionPolicy,
)
from src.factory import CompanionConfig, CompanionDeps, build_companion  # noqa: E402
from src.recommender import load_songs  # noqa: E402
from src.service import RecommendationService  # noqa: E402


CATALOG_PATH = REPO_ROOT / "data" / "songs.csv"
GUIDES_DIR = REPO_ROOT / "data" / "context_guides"
CATALOG_CACHE = REPO_ROOT / "data" / "embeddings" / "catalog.json"
QUERY_CACHE = REPO_ROOT / "data" / "embeddings" / "queries.json"
EVENT_LOG = REPO_ROOT / "logs" / "events.jsonl"
DIVIDER = "-" * 64


def _provider_disabled() -> bool:
    """Honor the documented operator kill switch before constructing clients."""
    return os.environ.get("CADENCE_DISABLE_PROVIDER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
    if _provider_disabled() or not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        from src.embeddings import GeminiEmbedder

        return GeminiEmbedder()
    except Exception:  # noqa: BLE001 - any setup failure means "no live embedder"
        return None


def _text_generator():
    """A live Gemini text generator for Cadence's voice if a key is present."""
    if _provider_disabled() or not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        from src.generation import GeminiTextGenerator

        return GeminiTextGenerator()
    except Exception:  # noqa: BLE001 - any setup failure means "no live voice"
        return None


def _build_companion(
    *, log_events: bool = False, provider_enabled: bool = True
) -> MusicCompanion:
    """Build the CLI's companion through the shared factory — one construction path.

    Live provider objects are supplied as deps (``None`` without a key); the
    config's toggles decide whether they're used, so the config fingerprint the
    receipt records always reflects what actually ran.
    """
    _load_dotenv()
    allow_provider = provider_enabled and not _provider_disabled()
    embedder = _live_embedder() if allow_provider else None
    generator = _text_generator() if allow_provider else None
    config = CompanionConfig(
        catalog_path=str(CATALOG_PATH),
        guides_dir=str(GUIDES_DIR),
        catalog_cache_path=str(CATALOG_CACHE),
        query_cache_path=str(QUERY_CACHE),
        use_live_embedder=embedder is not None,
        use_generator=generator is not None,
        event_log_path=str(EVENT_LOG) if log_events else None,
    )
    deps = CompanionDeps(live_embedder=embedder, generator=generator)
    return build_companion(config, deps)


def print_companion_response(
    response: CompanionResponse, *, show_trace: bool = False
) -> None:
    """Print Cadence's voiced response plus a compact, privacy-safe trace line.

    Echoes only the *sanitized* query (PII/secrets already redacted), never the
    raw input — so captured terminal output can't retain an email or key.
    """
    shown = response.intent.query if response.intent is not None else ""
    if shown:
        print(f'\n🎧  You asked: "{shown}"\n')
    else:
        print("\n🎧  Cadence\n")
    print(response.message)

    trace = response.trace
    result = response.retrieval
    meta = f"[{response.action.value}]"
    if result is not None:
        meta += f"  ·  mode: {result.operating_mode.value}"
    if trace is not None:
        meta += f"  ·  voice: {trace.voice_source.value}"
        if trace.diversity_applied:
            meta += "  ·  diversified"
    print("\n" + meta)

    if result is not None:
        for guide in result.guides_used:
            print(f"context guide: {guide.title} → added {', '.join(guide.expansion_terms)}")
    if show_trace and trace is not None:
        print("trace: " + str(trace.model_dump()))


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
    flags = {"--trace", "--log", "--local-only"}
    args = [arg for arg in sys.argv[1:] if arg not in flags]
    show_trace = "--trace" in sys.argv[1:]
    log_events = "--log" in sys.argv[1:]  # opt-in privacy-safe receipt (see logs/events.jsonl)
    local_only = "--local-only" in sys.argv[1:]
    if args:
        query = " ".join(args)
        companion = _build_companion(
            log_events=log_events, provider_enabled=not local_only
        )
        print_companion_response(
            companion.respond(query, policy=ExecutionPolicy(force_local=local_only)),
            show_trace=show_trace,
        )
    else:
        run_structured_demo()


if __name__ == "__main__":
    main()
