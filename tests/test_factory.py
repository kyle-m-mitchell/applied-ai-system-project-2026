"""Tests for the public build_companion factory (one construction path)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.companion import MusicCompanion
from src.factory import CompanionConfig, CompanionDeps, build_companion
from src.recommender import load_songs
from src.retrieval import load_context_guides
from src.service import RecommendationService

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "songs.csv"
GUIDES_DIR = ROOT / "data" / "context_guides"


@pytest.fixture(scope="module")
def catalog():
    return RecommendationService(load_songs(str(CATALOG_PATH))).catalog


@pytest.fixture(scope="module")
def guides():
    return load_context_guides(str(GUIDES_DIR))


def _local_config() -> CompanionConfig:
    # No cache paths -> the factory builds the local TF-IDF retriever, matching a
    # bare MusicCompanion(catalog, guides).
    return CompanionConfig(catalog_path=str(CATALOG_PATH), guides_dir=str(GUIDES_DIR))


def test_config_fingerprint_is_stable_and_sensitive():
    a = _local_config()
    b = _local_config()
    assert a.fingerprint() == b.fingerprint()          # deterministic
    changed = a.model_copy(update={"use_generator": True})
    assert changed.fingerprint() != a.fingerprint()    # a real change shows up


def test_build_companion_returns_working_companion():
    companion = build_companion(_local_config(), CompanionDeps())
    assert isinstance(companion, MusicCompanion)
    response = companion.respond("some jazz please")
    assert response.action.value in {"recommend", "degraded"}


def test_factory_reproduces_direct_construction(catalog, guides):
    # The Phase 2 reproduction gate: text-only queries must return the SAME
    # recommendation ids whether the companion is built by the factory or
    # constructed directly. The foundation adds structure without moving results.
    factory_companion = build_companion(_local_config())
    direct_companion = MusicCompanion(catalog, guides)

    for query in [
        "clean instrumental lofi for studying",
        "some jazz please",
        "high energy workout",
        "music to concentrate",
        "upbeat party music",
    ]:
        factory_ids = _recommended_ids(factory_companion, query)
        direct_ids = _recommended_ids(direct_companion, query)
        assert factory_ids == direct_ids, f"factory drifted from direct on: {query!r}"


def _recommended_ids(companion: MusicCompanion, query: str) -> tuple[int, ...]:
    response = companion.respond(query)
    if response.retrieval is None:
        return ()
    return tuple(hit.track.id for hit in response.retrieval.hits)
