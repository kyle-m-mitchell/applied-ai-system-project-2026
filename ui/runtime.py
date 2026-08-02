"""Session-safe construction of the stateless Cadence engine for Streamlit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from src.companion import MusicCompanion
from src.embeddings import GeminiEmbedder
from src.factory import CompanionConfig, CompanionDeps, build_companion
from src.generation import GeminiTextGenerator


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RuntimeBundle:
    """The stateless engine plus a boolean capability disclosure (never the key)."""

    companion: MusicCompanion
    provider_configured: bool


def _dotenv_key() -> str | None:
    """Read only GEMINI_API_KEY from the ignored local file; never mutate env."""
    path = ROOT / ".env"
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "GEMINI_API_KEY":
            candidate = value.strip().strip('"').strip("'")
            return candidate or None
    return None


def _provider_key() -> str | None:
    """Resolve a secret from Cloud settings, environment, or ignored local file."""
    if os.environ.get("CADENCE_DISABLE_PROVIDER") == "1":
        return None  # operational/test kill switch; guarantees no provider objects
    try:
        secret = st.secrets.get("GEMINI_API_KEY")
        if secret:
            return str(secret)
    except Exception:  # no secrets file is the normal local/offline case
        pass
    return os.environ.get("GEMINI_API_KEY") or _dotenv_key()


@st.cache_resource(show_spinner=False)
def get_runtime() -> RuntimeBundle:
    """Build one audited stateless engine shared across sessions.

    The companion stores immutable indexes and provider clients only.  Per-user
    intents, turns, history, receipts, and feedback live in ``st.session_state``;
    the event sink remains the default ``NullEventSink`` so no shared file is read
    or written by the UI.
    """
    key = _provider_key()
    embedder = None
    generator = None
    if key:
        try:
            embedder = GeminiEmbedder(api_key=key, timeout=8.0, max_retries=0)
        except Exception:
            embedder = None
        try:
            generator = GeminiTextGenerator(api_key=key, timeout=8.0, max_retries=0)
        except Exception:
            generator = None

    config = CompanionConfig(
        catalog_path=str(ROOT / "data" / "songs.csv"),
        guides_dir=str(ROOT / "data" / "context_guides"),
        catalog_cache_path=str(ROOT / "data" / "embeddings" / "catalog.json"),
        query_cache_path=str(ROOT / "data" / "embeddings" / "queries.json"),
        use_live_embedder=embedder is not None,
        use_generator=generator is not None,
        event_log_path=None,
    )
    companion = build_companion(
        config,
        CompanionDeps(live_embedder=embedder, generator=generator),
    )
    return RuntimeBundle(
        companion=companion,
        provider_configured=embedder is not None or generator is not None,
    )
