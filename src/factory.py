"""One public construction path for the companion.

``build_companion(config, deps)`` is the single place that turns a declarative
:class:`CompanionConfig` (paths and feature toggles — reproducible, hashable)
plus runtime :class:`CompanionDeps` (live provider objects, an event sink) into a
wired :class:`~src.companion.MusicCompanion`. The CLI, the coming UI, and any
script share it, so "which retriever, which generator, which log" is decided in
exactly one place — and ``config.fingerprint()`` identifies that decision in the
event log.

Config is the *what* (serializable, safe to hash and record); deps is the *how*
(objects that can't be serialized — an embedder, a generator, a sink). Keeping
them apart is what lets a receipt carry a config fingerprint without ever
touching a secret or a live connection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pydantic import Field

from src.companion import MusicCompanion
from src.contracts import ContractModel
from src.embeddings import Embedder
from src.generation import TextGenerator
from src.observability import EventSink, JsonlEventSink
from src.recommender import load_songs
from src.retrieval import build_default_retriever, load_context_guides
from src.service import RecommendationService


class CompanionConfig(ContractModel):
    """Declarative, reproducible companion configuration.

    Every field actually changes the build, so hashing it (``fingerprint``) yields
    an identifier that means "these exact settings produced this run." Fusion
    weights are deliberately absent: they live in the retriever until the
    structured leg calibrates them, so the config never claims a knob it can't turn.
    """

    catalog_path: str = Field(min_length=1)
    guides_dir: str = Field(min_length=1)
    catalog_cache_path: str | None = None
    query_cache_path: str | None = None
    use_live_embedder: bool = False
    use_generator: bool = False
    event_log_path: str | None = None

    def fingerprint(self) -> str:
        """A short, stable hash of these settings, for the event log and reports."""
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CompanionDeps:
    """Runtime dependencies that can't be serialized into config.

    Held apart from :class:`CompanionConfig` so the config stays hashable and
    secret-free. A dep is used only when the matching config toggle is on, so the
    fingerprint always reflects what actually ran.
    """

    live_embedder: Embedder | None = None
    generator: TextGenerator | None = None
    event_sink: EventSink | None = None


def build_companion(
    config: CompanionConfig, deps: CompanionDeps | None = None
) -> MusicCompanion:
    """Build a wired companion from declarative config plus runtime deps."""
    deps = deps or CompanionDeps()

    catalog = RecommendationService(load_songs(config.catalog_path)).catalog
    guides = load_context_guides(config.guides_dir)

    retriever = build_default_retriever(
        catalog,
        guides,
        catalog_cache_path=config.catalog_cache_path,
        query_cache_path=config.query_cache_path,
        live_embedder=deps.live_embedder if config.use_live_embedder else None,
    )

    sink = deps.event_sink
    if sink is None and config.event_log_path:
        sink = JsonlEventSink(config.event_log_path)

    return MusicCompanion(
        catalog,
        guides,
        default_retriever=retriever,
        generator=deps.generator if config.use_generator else None,
        event_sink=sink,
        config_fingerprint=config.fingerprint(),
    )
