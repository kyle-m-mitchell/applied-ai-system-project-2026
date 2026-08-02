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
from pathlib import Path

from pydantic import Field

from src.companion import MusicCompanion
from src.contracts import (
    CatalogCapabilities,
    CatalogDescriptor,
    CatalogEdition,
    ContractModel,
    FieldCoverage,
    OperatingMode,
)
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

    catalog_id: str = Field(
        default="fictional", pattern=r"^(?:fictional|fma)$"
    )
    catalog_path: str | None = Field(default=None, min_length=1)
    guides_dir: str | None = Field(default=None, min_length=1)
    catalog_cache_path: str | None = None
    query_cache_path: str | None = None
    fma_local_full_path: str | None = None
    fma_local_full_manifest_path: str | None = None
    fma_lite_path: str | None = None
    fma_lite_manifest_path: str | None = None
    fma_release_url: str | None = None
    fma_release_manifest_path: str | None = None
    fma_release_cache_path: str | None = None
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

    if config.catalog_id == "fma":
        return _build_fma_companion(config, deps)

    if config.catalog_path is None or config.guides_dir is None:
        raise ValueError("fictional catalog builds require catalog_path and guides_dir")

    catalog = RecommendationService(load_songs(config.catalog_path)).catalog
    guides = load_context_guides(config.guides_dir)

    retriever = build_default_retriever(
        catalog,
        guides,
        catalog_cache_path=config.catalog_cache_path,
        query_cache_path=config.query_cache_path,
        live_embedder=deps.live_embedder if config.use_live_embedder else None,
    )
    # A separate provider-free path can still use an exact committed query
    # vector. Cache misses are expected local operation, not an outage.
    provider_free_retriever = build_default_retriever(
        catalog,
        guides,
        catalog_cache_path=config.catalog_cache_path,
        query_cache_path=config.query_cache_path,
        live_embedder=None,
        fallback_mode=OperatingMode.LOCAL,
    )

    sink = deps.event_sink
    if sink is None and config.event_log_path:
        sink = JsonlEventSink(config.event_log_path)

    return MusicCompanion(
        catalog,
        guides,
        default_retriever=retriever,
        local_retriever=provider_free_retriever,
        generator=deps.generator if config.use_generator else None,
        event_sink=sink,
        config_fingerprint=config.fingerprint(),
        catalog_descriptor=_fictional_descriptor(config.catalog_path, len(catalog)),
        catalog_artifact_source="bundled-fictional",
    )


def _fictional_descriptor(catalog_path: str, count: int) -> CatalogDescriptor:
    """Describe the immutable synthetic regression catalog honestly."""
    checksum = hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest()
    covered = (
        "primary_genre", "mood", "energy", "tempo_bpm", "valence",
        "danceability", "acousticness", "description", "instrumental", "explicit",
    )
    return CatalogDescriptor(
        catalog_id="fictional",
        artifact_id=f"fictional-v1-{checksum[:12]}",
        edition=CatalogEdition.FICTIONAL,
        schema_version="fictional-csv-v1",
        etl_version="generate-catalog-v1",
        source_checksum=checksum,
        artifact_checksum=checksum,
        accepted_count=count,
        licenses=("Synthetic course-project metadata",),
        attribution=("Cadence fictional regression catalog",),
        field_coverage=tuple(
            FieldCoverage(field_name=name, ratio=1.0) for name in covered
        ),
        capabilities=CatalogCapabilities(
            supported_filters=("instrumental_only", "exclude_explicit"),
            supported_features=(
                "genre", "mood", "energy", "tempo_bpm", "valence",
                "danceability", "acousticness",
            ),
            retrieval_methods=("tfidf", "cached_semantic", "structured_fusion"),
            context_guides=True,
            research=False,
        ),
    )


def _descriptor_from_manifest(manifest) -> CatalogDescriptor:
    """Adapt the stdlib ETL manifest into the shared immutable UI contract."""
    source_payload = "|".join(
        f"{name}:{digest}" for name, digest in sorted(manifest.source_sha256.items())
    )
    source_checksum = (
        hashlib.sha256(source_payload.encode("utf-8")).hexdigest()
        if source_payload
        else None
    )
    return CatalogDescriptor(
        catalog_id=manifest.catalog_id,
        artifact_id=manifest.artifact_id,
        edition=CatalogEdition(manifest.edition),
        schema_version=manifest.schema_version,
        etl_version=manifest.etl_version,
        source_checksum=source_checksum,
        artifact_checksum=manifest.artifact_sha256,
        accepted_count=manifest.accepted_count,
        quarantined_count=manifest.quarantined_count,
        licenses=tuple(manifest.licenses),
        attribution=(manifest.attribution,),
        field_coverage=manifest.field_coverage,
        capabilities=CatalogCapabilities(
            supported_filters=tuple(manifest.supported_filters),
            supported_features=tuple(manifest.supported_features),
            retrieval_methods=tuple(manifest.retrieval_methods),
            context_guides=manifest.context_guides,
            research=manifest.research,
        ),
        calibration_status=manifest.calibration_status,
    )


def _build_fma_companion(config: CompanionConfig, deps: CompanionDeps) -> MusicCompanion:
    """Resolve a verified full/lite artifact and wire the lazy SQLite path."""
    if config.fma_lite_path is None or config.fma_lite_manifest_path is None:
        raise ValueError("FMA builds require fma_lite_path and fma_lite_manifest_path")

    from src.catalog_artifacts import ArtifactCandidate, CatalogArtifactResolver
    from src.fma_retrieval import FmaRetriever
    from src.fma_store import FmaCatalogStore

    local_values = (config.fma_local_full_path, config.fma_local_full_manifest_path)
    if any(value is not None for value in local_values) and not all(
        value is not None for value in local_values
    ):
        raise ValueError("both local full FMA paths must be supplied together")
    local = (
        ArtifactCandidate(
            Path(config.fma_local_full_path),
            Path(config.fma_local_full_manifest_path),
            "local-full",
        )
        if all(value is not None for value in local_values)
        else None
    )
    lite = ArtifactCandidate(
        Path(config.fma_lite_path),
        Path(config.fma_lite_manifest_path),
        "bundled-lite",
    )
    resolved = CatalogArtifactResolver(
        local_full=local,
        bundled_lite=lite,
        release_url=config.fma_release_url,
        release_manifest_path=config.fma_release_manifest_path,
        release_cache_path=config.fma_release_cache_path,
    ).resolve()
    store = FmaCatalogStore(resolved.database_path, resolved.manifest_path)
    retriever = FmaRetriever(store)
    vocabulary = store.vocabulary()

    sink = deps.event_sink
    if sink is None and config.event_log_path:
        sink = JsonlEventSink(config.event_log_path)

    return MusicCompanion(
        (),
        (),
        default_retriever=retriever,
        local_retriever=retriever,
        generator=deps.generator if config.use_generator else None,
        event_sink=sink,
        config_fingerprint=config.fingerprint(),
        catalog_descriptor=_descriptor_from_manifest(resolved.manifest),
        valid_ids=store.valid_track_ids(),
        valid_genres=vocabulary.genres,
        valid_moods=vocabulary.mood_labels,
        catalog_artifact_source=resolved.source,
        catalog_warnings=resolved.warnings,
    )
