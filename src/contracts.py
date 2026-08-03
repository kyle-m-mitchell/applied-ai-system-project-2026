"""Validated data contracts shared by every application interface.

Type hints describe what developers intend. Pydantic models also enforce that
intent at runtime, which makes these contracts our first reliability layer.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContractModel(BaseModel):
    """Strict, immutable defaults shared by the project's public contracts."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class OperatingMode(str, Enum):
    """How the recommendation was produced."""

    LOCAL = "local"
    GEMINI = "gemini"
    DEGRADED = "degraded"


class EmbeddingSource(str, Enum):
    """Where the query vector came from for this retrieval turn.

    ``OperatingMode.GEMINI`` historically meant that Gemini-built vectors were
    involved, but it could not distinguish an offline committed query vector
    from a live network request.  The UI and event receipts need that distinction
    to make an honest privacy claim.
    """

    CACHE = "cache"
    LIVE = "live"
    LOCAL = "local"  # deterministic fake/test embedder; never a network call


class FieldOrigin(str, Enum):
    """How one catalog field came to exist.

    A value's origin is deliberately separate from the value itself. In
    particular, an estimate produced from audio features must never become
    indistinguishable from artist-authored metadata or an Echo Nest-computed
    feature.
    """

    AUTHORED = "authored"
    ARTIST_SUPPLIED = "artist_supplied"
    FMA_METADATA = "fma_metadata"
    LIBROSA_COMPUTED = "librosa_computed"
    ECHONEST_COMPUTED = "echonest_computed"
    MODEL_ESTIMATED = "model_estimated"
    DETERMINISTIC_DERIVED = "deterministic_derived"
    UNKNOWN = "unknown"


class TrackRef(ContractModel):
    """A catalog-qualified track identifier.

    Local integer IDs are only unique *inside* one catalog. Persisted receipts,
    research results, and cross-catalog UI state use this reference so FMA track
    ``1`` can never be confused with fictional track ``1``.
    """

    catalog_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    track_id: int = Field(gt=0)
    external_id: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("catalog_id", mode="before")
    @classmethod
    def normalize_catalog_id(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("track_id", mode="before")
    @classmethod
    def reject_boolean_track_id(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("track_id must be an integer, not a boolean")
        return value

    @property
    def source_id(self) -> str:
        """Return the stable source ID used by retrieval evidence."""
        return f"catalog:{self.catalog_id}:{self.track_id}"


class FieldLineage(ContractModel):
    """Typed provenance for one destination field on a catalog track."""

    field_name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    origin: FieldOrigin
    source_fields: tuple[str, ...] = Field(default=(), max_length=32)
    method_version: str | None = Field(default=None, min_length=1, max_length=160)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    interval_low: float | None = None
    interval_high: float | None = None

    @field_validator("source_fields", mode="before")
    @classmethod
    def normalize_source_fields(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)) or value is None:
            raise ValueError("source_fields must be a sequence of field names")
        try:
            fields = tuple(str(item).strip() for item in value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("source_fields must be a sequence of field names") from exc
        if any(not field for field in fields):
            raise ValueError("source_fields cannot contain empty values")
        if len(fields) != len(set(fields)):
            raise ValueError("source_fields cannot contain duplicates")
        return fields

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        if (self.interval_low is None) != (self.interval_high is None):
            raise ValueError("prediction interval requires both low and high")
        if (
            self.interval_low is not None
            and self.interval_high is not None
            and self.interval_low > self.interval_high
        ):
            raise ValueError("prediction interval low must be <= high")
        if self.origin is FieldOrigin.MODEL_ESTIMATED:
            if self.method_version is None:
                raise ValueError("model-estimated lineage requires method_version")
            if self.confidence is None:
                raise ValueError("model-estimated lineage requires confidence")
        return self

    @property
    def destination_field(self) -> str:
        """Readable alias matching the provenance concept in the data card."""
        return self.field_name


class MoodQuadrant(str, Enum):
    """Cadence's experimental valence/arousal mood vocabulary."""

    UPBEAT = "upbeat"
    CALM = "calm"
    INTENSE = "intense"
    SOMBER = "somber"


class MoodProfile(ContractModel):
    """A derived mood distribution; never an authored catalog mood."""

    upbeat: float = Field(ge=0.0, le=1.0)
    calm: float = Field(ge=0.0, le=1.0)
    intense: float = Field(ge=0.0, le=1.0)
    somber: float = Field(ge=0.0, le=1.0)
    label: MoodQuadrant | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    method_version: str = Field(
        default="cadence-va-quadrant-v1", min_length=1, max_length=160
    )
    experimental: bool = True

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        scores = {
            MoodQuadrant.UPBEAT: self.upbeat,
            MoodQuadrant.CALM: self.calm,
            MoodQuadrant.INTENSE: self.intense,
            MoodQuadrant.SOMBER: self.somber,
        }
        if abs(sum(scores.values()) - 1.0) > 1e-6:
            raise ValueError("mood profile scores must sum to 1")
        if self.label is not None and scores[self.label] != max(scores.values()):
            raise ValueError("mood profile label must name a highest-scoring quadrant")
        return self


class CatalogEdition(str, Enum):
    """Which distributable edition backs a catalog."""

    FICTIONAL = "fictional"
    FULL = "full"
    LITE = "lite"


class CatalogCapabilities(ContractModel):
    """Features a catalog can support without guessing."""

    supported_filters: tuple[str, ...] = ()
    supported_features: tuple[str, ...] = ()
    retrieval_methods: tuple[str, ...] = ()
    context_guides: bool = False
    research: bool = False

    @field_validator("supported_filters", "supported_features", "retrieval_methods", mode="before")
    @classmethod
    def normalize_capability_names(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)) or value is None:
            raise ValueError("capability names must be a sequence of strings")
        try:
            names = tuple(" ".join(str(item).split()).lower() for item in value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("capability names must be a sequence of strings") from exc
        if any(not name for name in names):
            raise ValueError("capability names cannot be empty")
        if len(names) != len(set(names)):
            raise ValueError("capability names cannot contain duplicates")
        return names

    def supports_filter(self, name: str) -> bool:
        """Return whether a hard filter is evidenced by this catalog."""
        return name.strip().lower() in self.supported_filters


class FieldCoverage(ContractModel):
    """Coverage of one field in a concrete catalog artifact."""

    field_name: str = Field(min_length=1, max_length=80)
    ratio: float = Field(ge=0.0, le=1.0)


class CatalogDescriptor(ContractModel):
    """Auditable identity, coverage, and capabilities for one artifact."""

    catalog_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    artifact_id: str = Field(min_length=1, max_length=200)
    edition: CatalogEdition
    schema_version: str = Field(min_length=1, max_length=80)
    etl_version: str = Field(min_length=1, max_length=80)
    source_checksum: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    artifact_checksum: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    accepted_count: int = Field(ge=0)
    quarantined_count: int = Field(default=0, ge=0)
    licenses: tuple[str, ...] = ()
    attribution: tuple[str, ...] = ()
    field_coverage: tuple[FieldCoverage, ...] = ()
    capabilities: CatalogCapabilities = CatalogCapabilities()
    calibration_status: str = Field(default="not_applicable", min_length=1, max_length=80)

    @field_validator("catalog_id", mode="before")
    @classmethod
    def normalize_descriptor_catalog_id(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("accepted_count", "quarantined_count", mode="before")
    @classmethod
    def reject_boolean_counts(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("catalog counts must be integers, not booleans")
        return value

    @field_validator("field_coverage", mode="before")
    @classmethod
    def accept_coverage_mapping(cls, value: object) -> object:
        if isinstance(value, dict):
            return tuple(
                {"field_name": field, "ratio": ratio}
                for field, ratio in sorted(value.items())
            )
        return value

    @model_validator(mode="after")
    def unique_coverage_fields(self) -> Self:
        names = [item.field_name for item in self.field_coverage]
        if len(names) != len(set(names)):
            raise ValueError("field_coverage cannot contain duplicate fields")
        return self


class ResearchStatus(str, Enum):
    """Bounded outcomes of optional post-ranking research."""

    NOT_REQUESTED = "not_requested"
    PUBLISHED = "published"  # grounded web research with validated citations
    CATALOG_NOTE = "catalog_note"  # non-grounded note from the track's own catalog facts
    NO_MATCH = "no_match"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    LOCAL_FALLBACK = "local_fallback"


def _validate_http_url(value: str) -> str:
    """Accept only absolute HTTP(S) URLs with a real host and no credentials."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL cannot contain credentials")
    return value


class ResearchCitation(ContractModel):
    """One allowlisted source cited by a research claim."""

    citation_id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=300)
    url: str = Field(min_length=1, max_length=2000)
    source_domain: str = Field(min_length=1, max_length=253)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _validate_http_url(value)

    @field_validator("source_domain", mode="before")
    @classmethod
    def normalize_source_domain(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower().rstrip(".")
        return value

    @model_validator(mode="after")
    def domain_matches_url(self) -> Self:
        hostname = (urlsplit(self.url).hostname or "").lower().rstrip(".")
        if hostname != self.source_domain and not hostname.endswith(f".{self.source_domain}"):
            raise ValueError("citation source_domain does not match URL host")
        return self


class ResearchClaim(ContractModel):
    """One short research statement with explicit citation references."""

    text: str = Field(min_length=1, max_length=500)
    citation_ids: tuple[str, ...] = Field(min_length=1, max_length=3)


class ResearchBrief(ContractModel):
    """Session-only, post-ranking research that can never become catalog truth."""

    track_ref: TrackRef
    status: ResearchStatus
    identity_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    narrative: str | None = Field(default=None, min_length=1, max_length=1200)
    claims: tuple[ResearchClaim, ...] = Field(default=(), max_length=3)
    citations: tuple[ResearchCitation, ...] = Field(default=(), max_length=9)
    source_domains: tuple[str, ...] = ()
    provider: str | None = Field(default=None, min_length=1, max_length=120)
    model_id: str | None = Field(default=None, min_length=1, max_length=160)
    timestamp: str | None = Field(default=None, min_length=1, max_length=80)
    warnings: tuple[str, ...] = ()

    @field_validator("source_domains", mode="before")
    @classmethod
    def normalize_source_domains(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)) or value is None:
            raise ValueError("source_domains must be a sequence")
        domains = tuple(str(item).strip().lower().rstrip(".") for item in value)  # type: ignore[arg-type]
        if any(not domain for domain in domains):
            raise ValueError("source_domains cannot contain empty values")
        if len(domains) != len(set(domains)):
            raise ValueError("source_domains cannot contain duplicates")
        return domains

    @model_validator(mode="after")
    def validate_citation_coverage(self) -> Self:
        citation_ids = [citation.citation_id for citation in self.citations]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("research citations cannot reuse citation_id")
        known_ids = set(citation_ids)
        for claim in self.claims:
            if not set(claim.citation_ids) <= known_ids:
                raise ValueError("research claim references an unknown citation")
        actual_domains = tuple(dict.fromkeys(c.source_domain for c in self.citations))
        if self.source_domains and set(self.source_domains) != set(actual_domains):
            raise ValueError("source_domains must match citation source domains")
        if self.status is ResearchStatus.PUBLISHED and (not self.claims or not self.citations):
            raise ValueError("published research requires claims and citations")
        if self.status is ResearchStatus.CATALOG_NOTE and not self.narrative:
            raise ValueError("a catalog note requires a narrative")
        return self


class RecommendationRequest(ContractModel):
    """Structured listener preferences accepted by the deterministic scorer.

    This request stays structured-only by design. Natural-language input is
    handled separately by ``MusicCompanion`` (guard → intent parser → retrieval),
    so the trusted scorer path never has to interpret free text.
    """

    PREFERENCE_FIELDS: ClassVar[tuple[str, ...]] = (
        "genre",
        "mood",
        "energy",
        "acousticness",
        "valence",
        "danceability",
        "tempo_bpm",
    )
    NUMERIC_FIELDS: ClassVar[tuple[str, ...]] = (
        "energy",
        "acousticness",
        "valence",
        "danceability",
        "tempo_bpm",
    )

    genre: str | None = Field(default=None, min_length=1, max_length=80)
    mood: str | None = Field(default=None, min_length=1, max_length=80)
    energy: float | None = Field(default=None, ge=0.0, le=1.0)
    acousticness: float | None = Field(default=None, ge=0.0, le=1.0)
    valence: float | None = Field(default=None, ge=0.0, le=1.0)
    danceability: float | None = Field(default=None, ge=0.0, le=1.0)
    tempo_bpm: float | None = Field(default=None, ge=50.0, le=200.0)
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("genre", "mood", mode="after")
    @classmethod
    def normalize_category(cls, value: str | None) -> str | None:
        """Normalize categories so matching is stable across interfaces."""
        return value.lower() if value is not None else None

    @field_validator(*NUMERIC_FIELDS, mode="before")
    @classmethod
    def reject_boolean_numbers(cls, value: object) -> object:
        """Reject booleans instead of silently treating True/False as 1/0."""
        if isinstance(value, bool):
            raise ValueError("boolean values are not valid numeric preferences")
        return value

    @field_validator("limit", mode="before")
    @classmethod
    def reject_boolean_limit(cls, value: object) -> object:
        """Reject True as a request for one recommendation."""
        if isinstance(value, bool):
            raise ValueError("limit must be an integer, not a boolean")
        return value

    @model_validator(mode="after")
    def require_preference(self) -> Self:
        """An all-empty request cannot produce a meaningful recommendation."""
        if not any(getattr(self, field) is not None for field in self.PREFERENCE_FIELDS):
            raise ValueError("provide at least one music preference")
        return self


class CatalogTrack(ContractModel):
    """Validated form of one authoritative or evidence-enriched catalog record.

    Only identity is universally required. Optional values stay ``None`` when a
    source cannot support them; ``False`` and ``0.0`` remain real, known values.
    The fictional catalog still supplies every legacy field, so its documents
    and scores remain byte-for-byte unchanged.
    """

    NUMERIC_FIELDS: ClassVar[tuple[str, ...]] = (
        "energy",
        "tempo_bpm",
        "valence",
        "danceability",
        "acousticness",
        "instrumentalness",
    )

    id: int = Field(gt=0)
    catalog_id: str = Field(
        default="fictional",
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    external_id: str | None = Field(default=None, min_length=1, max_length=200)
    # Real FMA identity/text fields have no published source length ceiling.
    # They are normalized during ETL; imposing the fictional catalog's legacy
    # caps here would turn otherwise valid real tracks into runtime failures.
    title: str = Field(min_length=1)
    artist: str = Field(min_length=1)
    genre: str | None = Field(default=None, min_length=1, max_length=80)
    genres: tuple[str, ...] = Field(default=(), max_length=64)
    mood: str | None = Field(default=None, min_length=1, max_length=80)
    mood_profile: MoodProfile | None = None
    energy: float | None = Field(default=None, ge=0.0, le=1.0)
    tempo_bpm: float | None = Field(default=None, ge=50.0, le=200.0)
    valence: float | None = Field(default=None, ge=0.0, le=1.0)
    danceability: float | None = Field(default=None, ge=0.0, le=1.0)
    acousticness: float | None = Field(default=None, ge=0.0, le=1.0)
    instrumentalness: float | None = Field(default=None, ge=0.0, le=1.0)
    description: str | None = Field(default=None, min_length=20, max_length=500)
    tags: tuple[str, ...] = Field(default=(), max_length=64)
    album_tags: tuple[str, ...] = Field(default=(), max_length=64)
    artist_tags: tuple[str, ...] = Field(default=(), max_length=64)
    contexts: tuple[str, ...] = Field(default=(), max_length=64)
    instruments: tuple[str, ...] = Field(default=(), max_length=64)
    instrumental: bool | None = None
    explicit: bool | None = None
    era: str | None = Field(default=None, pattern=r"^(?:19|20)\d0s$")
    track_information: str | None = Field(default=None, min_length=1)
    album_information: str | None = Field(default=None, min_length=1)
    artist_biography: str | None = Field(default=None, min_length=1)
    license: str | None = Field(default=None, min_length=1, max_length=500)
    source_url: str | None = Field(default=None, min_length=1, max_length=2048)
    track_url: str | None = Field(default=None, min_length=1, max_length=2048)
    artist_url: str | None = Field(default=None, min_length=1, max_length=2048)
    album_url: str | None = Field(default=None, min_length=1, max_length=2048)
    lineage: tuple[FieldLineage, ...] = ()

    @field_validator("catalog_id", mode="before")
    @classmethod
    def normalize_catalog_id(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("genre", "mood", mode="after")
    @classmethod
    def normalize_category(cls, value: str | None) -> str | None:
        """Store matching categories in one canonical form."""
        return value.lower() if value is not None else None

    @field_validator(
        "genres",
        "tags",
        "album_tags",
        "artist_tags",
        "contexts",
        "instruments",
        mode="before",
    )
    @classmethod
    def normalize_metadata_values(cls, value: object) -> tuple[str, ...]:
        """Require nonempty, unique metadata terms in a canonical form."""
        if isinstance(value, (str, bytes)) or value is None:
            raise ValueError("metadata collections must be a sequence of strings")

        try:
            raw_values = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(
                "metadata collections must be a sequence of strings"
            ) from exc

        normalized: list[str] = []
        for item in raw_values:
            if not isinstance(item, str):
                raise ValueError("metadata collection items must be strings")
            term = " ".join(item.split()).lower()
            if not term:
                raise ValueError("metadata collection items cannot be empty")
            if len(term) > 120:
                raise ValueError("metadata collection items cannot exceed 120 characters")
            normalized.append(term)

        if len(normalized) != len(set(normalized)):
            raise ValueError("metadata collections cannot contain duplicate values")
        return tuple(normalized)

    @field_validator("instrumental", "explicit", mode="before")
    @classmethod
    def require_real_booleans(cls, value: object) -> object:
        """Reject truthy strings and integers at the validated service boundary."""
        if value is not None and not isinstance(value, bool):
            raise ValueError("catalog boolean fields must be true booleans")
        return value

    @field_validator("id", *NUMERIC_FIELDS, mode="before")
    @classmethod
    def reject_boolean_numbers(cls, value: object) -> object:
        """Catalog data must not coerce booleans into numeric attributes."""
        if isinstance(value, bool):
            raise ValueError("boolean values are not valid catalog numbers")
        return value

    @field_validator("source_url", "track_url", "artist_url", "album_url")
    @classmethod
    def validate_urls(cls, value: str | None) -> str | None:
        return _validate_http_url(value) if value is not None else None

    @model_validator(mode="after")
    def unique_lineage_fields(self) -> Self:
        fields = [item.field_name for item in self.lineage]
        if len(fields) != len(set(fields)):
            raise ValueError("track lineage cannot contain duplicate destination fields")
        return self

    @property
    def ref(self) -> TrackRef:
        """Return this track's immutable, catalog-qualified identity."""
        return TrackRef(
            catalog_id=self.catalog_id,
            track_id=self.id,
            external_id=self.external_id,
        )


class RecommendationItem(ContractModel):
    """One ranked track and the evidence produced by the local scorer."""

    track: CatalogTrack
    raw_score: float = Field(ge=0.0)
    match_strength: float = Field(ge=0.0, le=1.0)
    reasons: tuple[str, ...] = ()


class RecommendationResult(ContractModel):
    """Validated response returned by ``RecommendationService``."""

    request: RecommendationRequest
    recommendations: tuple[RecommendationItem, ...]
    max_possible_score: float = Field(gt=0.0)
    operating_mode: OperatingMode
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def enforce_result_invariants(self) -> Self:
        """Prevent oversized results and duplicate catalog IDs."""
        if len(self.recommendations) > self.request.limit:
            raise ValueError("recommendation count exceeds the requested limit")

        track_ids = [item.track.id for item in self.recommendations]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("recommendations contain duplicate track IDs")
        return self


class SourceType(str, Enum):
    """Where a retrieved piece of evidence came from.

    ``CONTEXT_GUIDE`` identifies the versioned Markdown guides used as the
    implemented second retrieval source. Guides remain evidence, never tracks.
    """

    CATALOG = "catalog"
    CONTEXT_GUIDE = "context_guide"


class ContextGuide(ContractModel):
    """One curated, human-written guide about a listening situation.

    Guides are a second retrieval source. They are not recommendable tracks;
    they connect a listener's words to catalog vocabulary and provide grounded
    context for an explanation.
    """

    guide_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=20, max_length=2000)

    def index_text(self) -> str:
        """Return the text indexed for retrieval (title plus body)."""
        return f"{self.title} {self.body}"


class RetrievalHit(ContractModel):
    """One track surfaced by the retriever, with the provenance that justifies it.

    ``score`` is a cosine similarity in ``[0, 1]``, not a probability or a
    calibrated confidence. Every hit records where it came from so a later
    evaluator (and a human) can trace why it was retrieved.
    """

    source_type: SourceType
    source_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    fields_used: tuple[str, ...] = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    matched_terms: tuple[str, ...] = ()
    semantic_score: float | None = Field(default=None, ge=0.0, le=1.0)
    lexical_score: float | None = Field(default=None, ge=0.0, le=1.0)
    structured_score: float | None = Field(default=None, ge=0.0, le=1.0)
    structured_reasons: tuple[str, ...] = ()
    fusion_version: str | None = Field(default=None, min_length=1)
    track: CatalogTrack


class GuideEvidence(ContractModel):
    """A context guide that informed a retrieval, kept as cited evidence.

    ``expansion_terms`` are the catalog-vocabulary terms this guide contributed
    to the track query, which is how a guide improves retrieval without ever
    being recommended itself.
    """

    source_type: SourceType
    source_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    title: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    matched_terms: tuple[str, ...] = ()
    expansion_terms: tuple[str, ...] = ()


class RetrievalResult(ContractModel):
    """Validated response returned by a ``Retriever``.

    ``index_fingerprint`` identifies the exact index the hits came from, so a
    result can be tied back to a specific catalog content hash and retrieval
    method version. ``guides_used`` and ``expanded_query_terms`` record how a
    second source (context guides) shaped this result; both are empty when no
    guide fired.
    """

    query: str
    hits: tuple[RetrievalHit, ...]
    index_fingerprint: str = Field(min_length=1)
    filters_applied: tuple[str, ...] = ()
    guides_used: tuple[GuideEvidence, ...] = ()
    expanded_query_terms: tuple[str, ...] = ()
    operating_mode: OperatingMode = OperatingMode.LOCAL
    embedding_source: EmbeddingSource | None = None


class ScoreComponents(ContractModel):
    """The per-signal scores behind one ranked candidate.

    Each signal is ``float | None`` with a deliberate distinction the coming
    structured-preference leg must inherit intact:

    * ``None`` = **not evaluated** — this signal did not run for this query;
    * ``0.0``  = **evaluated, no match** — it ran and found nothing.

    Collapsing the two would make "we didn't look" indistinguishable from "we
    looked and it's a miss", silently biasing any fusion that averages them.
    ``fused`` is relevance after signal fusion but **before** MMR diversity; MMR
    may deliberately change the visible order while keeping a relevance floor.
    ``fusion_version`` names exactly how the relevance was produced, so the
    signal stays reproducible and comparable across runs.
    """

    semantic: float | None = Field(default=None, ge=0.0, le=1.0)
    lexical: float | None = Field(default=None, ge=0.0, le=1.0)
    categorical: float | None = Field(default=None, ge=0.0, le=1.0)
    numeric: float | None = Field(default=None, ge=0.0, le=1.0)
    structured: float | None = Field(default=None, ge=0.0, le=1.0)
    personalization: float | None = Field(default=None, ge=0.0, le=1.0)
    fused: float = Field(ge=0.0, le=1.0)
    available_signals: tuple[str, ...] = ()
    fusion_version: str = Field(min_length=1)
    reasons: tuple[str, ...] = ()


class RankedCandidate(ContractModel):
    """One track as ranked, with its full score breakdown and provenance.

    A single, retriever-agnostic view: the local scorer, the hybrid retriever,
    and the coming structured leg all describe a result the same way, so the
    evaluator, the UI, and the event log never have to special-case a source.
    """

    track: CatalogTrack
    components: ScoreComponents
    source_type: SourceType = SourceType.CATALOG
    content_hash: str = Field(min_length=1)


class GuardCategory(str, Enum):
    """How the input/privacy guard classified a raw natural-language query."""

    OK = "ok"
    EMPTY = "empty"
    TOO_LONG = "too_long"
    SENSITIVE = "sensitive"  # contained PII or a secret (now redacted)
    INJECTION = "injection"  # contained a prompt-injection directive (now stripped)
    HIGH_RISK = "high_risk"  # crisis/self-harm; routed to a safe response


class GuardVerdict(ContractModel):
    """The guard's decision about one raw query.

    ``sanitized_query`` is safe to continue processing under the category's
    routing policy: PII/secret spans are replaced with ``[redacted]`` and
    injection directives are stripped.  It is still user-authored text and must
    never be persisted in logs or encoded into a shareable URL.
    """

    category: GuardCategory
    sanitized_query: str = ""
    reason: str = ""


class FeatureRelation(str, Enum):
    """The *direction* a preference points a numeric feature."""

    PREFER_HIGH = "prefer_high"
    PREFER_LOW = "prefer_low"
    NEAR = "near"
    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    RANGE = "range"


class FeatureGoal(ContractModel):
    """A directional preference over one numeric feature — a soft signal, not a filter.

    Relations express *direction*, never a fabricated target: "high energy" is
    ``prefer_high``, not ``energy == 0.85``. ``NEAR``/``AT_LEAST``/``AT_MOST``
    carry a ``target`` in the feature's own units (0-1, or BPM for ``tempo_bpm``);
    ``RANGE`` carries ``low``/``high``. ``cue_id`` is a controlled identifier
    (e.g. ``energy_low_v1``) so a preference is reproducible and auditable — never
    free text. A goal only ever *reorders* candidates; hard constraints
    (``instrumental_only``, ``exclude_explicit``) remove them.
    """

    NUMERIC_FEATURES: ClassVar[tuple[str, ...]] = (
        "energy",
        "valence",
        "danceability",
        "acousticness",
        "tempo_bpm",
        "instrumentalness",
    )

    feature: str
    relation: FeatureRelation
    target: float | None = None
    low: float | None = None
    high: float | None = None
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    cue_id: str = Field(min_length=1, max_length=60)

    @field_validator("feature")
    @classmethod
    def known_feature(cls, value: str) -> str:
        if value not in cls.NUMERIC_FEATURES:
            raise ValueError(f"unknown numeric feature: {value}")
        return value

    @model_validator(mode="after")
    def require_relation_params(self) -> Self:
        minimum, maximum = (
            (50.0, 200.0) if self.feature == "tempo_bpm" else (0.0, 1.0)
        )
        needs_target = {
            FeatureRelation.NEAR, FeatureRelation.AT_LEAST, FeatureRelation.AT_MOST
        }
        if self.relation in needs_target and self.target is None:
            raise ValueError(f"{self.relation.value} requires a target")
        if self.relation in needs_target and (self.low is not None or self.high is not None):
            raise ValueError(f"{self.relation.value} accepts target only")
        if self.relation is FeatureRelation.RANGE:
            if self.low is None or self.high is None:
                raise ValueError("range requires low and high")
            if self.target is not None:
                raise ValueError("range accepts low and high only")
            if self.low > self.high:
                raise ValueError("range low must be <= high")
        if self.relation in (FeatureRelation.PREFER_HIGH, FeatureRelation.PREFER_LOW):
            if any(value is not None for value in (self.target, self.low, self.high)):
                raise ValueError(f"{self.relation.value} accepts no numeric parameters")
        for label, value in (("target", self.target), ("low", self.low), ("high", self.high)):
            if value is not None and not minimum <= value <= maximum:
                raise ValueError(
                    f"{self.feature} {label} must be between {minimum:g} and {maximum:g}"
                )
        return self


class MusicIntent(ContractModel):
    """Structured intent parsed from a guarded query.

    ``query`` is the sanitized text handed to the retriever; the categorical and
    filter fields are extracted deterministically from recognizable music words.
    ``feature_goals`` are the directional numeric preferences (soft signals) that
    the structured leg scores; they never filter.
    """

    query: str = ""
    genre: str | None = Field(default=None, max_length=80)
    mood: str | None = Field(default=None, max_length=80)
    instrumental_only: bool = False
    exclude_explicit: bool = False
    feature_goals: tuple[FeatureGoal, ...] = ()
    open_request: bool = False  # "surprise me" / "anything" — invite a varied set
    limit: int = Field(default=5, ge=1, le=20)
    needs_clarification: bool = False
    clarification: str | None = None
    source: str = "rules"

    @model_validator(mode="after")
    def require_one_goal_per_feature(self) -> Self:
        """Keep structured scoring from counting one feature more than once.

        A listener may replace an energy preference, but a completed intent may
        not contain both high- and low-energy goals. Refinement code performs the
        replacement before constructing this contract.
        """
        features = [goal.feature for goal in self.feature_goals]
        if len(features) != len(set(features)):
            raise ValueError("feature_goals must contain at most one goal per feature")
        return self


class FeatureVector(ContractModel):
    """A liked track's real audio-feature fingerprint, for session exemplar pull."""

    energy: float | None = Field(default=None, ge=0.0, le=1.0)
    valence: float | None = Field(default=None, ge=0.0, le=1.0)
    danceability: float | None = Field(default=None, ge=0.0, le=1.0)
    acousticness: float | None = Field(default=None, ge=0.0, le=1.0)
    instrumentalness: float | None = Field(default=None, ge=0.0, le=1.0)
    genre: str | None = Field(default=None, max_length=80)


class SessionPreference(ContractModel):
    """Session-only, reversible taste signal accumulated from listener feedback.

    Bounded so it only *nudges* ranking, never overriding a named intent. It is
    never stored on the shared companion — it rides on ``ExecutionPolicy`` per
    call and lives only in one browser session's state, so two sessions cannot
    influence each other. ``suppressed_ids`` is a soft session demotion (a
    reversible "not right now"), never a permanent ban. ``enabled=False`` is the
    listener's "don't learn" switch.
    """

    feature_bias: dict[str, float] = Field(default_factory=dict)
    genre_bias: dict[str, float] = Field(default_factory=dict)
    exemplars: tuple[FeatureVector, ...] = ()
    suppressed_ids: tuple[int, ...] = ()
    enabled: bool = True

    @field_validator("feature_bias", "genre_bias")
    @classmethod
    def bounded_bias(cls, value: dict[str, float]) -> dict[str, float]:
        """Bias magnitudes stay in [-1, 1] so learning can only ever nudge."""
        if any(not (-1.0 <= weight <= 1.0) for weight in value.values()):
            raise ValueError("session bias weights must be within [-1, 1]")
        return value

    @property
    def is_active(self) -> bool:
        """True when learning is on and some signal has actually accumulated."""
        return self.enabled and bool(
            self.feature_bias or self.genre_bias or self.exemplars or self.suppressed_ids
        )


class DiversityLevel(str, Enum):
    """How tightly MMR should stay near the top relevance ranking.

    This deliberately says *focused/exploratory*, not familiar/adventurous:
    Cadence has no popularity or listener-history signal yet, but it can honestly
    control the relevance-versus-variety trade-off.
    """

    FOCUSED = "focused"
    BALANCED = "balanced"
    EXPLORATORY = "exploratory"


class ExecutionPolicy(ContractModel):
    """Per-turn controls that change *how* a request may execute.

    The policy is deliberately separate from musical intent.  ``force_local`` is
    a privacy rule, while ``diversity`` controls the bounded MMR preset; neither
    should masquerade as something the listener asked to hear.
    """

    force_local: bool = False
    diversity: DiversityLevel = DiversityLevel.BALANCED
    preference: SessionPreference | None = None


class CompanionAction(str, Enum):
    """The bounded set of actions the companion may take for one query."""

    RECOMMEND = "recommend"
    CLARIFY = "clarify"
    NO_MATCH = "no_match"
    SAFE_RESPONSE = "safe_response"
    DEGRADED = "degraded"


class VoiceSource(str, Enum):
    """Which renderer produced the companion's message."""

    TEMPLATE = "template"  # deterministic, reproducible, always available
    GENERATED = "generated"  # provider selected approved microcopy; legacy event value


class EvaluationReport(ContractModel):
    """The grounding evaluator's verdict on a result or a rendered message."""

    ok: bool
    failures: tuple[str, ...] = ()


class AgentTrace(ContractModel):
    """A structured, privacy-safe record of one bounded-agent turn.

    It captures categories, ids, and decisions — never raw sensitive text — so a
    reviewer can see how the companion reached its answer. ``evaluation`` is the
    result check; ``text_evaluation`` is the grounding check on generated text (if
    a generator ran); ``voice_model`` names the generator that produced the voice.
    """

    guard_category: GuardCategory
    intent_summary: str = ""
    retrieved_ids: tuple[int, ...] = ()
    retrieved_refs: tuple[TrackRef, ...] = ()
    diversity_applied: bool = False
    evaluation: EvaluationReport = EvaluationReport(ok=True)
    text_evaluation: EvaluationReport | None = None
    action: CompanionAction
    voice_source: VoiceSource = VoiceSource.TEMPLATE
    voice_model: str | None = None
    network_used: bool = False
    fallback_reason: str | None = None


class CompanionResponse(ContractModel):
    """Validated response from the natural-language companion.

    ``retrieval`` reuses the retriever's own result (hits, provenance, operating
    mode, guide evidence); it is ``None`` for clarify/safe/empty outcomes.
    ``trace`` is the bounded-agent record of how the answer was produced.
    """

    action: CompanionAction
    message: str
    intro_message: str | None = None
    retrieval: RetrievalResult | None = None
    intent: MusicIntent | None = None
    trace: AgentTrace | None = None

    @model_validator(mode="after")
    def enforce_action_invariants(self) -> Self:
        """Keep action and payload consistent (e.g. recommend must have hits)."""
        if self.action in (CompanionAction.RECOMMEND, CompanionAction.DEGRADED):
            if self.retrieval is None or not self.retrieval.hits:
                raise ValueError(f"{self.action.value} response must include hits")
        if self.action is CompanionAction.NO_MATCH:
            if self.retrieval is None or self.retrieval.hits:
                raise ValueError("no_match response must include an empty retrieval result")
        if self.action in (CompanionAction.CLARIFY, CompanionAction.SAFE_RESPONSE):
            if self.retrieval is not None:
                raise ValueError(f"{self.action.value} response must not include retrieval")
        return self


class PipelineReceipt(ContractModel):
    """Request-local diagnostics returned to a UI without reading shared logs.

    It contains only allowlisted decisions, identifiers, provenance, timings, and
    fingerprints.  No query or prompt text is permitted in this contract.
    """

    request_id: str = Field(min_length=1)
    latency_ms: float = Field(ge=0.0)
    candidate_ids: tuple[int, ...] = ()
    final_ids: tuple[int, ...] = ()
    candidate_refs: tuple[TrackRef, ...] = ()
    final_refs: tuple[TrackRef, ...] = ()
    guard_category: GuardCategory
    action: CompanionAction
    force_local: bool
    diversity: DiversityLevel
    embedding_source: EmbeddingSource | None = None
    network_used: bool = False
    operating_mode: OperatingMode | None = None
    voice_source: VoiceSource | None = None
    index_fingerprint: str | None = None
    config_fingerprint: str | None = None


class SignalRow(ContractModel):
    """One pooled candidate's per-leg ranking signals (catalog data only)."""

    track_id: int = Field(gt=0)
    track_ref: TrackRef | None = None
    title: str = Field(min_length=1, max_length=200)
    text: float = Field(ge=0.0, le=1.0)
    structured: float | None = Field(default=None, ge=0.0, le=1.0)
    fused: float = Field(ge=0.0, le=1.0)


class SignalComparison(ContractModel):
    """Developer-only view of how the candidate pool ranked under each leg —
    text-only vs structured vs fused — *before* diversity re-ranking.

    It carries catalog titles for display, so it is deliberately kept out of the
    ``PipelineReceipt`` and the JSONL event log. ``structured_active`` is False
    when no structured signal ran, in which case ``text`` and ``fused`` agree.
    """

    structured_active: bool
    rows: tuple[SignalRow, ...] = ()


class CompanionTurn(ContractModel):
    """A response plus its request-local, privacy-safe pipeline receipt.

    ``comparison`` is an optional developer-only signal breakdown for the
    candidate pool; it is never logged (it carries catalog titles) and is present
    only for turns that actually retrieved a pool.
    """

    response: CompanionResponse
    receipt: PipelineReceipt
    comparison: SignalComparison | None = None


class CompanionEvent(ContractModel):
    """A privacy-safe receipt for one companion turn — decisions and ids, never words.

    It records *what the system did*: the guard category, allowlisted intent
    facets, which track ids were considered and returned, their scores, the
    operating mode, timings, and fingerprints — so a run can be audited and
    reproduced.

    It deliberately omits *what the person said*: no raw or sanitized query, no
    prompt, no persistent identifier. ``request_id`` is an ephemeral per-turn
    correlation id, not an identity. A receipt, not a diary.
    """

    schema_version: str = "2"
    request_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    guard_category: GuardCategory
    action: CompanionAction
    intent_summary: str = ""  # allowlisted facets only (genre/mood/flags), never free text
    operating_mode: OperatingMode | None = None
    voice_source: VoiceSource | None = None
    embedding_source: EmbeddingSource | None = None
    network_used: bool = False
    force_local: bool = False
    diversity: DiversityLevel = DiversityLevel.BALANCED
    candidate_ids: tuple[int, ...] = ()
    final_ids: tuple[int, ...] = ()
    candidate_refs: tuple[TrackRef, ...] = ()
    final_refs: tuple[TrackRef, ...] = ()
    components: tuple[ScoreComponents, ...] = ()  # parallel to final_ids; reasons stripped
    fallback_reason: str | None = None
    latency_ms: float = Field(ge=0.0)
    index_fingerprint: str | None = None
    config_fingerprint: str | None = None
