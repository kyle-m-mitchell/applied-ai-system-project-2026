"""Phase 5 evidence-first catalog contracts and unknown-safe retrieval tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.contracts import (
    CatalogCapabilities,
    CatalogDescriptor,
    CatalogEdition,
    CatalogTrack,
    FieldLineage,
    FieldOrigin,
    MoodProfile,
    ResearchBrief,
    ResearchCitation,
    ResearchClaim,
    ResearchStatus,
    TrackRef,
)
from src.retrieval import TfidfRetriever, build_document_text, document_fields_used


def test_minimal_track_keeps_unknown_distinct_from_false_and_zero():
    unknown = CatalogTrack(id=1, catalog_id="FMA", title="Unknown", artist="Artist")
    known = CatalogTrack(
        id=2,
        catalog_id="fma",
        title="Known",
        artist="Artist",
        energy=0.0,
        instrumental=False,
        explicit=False,
    )

    assert unknown.catalog_id == "fma"
    assert unknown.energy is None and unknown.instrumental is None and unknown.explicit is None
    assert known.energy == 0.0 and known.instrumental is False and known.explicit is False
    assert unknown.ref == TrackRef(catalog_id="fma", track_id=1)
    assert unknown.ref.source_id == "catalog:fma:1"


def test_retrieval_skips_unknown_fields_and_reports_only_real_evidence():
    track = CatalogTrack(
        id=1,
        catalog_id="fma",
        title="A Track",
        artist="An Artist",
        genre="ambient",
        tags=("drone",),
    )

    document = build_document_text(track)
    assert document == "ambient drone"
    assert "none" not in document.lower()
    assert document_fields_used(track) == ("genre", "tags")

    result = TfidfRetriever((track,)).search("ambient", k=1)
    assert result.hits[0].fields_used == ("genre", "tags")
    assert result.hits[0].source_id == "catalog:fma:1"


def test_unknown_booleans_are_excluded_from_verified_filters():
    common = {"catalog_id": "fma", "artist": "Artist", "genre": "ambient"}
    tracks = (
        CatalogTrack(id=1, title="Unknown", instrumental=None, explicit=None, **common),
        CatalogTrack(id=2, title="Verified", instrumental=True, explicit=False, **common),
        CatalogTrack(id=3, title="Vocal", instrumental=False, explicit=True, **common),
    )
    retriever = TfidfRetriever(tracks)

    instrumental = retriever.search("ambient", k=10, instrumental_only=True)
    clean = retriever.search("ambient", k=10, exclude_explicit=True)
    assert [hit.track.id for hit in instrumental.hits] == [2]
    assert [hit.track.id for hit in clean.hits] == [2]


def test_lineage_requires_honest_model_metadata_and_paired_interval():
    with pytest.raises(ValidationError, match="method_version"):
        FieldLineage(field_name="energy", origin=FieldOrigin.MODEL_ESTIMATED)
    with pytest.raises(ValidationError, match="both low and high"):
        FieldLineage(
            field_name="energy",
            origin=FieldOrigin.MODEL_ESTIMATED,
            method_version="energy-v1",
            confidence=0.8,
            interval_low=0.2,
        )

    lineage = FieldLineage(
        field_name="energy",
        origin=FieldOrigin.MODEL_ESTIMATED,
        source_fields=("librosa",),
        method_version="energy-v1",
        confidence=0.8,
        interval_low=0.2,
        interval_high=0.6,
    )
    assert lineage.destination_field == "energy"


def test_mood_profile_is_a_normalized_experimental_distribution():
    profile = MoodProfile(
        upbeat=0.6,
        calm=0.2,
        intense=0.1,
        somber=0.1,
        label="upbeat",
        confidence=0.75,
    )
    assert profile.experimental is True
    assert profile.label.value == "upbeat"

    with pytest.raises(ValidationError, match="sum to 1"):
        MoodProfile(upbeat=0.9, calm=0.2, intense=0.1, somber=0.1)


def test_descriptor_accepts_immutable_coverage_and_declares_capabilities():
    descriptor = CatalogDescriptor(
        catalog_id="fma",
        artifact_id="fma-lite-v1",
        edition=CatalogEdition.LITE,
        schema_version="2",
        etl_version="1",
        accepted_count=300,
        field_coverage={"energy": 1.0, "mood": 0.0},
        capabilities=CatalogCapabilities(
            supported_filters=("genre",),
            supported_features=("energy",),
            retrieval_methods=("fts5", "structured"),
            research=True,
        ),
        calibration_status="experimental",
    )
    assert descriptor.capabilities.supports_filter("GENRE")
    assert tuple(item.field_name for item in descriptor.field_coverage) == ("energy", "mood")


def test_research_fallback_needs_no_citations_but_published_claims_do():
    ref = TrackRef(catalog_id="fma", track_id=7, external_id="fma:7")
    fallback = ResearchBrief(track_ref=ref, status=ResearchStatus.LOCAL_FALLBACK)
    assert fallback.claims == () and fallback.citations == ()

    with pytest.raises(ValidationError, match="requires claims and citations"):
        ResearchBrief(track_ref=ref, status=ResearchStatus.PUBLISHED)

    citation = ResearchCitation(
        citation_id="mb-1",
        title="Recording",
        url="https://musicbrainz.org/recording/example",
        source_domain="musicbrainz.org",
    )
    brief = ResearchBrief(
        track_ref=ref,
        status=ResearchStatus.PUBLISHED,
        identity_confidence=0.99,
        claims=(ResearchClaim(text="A grounded claim.", citation_ids=("mb-1",)),),
        citations=(citation,),
        source_domains=("musicbrainz.org",),
        provider="musicbrainz",
        timestamp="2026-08-02T12:00:00Z",
    )
    assert brief.claims[0].citation_ids == ("mb-1",)

    with pytest.raises(ValidationError, match="unknown citation"):
        ResearchBrief(
            track_ref=ref,
            status=ResearchStatus.PUBLISHED,
            claims=(ResearchClaim(text="Ungrounded.", citation_ids=("missing",)),),
            citations=(citation,),
        )
