"""End-to-end checks for catalog capability validation and namespaced receipts."""

from __future__ import annotations

from src.companion import MusicCompanion
from src.contracts import (
    CatalogCapabilities,
    CatalogDescriptor,
    CatalogEdition,
    CatalogTrack,
    CompanionAction,
)


def _descriptor(*, filters=(), features=("instrumentalness",)):
    return CatalogDescriptor(
        catalog_id="fma",
        artifact_id="fma-lite-test",
        edition=CatalogEdition.LITE,
        schema_version="2",
        etl_version="test",
        accepted_count=1,
        capabilities=CatalogCapabilities(
            supported_filters=filters,
            supported_features=features,
            retrieval_methods=("local_tfidf",),
            research=True,
        ),
        calibration_status="experimental",
    )


def _track():
    return CatalogTrack(
        id=7,
        catalog_id="fma",
        title="Known Identity",
        artist="Known Artist",
        genre="electronic",
        genres=("electronic",),
        energy=0.8,
        valence=0.7,
        instrumentalness=0.9,
        tags=("independent", "electronic"),
    )


def test_fma_clarifies_unknown_clean_and_instrumental_booleans_before_retrieval():
    companion = MusicCompanion([_track()], catalog_descriptor=_descriptor())
    response = companion.respond("clean instrumental electronic music")

    assert response.action is CompanionAction.CLARIFY
    assert response.retrieval is None
    assert "can’t verify clean lyrics or instrumental-only status" in response.message
    assert "rather not guess" in response.message


def test_supported_filter_declaration_allows_the_legacy_boolean_path():
    companion = MusicCompanion(
        [_track().model_copy(update={"explicit": False, "instrumental": True})],
        catalog_descriptor=_descriptor(
            filters=("exclude_explicit", "instrumental_only")
        ),
    )
    response = companion.respond("clean instrumental electronic music")
    assert response.action in {CompanionAction.RECOMMEND, CompanionAction.DEGRADED}


def test_receipt_and_trace_use_catalog_qualified_references():
    companion = MusicCompanion([_track()], catalog_descriptor=_descriptor())
    turn = companion.respond_detailed("more instrumental electronic music")

    assert turn.response.action in {CompanionAction.RECOMMEND, CompanionAction.DEGRADED}
    assert all(ref.catalog_id == "fma" for ref in turn.receipt.candidate_refs)
    assert all(ref.catalog_id == "fma" for ref in turn.receipt.final_refs)
    assert turn.response.trace is not None
    assert turn.response.trace.retrieved_refs == turn.receipt.final_refs
    assert "instrumental_only=False" in turn.response.trace.intent_summary
    assert "instrumentalness_prefer_high_v1" in turn.response.trace.intent_summary
