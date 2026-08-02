"""Tests for the grounding evaluator."""

from __future__ import annotations

from src.contracts import CatalogTrack, MusicIntent, RetrievalHit, SourceType
from src.evaluator import GroundingEvaluator


def _hit(track_id, *, title="Track", instrumental=True, explicit=False, evidence=True):
    track = CatalogTrack.model_validate(
        {
            "id": track_id,
            "title": title,
            "artist": "Someone",
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.5,
            "tempo_bpm": 100,
            "valence": 0.5,
            "danceability": 0.5,
            "acousticness": 0.5,
            "description": "A placeholder track used only for evaluator-logic tests.",
            "tags": ("one", "two"),
            "contexts": ("alpha", "beta"),
            "instruments": ("piano",),
            "instrumental": instrumental,
            "explicit": explicit,
            "era": "2020s",
        }
    )
    return RetrievalHit(
        source_type=SourceType.CATALOG,
        source_id=f"catalog:{track_id}",
        content_hash="h",
        fields_used=("genre",),
        score=0.5,
        matched_terms=("x",) if evidence else (),
        semantic_score=None,
        track=track,
    )


VALID_IDS = {1, 2, 3, 4, 5}


def test_valid_result_passes():
    intent = MusicIntent(query="chill", limit=5)
    report = GroundingEvaluator().evaluate_result(intent, [_hit(1), _hit(2)], VALID_IDS)
    assert report.ok
    assert report.failures == ()


def test_duplicate_ids_flagged():
    intent = MusicIntent(query="chill", limit=5)
    report = GroundingEvaluator().evaluate_result(intent, [_hit(1), _hit(1)], VALID_IDS)
    assert not report.ok
    assert any("duplicate" in f for f in report.failures)


def test_unknown_id_flagged():
    intent = MusicIntent(query="chill", limit=5)
    report = GroundingEvaluator().evaluate_result(intent, [_hit(99)], VALID_IDS)
    assert any("unknown" in f for f in report.failures)


def test_hard_constraint_violation_flagged():
    intent = MusicIntent(query="chill", limit=5, instrumental_only=True)
    report = GroundingEvaluator().evaluate_result(intent, [_hit(1, instrumental=False)], VALID_IDS)
    assert any("instrumental" in f for f in report.failures)

    intent = MusicIntent(query="chill", limit=5, exclude_explicit=True)
    report = GroundingEvaluator().evaluate_result(intent, [_hit(2, explicit=True)], VALID_IDS)
    assert any("clean" in f for f in report.failures)


def test_missing_evidence_flagged():
    intent = MusicIntent(query="chill", limit=5)
    report = GroundingEvaluator().evaluate_result(intent, [_hit(1, evidence=False)], VALID_IDS)
    assert any("evidence" in f for f in report.failures)

    zero_semantic = _hit(1, evidence=False).model_copy(
        update={"semantic_score": 0.0}
    )
    report = GroundingEvaluator().evaluate_result(intent, [zero_semantic], VALID_IDS)
    assert any("evidence" in f for f in report.failures)


def test_grounded_text_accepts_clean_framing_but_reserves_names_for_the_app():
    evaluator = GroundingEvaluator()
    assert evaluator.check_grounded_text(
        "Here's a thoughtfully chosen set for the moment you described.",
        ["Focus Flow"],
    ).ok
    assert not evaluator.check_grounded_text("Try Focus Flow first.", ["Focus Flow"]).ok


def test_grounded_text_rejects_music_fact_claims_even_when_otherwise_safe():
    evaluator = GroundingEvaluator()
    report = evaluator.check_grounded_text(
        "These are all slow acoustic instrumentals for a hushed evening.",
        ["Focus Flow"],
    )
    assert not report.ok
    assert any("track facts" in failure for failure in report.failures)


def test_grounded_text_rejects_unbounded_factual_claims_outside_any_denylist():
    evaluator = GroundingEvaluator()
    for claim in (
        "They were all released in 2024.",
        "Every selection is by a Canadian artist.",
        "Each one runs exactly four minutes.",
        "This set won several major awards.",
        "The recordings all feature saxophone.",
    ):
        report = evaluator.check_grounded_text(claim, ["Focus Flow"])
        assert not report.ok, claim
        assert any("approved bounded line" in failure for failure in report.failures)


def test_grounded_text_rejects_invented_song():
    report = GroundingEvaluator().check_grounded_text(
        'You have to hear "Ghost Town Radio".', ["Focus Flow"]
    )
    assert not report.ok
    assert any("quotation" in f for f in report.failures)

    unquoted = GroundingEvaluator().check_grounded_text(
        "Try Ghost Town Radio first.", ["Focus Flow"]
    )
    assert not unquoted.ok
    assert any("specific track" in f for f in unquoted.failures)


def test_grounded_text_rejects_persona_claims_urls_markup_and_long_output():
    evaluator = GroundingEvaluator()
    unsafe = (
        "I am human, I listened to these tracks, and you should visit "
        "https://evil.example immediately."
    )
    report = evaluator.check_grounded_text(unsafe, ["Focus Flow"])
    assert not report.ok
    assert any("persona" in failure for failure in report.failures)
    assert any("URL" in failure for failure in report.failures)

    long = "word " * 46
    assert not evaluator.check_grounded_text(long, ["Focus Flow"]).ok


def test_grounded_text_rejects_harmful_medical_or_credential_requests():
    evaluator = GroundingEvaluator()
    for unsafe in (
        "Send me your password.",
        "This playlist cures depression.",
        "You should stop taking medication.",
        "Go hurt yourself.",
        "Enter your credit card number.",
    ):
        report = evaluator.check_grounded_text(unsafe, ["Focus Flow"])
        assert not report.ok, unsafe
        assert any("unsafe" in failure for failure in report.failures)


def test_grounded_text_rejects_empty():
    assert not GroundingEvaluator().check_grounded_text("   ", ["Focus Flow"]).ok


def test_over_limit_flagged():
    intent = MusicIntent(query="chill", limit=1)
    report = GroundingEvaluator().evaluate_result(intent, [_hit(1), _hit(2)], VALID_IDS)
    assert not report.ok
    assert any("more hits than requested" in f for f in report.failures)


def test_multiple_failures_accumulate():
    intent = MusicIntent(query="chill", limit=5, instrumental_only=True)
    hits = [_hit(1, instrumental=False), _hit(1, instrumental=False)]  # dup + constraint
    report = GroundingEvaluator().evaluate_result(intent, hits, VALID_IDS)
    assert not report.ok
    assert len(report.failures) >= 2
