"""Tests for the evaluation harness itself (offline, deterministic)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation import build_scenario_companion, evaluate, render_markdown, run_case
from src.recommender import load_songs
from src.retrieval import load_context_guides
from src.service import RecommendationService


ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "eval" / "cases.json").read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="module")
def catalog():
    return RecommendationService(load_songs(str(ROOT / "data" / "songs.csv"))).catalog


@pytest.fixture(scope="module")
def guides():
    return load_context_guides(str(ROOT / "data" / "context_guides"))


def _case(case_id):
    return next(c for c in CASES if c["id"] == case_id)


def test_report_structure_and_required_gate(catalog, guides):
    report = evaluate(CASES, catalog, guides, ("local_tfidf",))
    assert set(report) >= {"metadata", "summary", "by_category", "runs"}
    s = report["summary"]
    assert s["required_now_all_pass"] is True
    assert s["hard_constraint_adherence"] == 1.0
    assert s["faithfulness_ok"] is True
    # the genre gap the structured leg will fix is measured, not hidden
    assert s["genre_satisfaction_avg"] is not None


def test_results_never_contain_raw_query_text(catalog, guides):
    report = evaluate(CASES, catalog, guides, ("local_tfidf",))
    blob = json.dumps(report)
    for needle in ("alice@example", "xyzzy", "end my life", "ignore all previous", "120 bpm"):
        assert needle not in blob


def test_known_case_outcomes(catalog, guides):
    companion = build_scenario_companion("local_tfidf", catalog, guides)
    gibberish = run_case(companion, _case("gibberish_best_effort"), "local_tfidf")
    assert gibberish["action"] == "recommend" and gibberish["passed"]
    lofi = run_case(companion, _case("clean_instrumental_lofi"), "local_tfidf")
    assert lofi["action"] == "recommend"
    assert lofi["hard_constraint_ok"] is True and lofi["passed"]


def test_embedding_outage_degrades_and_still_passes(catalog, guides):
    companion = build_scenario_companion("embedding_outage", catalog, guides)
    result = run_case(companion, _case("jazz"), "embedding_outage")
    assert result["operating_mode"] == "degraded"  # fallback engaged
    assert result["passed"]  # a degraded recommend satisfies a recommend expectation


def test_fake_hybrid_is_graded_as_plumbing(catalog, guides):
    # Under fake_hybrid the gibberish case "recommends" noise; that must not fail,
    # because plumbing scenarios aren't graded on semantic quality.
    companion = build_scenario_companion("fake_hybrid", catalog, guides)
    result = run_case(companion, _case("gibberish_best_effort"), "fake_hybrid")
    assert result["passed"]


def test_markdown_renders(catalog, guides):
    md = render_markdown(evaluate(CASES, catalog, guides, ("local_tfidf",)))
    assert "# Evaluation report card" in md and "Gate:" in md
