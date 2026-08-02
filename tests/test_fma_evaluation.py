"""Offline checks for the fixed FMA product report."""

from __future__ import annotations

import json
from pathlib import Path

from src.factory import CompanionConfig, build_companion
from src.fma_evaluation import FMA_BENCHMARK_QUERIES, evaluate_fma, render_fma_markdown


ROOT = Path(__file__).resolve().parents[1]


def _companion():
    return build_companion(
        CompanionConfig(
            catalog_id="fma",
            fma_lite_path=str(ROOT / "data" / "catalogs" / "fma-lite.sqlite"),
            fma_lite_manifest_path=str(
                ROOT / "data" / "catalogs" / "fma-lite.manifest.json"
            ),
        )
    )


def test_fixed_benchmark_reports_provenance_without_persisting_queries():
    report = evaluate_fma(_companion())
    blob = json.dumps(report)

    assert report["metadata"]["n_queries"] == 50
    assert report["metadata"]["queries_persisted"] is False
    assert report["summary"]["passed"]
    assert "echonest_computed" in report["summary"]["feature_satisfaction_by_provenance"]
    assert report["summary"]["full_catalog_p95_under_one_second"] is None
    assert all(query not in blob for query in FMA_BENCHMARK_QUERIES)


def test_fma_markdown_is_generated_from_report_values():
    report = evaluate_fma(_companion())
    markdown = render_fma_markdown(report)
    assert "# FMA product evaluation" in markdown
    assert str(report["summary"]["latency_ms_p95"]) in markdown
    assert "not applicable (lite)" in markdown
