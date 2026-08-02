"""Deterministic product evaluation for the baked FMA catalog path.

Unlike the immutable fictional regression suite, this report separates genre
fit from numeric-feature fit and breaks numeric evidence down by provenance.
It also runs a fixed 50-query warm-latency benchmark without persisting listener
text: result rows contain only stable case IDs, decisions, references, and
metrics.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from src.companion import MusicCompanion
from src.contracts import CatalogTrack, FieldOrigin
from src.structured import goal_score


_BASE_QUERIES: tuple[str, ...] = (
    "calm independent folk",
    "upbeat folk with more energy",
    "somber acoustic folk",
    "intense independent rock",
    "upbeat rock with more movement",
    "calm jazz for reading",
    "somber blues with low energy",
    "upbeat hip hop with more movement",
    "calm classical music",
    "more instrumental electronic music",
)
_QUALIFIERS = ("", " for a late evening", " for a morning walk", " for focus", " for discovery")

# Ten musically distinct requests × five context variants. The exact tuple is
# public and stable; changing it is an evaluation-version change.
FMA_BENCHMARK_QUERIES: tuple[str, ...] = tuple(
    base + qualifier for qualifier in _QUALIFIERS for base in _BASE_QUERIES
)
assert len(FMA_BENCHMARK_QUERIES) == 50


def _percentile_95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(round(0.95 * (len(ordered) - 1)))]


def _feature_origin(track: CatalogTrack, feature: str) -> str:
    lineage = next(
        (item for item in track.lineage if item.field_name == feature), None
    )
    return lineage.origin.value if lineage is not None else FieldOrigin.UNKNOWN.value


def _genre_match(track: CatalogTrack, requested: str) -> bool:
    requested = requested.casefold()
    return requested == (track.genre or "").casefold() or requested in {
        genre.casefold() for genre in track.genres
    }


def evaluate_fma(
    companion: MusicCompanion,
    *,
    queries: Sequence[str] = FMA_BENCHMARK_QUERIES,
) -> dict[str, Any]:
    """Evaluate quality/provenance and warm latency through the public companion."""
    if companion.catalog_id != "fma":
        raise ValueError("FMA evaluation requires an FMA companion")
    if len(queries) != 50:
        raise ValueError("the product latency benchmark requires exactly 50 queries")

    # Warm SQLite page cache and Python code paths before measuring the fixed run.
    companion.respond(queries[0])
    feature_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    genre_values: list[float] = []
    runs: list[dict[str, Any]] = []
    latencies: list[float] = []

    for index, query in enumerate(queries, start=1):
        start = time.perf_counter()
        turn = companion.respond_detailed(query)
        latency_ms = (time.perf_counter() - start) * 1_000
        latencies.append(latency_ms)
        response = turn.response
        hits = response.retrieval.hits if response.retrieval else ()
        intent = response.intent

        genre_satisfaction: float | None = None
        if intent is not None and intent.genre and hits:
            genre_satisfaction = sum(
                _genre_match(hit.track, intent.genre) for hit in hits
            ) / len(hits)
            genre_values.append(genre_satisfaction)

        numeric_observations = 0
        if intent is not None:
            for hit in hits:
                for goal in intent.feature_goals:
                    satisfaction = goal_score(goal, hit.track)
                    if satisfaction is None:
                        continue
                    origin = _feature_origin(hit.track, goal.feature)
                    feature_values[origin][goal.feature].append(satisfaction)
                    numeric_observations += 1

        refs = tuple(hit.track.ref.source_id for hit in hits)
        failures: list[str] = []
        if len(refs) != len(set(refs)):
            failures.append("duplicate namespaced track references")
        if any(not ref.startswith("catalog:fma:") for ref in refs):
            failures.append("non-FMA reference in FMA result")
        runs.append(
            {
                "id": f"fma-{index:02d}",
                "action": response.action.value,
                "n_hits": len(hits),
                "track_refs": refs,
                "genre_satisfaction": (
                    round(genre_satisfaction, 4)
                    if genre_satisfaction is not None
                    else None
                ),
                "numeric_observations": numeric_observations,
                "latency_ms": round(latency_ms, 3),
                "passed": not failures and bool(hits),
                "failures": tuple(failures),
            }
        )

    by_provenance: dict[str, dict[str, dict[str, float | int]]] = {}
    for origin, features in sorted(feature_values.items()):
        by_provenance[origin] = {}
        for feature, values in sorted(features.items()):
            by_provenance[origin][feature] = {
                "mean_satisfaction": round(statistics.fmean(values), 4),
                "n": len(values),
            }

    descriptor = companion.catalog_descriptor
    p95 = _percentile_95(latencies)
    full_latency_gate: bool | None = None
    if descriptor is not None and descriptor.edition.value == "full":
        full_latency_gate = p95 < 1_000.0
    return {
        "metadata": {
            "evaluation_version": "fma-product-eval-v1",
            "catalog_artifact_id": descriptor.artifact_id if descriptor else None,
            "catalog_edition": descriptor.edition.value if descriptor else None,
            "catalog_tracks": descriptor.accepted_count if descriptor else None,
            "artifact_source": companion.catalog_artifact_source,
            "n_queries": len(queries),
            "queries_persisted": False,
        },
        "summary": {
            "passed": all(run["passed"] for run in runs),
            "genre_satisfaction_mean": (
                round(statistics.fmean(genre_values), 4) if genre_values else None
            ),
            "feature_satisfaction_by_provenance": by_provenance,
            "latency_ms_p50": round(statistics.median(latencies), 3),
            "latency_ms_p95": round(p95, 3),
            "full_catalog_p95_under_one_second": full_latency_gate,
        },
        "runs": runs,
    }


def render_fma_markdown(report: dict[str, Any]) -> str:
    """Render a concise, generated report card for committed/reviewer evidence."""
    metadata = report["metadata"]
    summary = report["summary"]
    gate = summary["full_catalog_p95_under_one_second"]
    gate_text = "not applicable (lite)" if gate is None else ("PASS" if gate else "FAIL")
    lines = [
        "# FMA product evaluation",
        "",
        f"- Artifact: `{metadata['catalog_artifact_id']}` ({metadata['catalog_edition']})",
        f"- Queries: {metadata['n_queries']} (query text persisted: no)",
        f"- Overall invariant gate: {'PASS' if summary['passed'] else 'FAIL'}",
        f"- Mean genre satisfaction: {summary['genre_satisfaction_mean']}",
        f"- Warm latency p50 / p95: {summary['latency_ms_p50']} / {summary['latency_ms_p95']} ms",
        f"- Full-catalog p95 < 1 second: {gate_text}",
        "",
        "## Numeric satisfaction by provenance",
        "",
    ]
    provenance = summary["feature_satisfaction_by_provenance"]
    if not provenance:
        lines.append("No numeric goals had usable evidence.")
    for origin, features in provenance.items():
        lines.append(f"### {origin}")
        lines.append("")
        for feature, metric in features.items():
            lines.append(
                f"- {feature}: {metric['mean_satisfaction']:.4f} "
                f"across {metric['n']} result-goal pairs"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
