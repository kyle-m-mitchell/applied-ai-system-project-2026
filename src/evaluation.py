"""Evaluation harness: a reproducible report card for the companion.

It runs labeled cases through the **same public `MusicCompanion` path** the CLI
uses, under an explicit **scenario matrix** (which retrieval/generation sources
ran), and reports pass/fail/skip/planned plus quality metrics. It is deterministic
and offline by default.

Two rules the reviewer insisted on, enforced here:
* results **never** contain raw or sanitized query text — only case ids,
  categories, decisions, ids, and scores;
* "no cache" is not a semantic test — each result records the exact scenario, so a
  TF-IDF run is never mistaken for a semantic one.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence

from src.companion import MusicCompanion
from src.contracts import CatalogTrack
from src.embeddings import EmbeddingCache, FakeEmbedder
from src.generation import FewShot, TextGenerator
from src.retrieval import (
    ContextGuide,
    HybridRetriever,
    build_document_text,
    embedding_content_hash,
)


# Scenarios that run fully offline and deterministically (the reproducible
# baseline). "cached_semantic" and "live" need the real cache/key and are recorded
# as skipped when unavailable, so a TF-IDF run is never mislabeled as semantic.
OFFLINE_SCENARIOS = ("local_tfidf", "fake_hybrid", "embedding_outage", "generation_outage")

# fake_hybrid exercises the hybrid *plumbing* deterministically, but the fake
# embedder has no real semantics (it finds spurious matches for gibberish). So we
# grade structural invariants there (guard, hard constraints, faithfulness), not
# semantic quality (action / min_hits).
PLUMBING_SCENARIOS = frozenset({"fake_hybrid"})

# An absolute quality floor, added once the structured leg let us measure a real
# distribution (a weight sweep peaked at 0.863). Set below the achieved value with
# headroom, so a genuine ranking regression fails the gate — not normal variation.
MIN_GENRE_SATISFACTION = 0.75


class _OutageEmbedder(FakeEmbedder):
    """A fake embedder whose query embedding always fails (provider outage)."""

    def embed_query(self, text: str):
        raise RuntimeError("embedding provider outage")


class _OutageGenerator(TextGenerator):
    """A generator that always fails (language-provider outage)."""

    model_id = "outage-generator"

    def generate(self, system: str, few_shot: FewShot, user: str) -> str:
        raise RuntimeError("generation provider outage")


def _fake_cache(tracks: Sequence[CatalogTrack], embedder: FakeEmbedder) -> EmbeddingCache:
    vectors = embedder.embed_documents([build_document_text(t) for t in tracks])
    return EmbeddingCache(
        embedder.model_id,
        embedder.dimension,
        embedding_content_hash(tracks, embedder.model_id, embedder.dimension),
        {track.id: vector for track, vector in zip(tracks, vectors)},
    )


def build_scenario_companion(scenario: str, tracks, guides) -> MusicCompanion:
    """Build a companion wired for one eval scenario (all offline, deterministic).

    Distinct from ``src.factory.build_companion``: that constructs the production
    companion from config; this injects the deterministic outage/fake doubles a
    scenario needs, so the two never share a name.
    """
    if scenario == "local_tfidf":
        return MusicCompanion(tracks, guides)  # TF-IDF default, no generator
    if scenario == "fake_hybrid":
        embedder = FakeEmbedder(dimension=64)
        hybrid = HybridRetriever(tracks, embedder, _fake_cache(tracks, embedder), guides)
        from src.generation import FakeTextGenerator

        return MusicCompanion(tracks, guides, default_retriever=hybrid, generator=FakeTextGenerator())
    if scenario == "embedding_outage":
        cache = _fake_cache(tracks, FakeEmbedder(dimension=64))
        hybrid = HybridRetriever(tracks, _OutageEmbedder(dimension=64), cache, guides)
        return MusicCompanion(tracks, guides, default_retriever=hybrid)
    if scenario == "generation_outage":
        return MusicCompanion(tracks, guides, generator=_OutageGenerator())
    raise ValueError(f"unknown scenario: {scenario}")


def _has_hard_constraint(expect: dict) -> bool:
    return bool(expect.get("instrumental_only") or expect.get("exclude_explicit"))


def run_case(companion: MusicCompanion, case: dict, scenario: str) -> dict:
    """Run one case and return a privacy-safe result (no query text)."""
    expect = case.get("expect", {})
    start = time.perf_counter()
    response = companion.respond(case["query"], limit=case.get("limit", 5))
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    hits = response.retrieval.hits if response.retrieval else ()
    filters = response.retrieval.filters_applied if response.retrieval else ()
    trace = response.trace
    action = response.action.value
    plumbing = scenario in PLUMBING_SCENARIOS
    failures: list[str] = []

    expected_action = expect.get("action")
    if expected_action is not None and not plumbing:
        allowed = [expected_action] if isinstance(expected_action, str) else list(expected_action)
        if "recommend" in allowed and "degraded" not in allowed:
            allowed = allowed + ["degraded"]  # a fallback (degraded) recommend is still a recommend
        if action not in allowed:
            failures.append(f"action {action} not in {allowed}")

    # The guard runs before retrieval, so its category is meaningful in every scenario.
    if "guard_category" in expect and trace and trace.guard_category.value != expect["guard_category"]:
        failures.append(f"guard {trace.guard_category.value} != {expect['guard_category']}")

    hard_ok: bool | None = None
    if _has_hard_constraint(expect):
        hard_ok = True
        if expect.get("instrumental_only") and (
            not all(h.track.instrumental for h in hits) or "instrumental_only" not in filters
        ):
            failures.append("instrumental_only not enforced")
            hard_ok = False
        if expect.get("exclude_explicit") and any(h.track.explicit for h in hits):
            failures.append("explicit track leaked")
            hard_ok = False

    if not plumbing and "min_hits" in expect and len(hits) < expect["min_hits"]:
        failures.append(f"hits {len(hits)} < min {expect['min_hits']}")

    ids = [h.track.id for h in hits]
    if len(ids) != len(set(ids)):
        failures.append("duplicate track ids")

    genres_any = expect.get("genres_any")
    genre_satisfaction = None
    if genres_any and hits:
        genre_satisfaction = round(sum(1 for h in hits if h.track.genre in genres_any) / len(hits), 3)

    return {
        "id": case["id"],
        "category": case["category"],
        "scenario": scenario,
        "action": action,
        "operating_mode": response.retrieval.operating_mode.value if response.retrieval else None,
        "voice_source": trace.voice_source.value if trace else None,
        "guard_category": trace.guard_category.value if trace else None,
        "n_hits": len(hits),
        "retrieved_ids": ids,
        "hard_constraint_ok": hard_ok,
        "genre_satisfaction": genre_satisfaction,
        "latency_ms": latency_ms,
        "passed": not failures,
        "failures": failures,
    }  # NOTE: deliberately no query text


def _code_version() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def evaluate(cases, tracks, guides, scenarios: Sequence[str] = OFFLINE_SCENARIOS) -> dict:
    """Run every case under every scenario and return an aggregated report."""
    companions = {
        scenario: build_scenario_companion(scenario, tracks, guides) for scenario in scenarios
    }
    valid_ids = {track.id for track in tracks}
    runs = [
        run_case(companions[scenario], case, scenario)
        for case in cases
        for scenario in scenarios
    ]
    return _aggregate(runs, valid_ids, scenarios, tracks)


def _aggregate(runs, valid_ids, scenarios, tracks) -> dict:
    from datetime import datetime, timezone

    from src.retrieval import TfidfRetriever

    by_category: dict[str, dict[str, int]] = {}
    for run in runs:
        bucket = by_category.setdefault(run["category"], {"pass": 0, "fail": 0})
        bucket["pass" if run["passed"] else "fail"] += 1

    required = [r for r in runs if r["category"] == "required_now"]
    required_pass = all(r["passed"] for r in required)

    hard_runs = [r for r in runs if r["hard_constraint_ok"] is not None]
    hard_adherence = (
        sum(1 for r in hard_runs if r["hard_constraint_ok"]) / len(hard_runs) if hard_runs else 1.0
    )
    faithfulness_ok = all(all(i in valid_ids for i in r["retrieved_ids"]) for r in runs)

    genre_vals = [r["genre_satisfaction"] for r in runs if r["genre_satisfaction"] is not None]
    genre_avg = round(sum(genre_vals) / len(genre_vals), 3) if genre_vals else None

    # Under an embedding outage, a NON-sensitive request that returns hits must be
    # labeled degraded. Sensitive requests legitimately stay on the local retriever.
    emb_out = [
        r for r in runs
        if r["scenario"] == "embedding_outage" and r["guard_category"] != "sensitive" and r["n_hits"] > 0
    ]
    emb_fallback_ok = all(r["operating_mode"] == "degraded" for r in emb_out) if emb_out else True
    gen_out = [r for r in runs if r["scenario"] == "generation_outage" and r["action"] == "recommend"]
    gen_fallback_ok = all(r["voice_source"] == "template" for r in gen_out) if gen_out else True

    latencies = sorted(r["latency_ms"] for r in runs)
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = latencies[int(round(0.95 * (len(latencies) - 1)))] if latencies else 0.0

    # Absolute quality floor: genre satisfaction must clear MIN_GENRE_SATISFACTION
    # (skipped only if no case measured it). This is the "measure, then gate" step.
    genre_ok = genre_avg is None or genre_avg >= MIN_GENRE_SATISFACTION

    gate_passed = (
        required_pass and hard_adherence == 1.0 and faithfulness_ok
        and emb_fallback_ok and gen_fallback_ok and genre_ok
    )

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_version": _code_version(),
            "catalog_fingerprint": TfidfRetriever(tracks).index_fingerprint[:16],
            "scenarios": list(scenarios),
            "seed": "deterministic",
            "n_cases": len({r["id"] for r in runs}),
            "n_runs": len(runs),
        },
        "summary": {
            "gate_passed": gate_passed,
            "required_now_all_pass": required_pass,
            "hard_constraint_adherence": round(hard_adherence, 3),
            "faithfulness_ok": faithfulness_ok,
            "embedding_fallback_ok": emb_fallback_ok,
            "generation_fallback_ok": gen_fallback_ok,
            "genre_satisfaction_avg": genre_avg,
            "genre_satisfaction_ok": genre_ok,
            "min_genre_satisfaction": MIN_GENRE_SATISFACTION,
            "latency_ms_p50": round(p50, 2),
            "latency_ms_p95": round(p95, 2),
        },
        "by_category": by_category,
        "runs": runs,
    }


def render_markdown(report: dict) -> str:
    """Render a readable pass/fail report (no query text)."""
    meta, summary = report["metadata"], report["summary"]
    lines = [
        "# Evaluation report card",
        "",
        f"- Generated: {meta['timestamp']}  ·  code `{meta['code_version']}`  ·  "
        f"catalog `{meta['catalog_fingerprint']}`  ·  seed {meta['seed']}",
        f"- Scenarios: {', '.join(meta['scenarios'])}  ·  "
        f"{meta['n_cases']} cases × {len(meta['scenarios'])} scenarios = {meta['n_runs']} runs",
        "",
        f"## Gate: {'PASS ✅' if summary['gate_passed'] else 'FAIL ❌'}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| required_now all pass | {summary['required_now_all_pass']} |",
        f"| hard-constraint adherence | {summary['hard_constraint_adherence']:.0%} (target 100%) |",
        f"| catalog faithfulness | {summary['faithfulness_ok']} |",
        f"| embedding-outage fallback | {summary['embedding_fallback_ok']} |",
        f"| generation-outage fallback | {summary['generation_fallback_ok']} |",
        f"| genre satisfaction (avg) | {summary['genre_satisfaction_avg']} "
        f"(floor {summary['min_genre_satisfaction']}: {summary['genre_satisfaction_ok']}) |",
        f"| latency p50 / p95 (ms) | {summary['latency_ms_p50']} / {summary['latency_ms_p95']} |",
        "",
        "## By category",
        "",
        "| Category | Pass | Fail |",
        "|---|---:|---:|",
    ]
    for category, counts in sorted(report["by_category"].items()):
        lines.append(f"| {category} | {counts['pass']} | {counts['fail']} |")

    failing = [r for r in report["runs"] if not r["passed"]]
    lines += ["", f"## Failing runs ({len(failing)})", ""]
    if failing:
        lines += ["| case | category | scenario | action | reasons |", "|---|---|---|---|---|"]
        for r in failing:
            lines.append(
                f"| {r['id']} | {r['category']} | {r['scenario']} | {r['action']} | "
                f"{'; '.join(r['failures'])} |"
            )
    else:
        lines.append("_none_")
    return "\n".join(lines) + "\n"
