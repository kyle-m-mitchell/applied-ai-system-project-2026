"""Run the evaluation report card.

    python scripts/evaluate.py            # run, write latest.{json,md}, print summary
    python scripts/evaluate.py --accept   # also promote latest -> baseline.json

Offline and deterministic by default. Exits non-zero if the gate fails
(required_now regressions, a hard-constraint miss, faithfulness, or a broken
fallback), so it can guard CI. Never writes baseline.json without --accept, and
never records raw query text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import OFFLINE_SCENARIOS, evaluate, render_markdown  # noqa: E402
from src.recommender import load_songs  # noqa: E402
from src.retrieval import load_context_guides  # noqa: E402
from src.service import RecommendationService  # noqa: E402


CATALOG_PATH = REPO_ROOT / "data" / "songs.csv"
GUIDES_DIR = REPO_ROOT / "data" / "context_guides"
CASES_PATH = REPO_ROOT / "eval" / "cases.json"
RESULTS_DIR = REPO_ROOT / "eval" / "results"


def _print_summary(report: dict) -> None:
    s = report["summary"]
    print(f"\nEvaluation — gate {'PASS' if s['gate_passed'] else 'FAIL'}")
    print(f"  required_now all pass : {s['required_now_all_pass']}")
    print(f"  hard-constraint 100%  : {s['hard_constraint_adherence']:.0%}")
    print(f"  faithfulness          : {s['faithfulness_ok']}")
    print(f"  embedding fallback    : {s['embedding_fallback_ok']}")
    print(f"  generation fallback   : {s['generation_fallback_ok']}")
    print(f"  genre satisfaction    : {s['genre_satisfaction_avg']} "
          f"(floor {s['min_genre_satisfaction']}: {s['genre_satisfaction_ok']})")
    print(f"  latency p50/p95 (ms)  : {s['latency_ms_p50']} / {s['latency_ms_p95']}")
    for category, counts in sorted(report["by_category"].items()):
        print(f"    {category:12} pass {counts['pass']:>3}  fail {counts['fail']:>3}")
    for run in report["runs"]:
        if not run["passed"]:
            print(f"    FAIL {run['id']} [{run['scenario']}]: {'; '.join(run['failures'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cadence evaluation report card.")
    parser.add_argument("--accept", action="store_true", help="promote latest results to baseline.json")
    parser.add_argument("--scenarios", default=",".join(OFFLINE_SCENARIOS),
                        help="comma-separated scenario list")
    args = parser.parse_args()

    scenarios = tuple(s.strip() for s in args.scenarios.split(",") if s.strip())
    tracks = RecommendationService(load_songs(str(CATALOG_PATH))).catalog
    guides = load_context_guides(str(GUIDES_DIR))
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]

    report = evaluate(cases, tracks, guides, scenarios)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS_DIR / "latest.md").write_text(render_markdown(report), encoding="utf-8")
    _print_summary(report)
    print(f"\nWrote {RESULTS_DIR.relative_to(REPO_ROOT)}/latest.json + latest.md")

    baseline = RESULTS_DIR / "baseline.json"
    if args.accept:
        baseline.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Promoted latest -> {baseline.relative_to(REPO_ROOT)} (accepted baseline)")
    elif not baseline.exists():
        print("No baseline.json yet — rerun with --accept to record this as the baseline.")

    sys.exit(0 if report["summary"]["gate_passed"] else 1)


if __name__ == "__main__":
    main()
