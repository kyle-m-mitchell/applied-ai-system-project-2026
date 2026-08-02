#!/usr/bin/env python3
"""Run the fixed 50-query FMA quality/provenance/latency report."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.etl.manifest import CatalogManifest  # noqa: E402
from src.factory import CompanionConfig, build_companion  # noqa: E402
from src.fma_evaluation import evaluate_fma, render_fma_markdown  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=ROOT / "data" / "catalogs" / "fma-lite.sqlite",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "catalogs" / "fma-lite.manifest.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "eval" / "results" / "fma-latest.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "eval" / "results" / "fma-latest.md",
    )
    return parser.parse_args()


def _memory_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return value / (1024**2) if sys.platform == "darwin" else value / 1024


def main() -> None:
    args = _args()
    manifest = CatalogManifest.read(args.manifest)
    lite_database = ROOT / "data" / "catalogs" / "fma-lite.sqlite"
    lite_manifest = ROOT / "data" / "catalogs" / "fma-lite.manifest.json"
    config_kwargs = {
        "catalog_id": "fma",
        "fma_lite_path": str(lite_database),
        "fma_lite_manifest_path": str(lite_manifest),
    }
    if manifest.edition == "full":
        config_kwargs.update(
            {
                "fma_local_full_path": str(args.database),
                "fma_local_full_manifest_path": str(args.manifest),
            }
        )
    elif args.database.resolve() != lite_database.resolve():
        raise ValueError("a lite evaluation must use the committed fallback path")

    open_start = time.perf_counter()
    companion = build_companion(CompanionConfig(**config_kwargs))
    open_ms = (time.perf_counter() - open_start) * 1_000
    report = evaluate_fma(companion)
    report["metadata"].update(
        {
            "database_open_ms": round(open_ms, 3),
            "process_peak_memory_mib": round(_memory_mib(), 3),
            "artifact_bytes": args.database.stat().st_size,
            "first_download_ms": None,
            "first_download_note": "not exercised by this local artifact run",
        }
    )

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_fma_markdown(report), encoding="utf-8")
    summary = report["summary"]
    print("FMA product evaluation")
    print(f"  gate            : {'PASS' if summary['passed'] else 'FAIL'}")
    print(f"  queries         : {report['metadata']['n_queries']}")
    print(f"  genre fit       : {summary['genre_satisfaction_mean']}")
    print(f"  latency p50/p95 : {summary['latency_ms_p50']} / {summary['latency_ms_p95']} ms")
    print(f"  report          : {args.json_output}")


if __name__ == "__main__":
    main()
