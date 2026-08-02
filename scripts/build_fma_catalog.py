#!/usr/bin/env python3
"""Build a verified FMA full or lite SQLite catalog.

Examples:
    python scripts/build_fma_catalog.py --archive /tmp/fma_metadata.zip \
        --edition lite --output data/catalogs/fma-lite.sqlite

    python scripts/build_fma_catalog.py --metadata-dir /tmp/fma_metadata \
        --predictions artifacts/fma-predictions.jsonl --edition full \
        --output artifacts/fma-full.sqlite
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.fma import (
    OFFICIAL_FMA_METADATA_SHA256,
    FmaBuildProfile,
    FmaSourcePaths,
    build_fma_catalog,
)
from src.etl.integrity import safe_extract_zip


# SHA-256 of the official ``fma_metadata.zip`` identified by the upstream
# published SHA-1 f0df49ffe5f2a6008d7dc83c6915b31835dfe733.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="official fma_metadata.zip")
    source.add_argument("--metadata-dir", type=Path, help="extracted fma_metadata directory")
    parser.add_argument(
        "--archive-sha256",
        default=OFFICIAL_FMA_METADATA_SHA256,
        help="pinned archive digest (defaults to the verified official archive)",
    )
    parser.add_argument("--predictions", type=Path, help="optional released model prediction JSONL")
    parser.add_argument("--edition", choices=("full", "lite"), default="lite")
    parser.add_argument("--lite-size", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--quarantine", type=Path)
    return parser.parse_args()


def _sources(metadata_dir: Path, predictions: Path | None) -> FmaSourcePaths:
    required = {
        "tracks": metadata_dir / "tracks.csv",
        "genres": metadata_dir / "genres.csv",
        "echonest": metadata_dir / "echonest.csv",
    }
    missing = [path.name for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"FMA metadata directory is missing: {', '.join(missing)}")
    return FmaSourcePaths(
        tracks=required["tracks"],
        genres=required["genres"],
        echonest=required["echonest"],
        predictions=predictions,
    )


def _build(args: argparse.Namespace, metadata_dir: Path) -> None:
    result = build_fma_catalog(
        _sources(metadata_dir, args.predictions),
        args.output,
        profile=FmaBuildProfile(edition=args.edition, lite_size=args.lite_size),
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )
    print(
        json.dumps(
            {
                "artifact": str(result.database_path),
                "manifest": str(result.manifest_path),
                "quarantine": str(result.quarantine_path),
                "artifact_id": result.manifest.artifact_id,
                "tracks": result.manifest.accepted_count,
                "sha256": result.manifest.artifact_sha256,
                "edition": result.manifest.edition,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    if args.metadata_dir is not None:
        _build(args, args.metadata_dir)
        return
    with tempfile.TemporaryDirectory(prefix="cadence-fma-build-") as temporary:
        extracted = Path(temporary) / "metadata"
        safe_extract_zip(
            args.archive,
            extracted,
            expected_sha256=args.archive_sha256,
            max_members=100,
            selected_files=frozenset({"tracks.csv", "genres.csv", "echonest.csv"}),
        )
        _build(args, extracted / "fma_metadata")


if __name__ == "__main__":
    main()
