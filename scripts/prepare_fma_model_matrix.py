#!/usr/bin/env python3
"""Prepare the deterministic 518-Librosa/Echo-Nest training matrix."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.fma import (  # noqa: E402
    OFFICIAL_FMA_METADATA_SHA256,
    FmaSourcePaths,
    prepare_model_matrix,
)
from src.etl.integrity import safe_extract_zip  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--archive-sha256", default=OFFICIAL_FMA_METADATA_SHA256)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _prepare(metadata: Path, output: Path) -> None:
    required = {name: metadata / f"{name}.csv" for name in ("tracks", "genres", "echonest", "features")}
    missing = [path.name for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"FMA metadata directory is missing: {', '.join(missing)}")
    result = prepare_model_matrix(
        FmaSourcePaths(
            tracks=required["tracks"],
            genres=required["genres"],
            echonest=required["echonest"],
            features=required["features"],
        ),
        output,
    )
    print(json.dumps({
        "matrix": str(result.output_path),
        "rows": result.row_count,
        "features": result.feature_count,
        "sha256": result.sha256,
    }, indent=2, sort_keys=True))


def main() -> None:
    args = _arguments()
    if args.metadata_dir is not None:
        _prepare(args.metadata_dir, args.output)
        return
    with tempfile.TemporaryDirectory(prefix="cadence-fma-matrix-") as temporary:
        extracted = Path(temporary) / "metadata"
        safe_extract_zip(
            args.archive,
            extracted,
            expected_sha256=args.archive_sha256,
            max_members=100,
            selected_files=frozenset(
                {"tracks.csv", "genres.csv", "echonest.csv", "features.csv"}
            ),
        )
        _prepare(extracted / "fma_metadata", args.output)


if __name__ == "__main__":
    main()
