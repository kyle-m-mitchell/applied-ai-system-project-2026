#!/usr/bin/env python3
"""Package an FMA SQLite database as a deterministic checksummed gzip release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.catalog_artifacts import package_catalog_gzip  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release-manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest = package_catalog_gzip(
        args.database,
        args.manifest,
        args.output,
        release_manifest_path=args.release_manifest,
    )
    print(json.dumps({
        "distribution": str(args.output),
        "distribution_sha256": manifest.distribution_sha256,
        "distribution_bytes": manifest.distribution_bytes,
        "artifact_sha256": manifest.artifact_sha256,
        "manifest": str(args.release_manifest),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
