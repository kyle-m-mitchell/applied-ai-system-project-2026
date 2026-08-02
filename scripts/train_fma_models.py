"""Train gated FMA feature models and export baked, abstaining predictions.

The input is a prepared CSV produced by the FMA ETL. It must contain
``track_id``, ``artist``, normalized Librosa columns, and any requested Echo
Nest target columns. The product runtime consumes only the JSONL output.

Example:
    python3 scripts/train_fma_models.py \
      --input build/fma_model_matrix.csv \
      --feature-prefix librosa__ \
      --predictions-output build/fma_predictions.jsonl \
      --report-output build/fma_model_report.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling import (  # noqa: E402
    FEATURE_SPECS,
    train_feature_models,
    write_model_report,
    write_prediction_jsonl,
)


def _parse_names(value: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in value.split(",") if name.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train offline FMA feature models with release/abstention gates."
    )
    parser.add_argument("--input", required=True, type=Path, help="prepared CSV matrix")
    parser.add_argument(
        "--feature-prefix",
        default="librosa__",
        help="select model input columns by prefix (default: librosa__)",
    )
    parser.add_argument(
        "--feature-columns",
        default="",
        help="optional comma-separated columns; overrides --feature-prefix",
    )
    parser.add_argument(
        "--targets",
        default=",".join(FEATURE_SPECS),
        help="comma-separated target fields",
    )
    parser.add_argument("--track-id-column", default="track_id")
    parser.add_argument("--artist-column", default="artist")
    parser.add_argument("--predictions-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment-specific
        parser.error(
            "pandas is unavailable; install the pinned requirements-ml.txt environment"
        )
        raise AssertionError from exc

    frame = pd.read_csv(args.input, low_memory=False)
    feature_columns = _parse_names(args.feature_columns)
    if not feature_columns:
        feature_columns = tuple(
            column for column in frame.columns if column.startswith(args.feature_prefix)
        )
    if not feature_columns:
        parser.error("no Librosa feature columns matched the requested prefix/list")

    run = train_feature_models(
        frame,
        feature_columns=feature_columns,
        targets=_parse_names(args.targets),
        track_id_column=args.track_id_column,
        artist_column=args.artist_column,
    )
    write_prediction_jsonl(args.predictions_output, run.predictions)
    write_model_report(args.report_output, run.report)

    print("FMA specialized model build")
    print(f"  rows/features : {len(frame)} / {len(feature_columns)}")
    print(f"  predictions   : {len(run.predictions)}")
    for feature, report in run.report["features"].items():
        print(f"  {feature:18} {report['status']}")
    print(f"  wrote {args.predictions_output}")
    print(f"  wrote {args.report_output}")


if __name__ == "__main__":
    main()
