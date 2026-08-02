"""Optional, isolated non-commercial DEAM comparison.

This script never downloads DEAM and does not train or calibrate the production
model. Provide a local derived CSV with columns:

``track_id,predicted_energy,predicted_valence,human_arousal,human_valence``

The required acknowledgement and output-path restriction keep DEAM's
CC BY-NC research use visibly separate from product artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
NONCOMMERCIAL_ROOT = (REPO_ROOT / "eval" / "noncommercial").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mood import compute_mood_profile  # noqa: E402


def _human_quadrant(arousal: float, valence: float) -> str:
    if arousal >= 0.5 and valence >= 0.5:
        return "upbeat"
    if arousal < 0.5 and valence >= 0.5:
        return "calm"
    if arousal >= 0.5 and valence < 0.5:
        return "intense"
    return "somber"


def benchmark(rows: list[dict[str, str]]) -> dict[str, object]:
    arousal_errors: list[float] = []
    valence_errors: list[float] = []
    quadrant_hits = 0
    labeled = 0
    abstained = 0
    for row in rows:
        predicted_energy = float(row["predicted_energy"])
        predicted_valence = float(row["predicted_valence"])
        human_arousal = float(row["human_arousal"])
        human_valence = float(row["human_valence"])
        values = (predicted_energy, predicted_valence, human_arousal, human_valence)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("all benchmark axes must be finite values from 0 to 1")
        arousal_errors.append(abs(predicted_energy - human_arousal))
        valence_errors.append(abs(predicted_valence - human_valence))
        profile = compute_mood_profile(predicted_energy, predicted_valence)
        if profile is None or profile.label is None:
            abstained += 1
            continue
        labeled += 1
        quadrant_hits += profile.label == _human_quadrant(human_arousal, human_valence)
    if not rows:
        raise ValueError("benchmark input cannot be empty")
    return {
        "license_boundary": "DEAM CC BY-NC; isolated non-commercial benchmark only",
        "production_effect": "none",
        "row_count": len(rows),
        "arousal_mae": sum(arousal_errors) / len(arousal_errors),
        "valence_mae": sum(valence_errors) / len(valence_errors),
        "quadrant_accuracy_on_labeled": quadrant_hits / labeled if labeled else None,
        "abstention_rate": abstained / len(rows),
    }


def _validated_output(path: Path) -> Path:
    destination = path.resolve()
    try:
        destination.relative_to(NONCOMMERCIAL_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"DEAM output must stay under {NONCOMMERCIAL_ROOT.relative_to(REPO_ROOT)}"
        ) from exc
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated DEAM research comparison.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--acknowledge-noncommercial",
        action="store_true",
        help="required acknowledgement of DEAM's CC BY-NC boundary",
    )
    args = parser.parse_args()
    if not args.acknowledge_noncommercial:
        parser.error("--acknowledge-noncommercial is required")

    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report = benchmark(rows)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.output:
        try:
            destination = _validated_output(args.output)
        except ValueError as exc:
            parser.error(str(exc))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
