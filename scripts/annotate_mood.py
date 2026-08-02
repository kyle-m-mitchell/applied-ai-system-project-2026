"""Local terminal harness for primary and independent audit mood labels.

Set ``CADENCE_LOCAL_ANNOTATION=1`` to acknowledge that this is a local research
tool. It is intentionally unavailable in the deployed Streamlit application.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.annotation import (  # noqa: E402
    AnnotationItem,
    MoodAnnotation,
    append_annotation,
    assess_annotation_readiness,
    load_annotations,
)


def _prompt_float(label: str) -> float:
    while True:
        value = input(f"{label} [0.0-1.0]: ").strip()
        try:
            parsed = float(value)
        except ValueError:
            print("Enter a number from 0.0 to 1.0.")
            continue
        if 0.0 <= parsed <= 1.0:
            return parsed
        print("Enter a number from 0.0 to 1.0.")


def _prompt_choice(label: str, choices: tuple[str, ...]) -> str:
    while True:
        value = input(f"{label} [{'/'.join(choices)}]: ").strip().lower()
        if value in choices:
            return value
        print(f"Choose one of: {', '.join(choices)}")


def _load_items(path: Path) -> tuple[AnnotationItem, ...]:
    items: list[AnnotationItem] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                items.append(AnnotationItem(**json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid sample item on line {line_number}") from exc
    return tuple(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label local FMA mood audit items.")
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rater", required=True, help="pseudonymous rater ID")
    parser.add_argument("--role", required=True, choices=("primary", "audit"))
    parser.add_argument("--limit", type=int, default=0, help="0 means no session limit")
    args = parser.parse_args()
    if os.environ.get("CADENCE_LOCAL_ANNOTATION") != "1":
        parser.error(
            "local harness disabled; set CADENCE_LOCAL_ANNOTATION=1 (never set it in deployment)"
        )

    prior = load_annotations(args.output)
    already_labeled = {
        annotation.track_key
        for annotation in prior
        if annotation.rater_id == args.rater
    }
    pending = [item for item in _load_items(args.sample) if item.key not in already_labeled]
    if args.limit > 0:
        pending = pending[: args.limit]

    added = 0
    for item in pending:
        print(f"\n{item.title} — {item.artist}")
        print(f"genre: {item.genre}")
        if item.source_url:
            print(f"source: {item.source_url}")
        if input("Press Enter to label, or q to stop: ").strip().lower() == "q":
            break
        valence = _prompt_float("Valence (negative → positive)")
        arousal = _prompt_float("Arousal (calm → energetic)")
        quadrant = _prompt_choice(
            "Quadrant", ("upbeat", "calm", "intense", "somber")
        )
        confidence = int(_prompt_choice("Confidence", ("1", "2", "3", "4", "5")))
        annotation = MoodAnnotation.create(
            item=item,
            rater_id=args.rater,
            role=args.role,
            valence=valence,
            arousal=arousal,
            quadrant=quadrant,  # type: ignore[arg-type]
            confidence=confidence,
        )
        append_annotation(args.output, annotation)
        added += 1

    readiness = assess_annotation_readiness(load_annotations(args.output))
    print(f"\nRecorded {added} labels. Status remains: {readiness.status}.")
    print(
        "Primary tracks / independent audits: "
        f"{readiness.primary_tracks} / {readiness.independent_audit_pairs}"
    )


if __name__ == "__main__":
    main()
