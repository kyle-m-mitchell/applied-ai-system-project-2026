"""Build a deterministic, prediction-free local mood annotation sample.

Input is catalog JSONL. Output contains only the explicit ``AnnotationItem``
fields, so model estimates cannot leak into the screen used by a human rater.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.annotation import select_annotation_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local 300-track mood audit set.")
    parser.add_argument("--input", required=True, type=Path, help="catalog JSONL")
    parser.add_argument("--output", required=True, type=Path, help="sample JSONL")
    parser.add_argument("--size", type=int, default=300)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                parser.error(f"invalid input JSON on line {line_number}: {exc}")
            if not isinstance(payload, dict):
                parser.error(f"line {line_number} is not a JSON object")
            records.append(payload)

    sample = select_annotation_sample(records, size=args.size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item in sample:
            handle.write(
                json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
            )
    temporary.replace(args.output)
    print(f"Wrote {len(sample)} prediction-free items to {args.output}")


if __name__ == "__main__":
    main()
