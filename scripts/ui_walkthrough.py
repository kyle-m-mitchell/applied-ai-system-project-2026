"""Reproducible, browser-free walkthrough of Cadence's flagship UI states.

    python scripts/ui_walkthrough.py

Drives the *real* Streamlit app with `AppTest` (no browser, no API key) and prints
what each flagship flow renders — so the UI's behavior is inspectable and citable
without a display. Provider access is disabled so it never touches a real key.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CADENCE_DISABLE_PROVIDER"] = "1"  # guarantee no provider objects

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

FLOWS = [
    ("Recommend + evidence (structured lift)", "some jazz please"),
    ("Privacy — PII redacted, stays local", "my email is a@b.com, find me melancholy piano"),
    ("Graceful safety response", "i want to end my life"),
    ("Provider-free recommend (Local-only default)", "gentle acoustic songs for writing by the window"),
]


def _run(query: str) -> AppTest:
    app = AppTest.from_file(str(REPO_ROOT / "streamlit_app.py"), default_timeout=30).run()
    app.text_input[0].set_value(query)
    next(button for button in app.button if button.label == "Build my set").click()
    return app.run()


def main() -> None:
    print("Cadence UI walkthrough — offline, no provider (AppTest)\n" + "=" * 58)
    for title, query in FLOWS:
        app = _run(query)
        turn = app.session_state["cadence_ui_session"].current.turn
        receipt = turn.receipt
        print(f"\n### {title}")
        print(f"  action           : {receipt.action.value}")
        print(f"  operating mode   : {receipt.operating_mode.value if receipt.operating_mode else '—'}")
        print(f"  force_local      : {receipt.force_local}    network_used: {receipt.network_used}")
        print(f"  guard category   : {receipt.guard_category.value}")
        print(f"  candidate → final: {len(receipt.candidate_ids)} → {len(receipt.final_ids)}")
        print(f"  latency          : {receipt.latency_ms:.2f} ms")
        comparison = turn.comparison
        if comparison is not None and comparison.structured_active:
            rows = comparison.rows
            text_top = [row.title for row in sorted(rows, key=lambda r: -r.text)[:3]]
            fused_top = [row.title for row in sorted(rows, key=lambda r: -r.fused)[:3]]
            print(f"  text-only top 3  : {text_top}")
            print(f"  fused     top 3  : {fused_top}  <- structured lift")
    print("\nEvery flow ran offline with no API key.")


if __name__ == "__main__":
    main()
