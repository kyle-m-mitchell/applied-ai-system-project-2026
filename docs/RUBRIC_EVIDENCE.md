# Rubric Evidence Map

Every rubric point → where it lives, how to run it, and what you should see.
All commands run from the repo root; the whole suite and every example run
**offline with no API key**. (Setup: `python3 -m venv .venv && source
.venv/bin/activate && pip install -r requirements.txt`.)

## Core (21 pts)

| Rubric point | Evidence (file) | Command | Expected |
|---|---|---|---|
| Original project & scope | [README.md](../README.md) "How The System Works"; [src/recommender.py](../src/recommender.py) | `python -m src.main` | The deterministic 200-track scorer runs; top-5 lofi/chill with per-feature "Why" and `Operating mode: local`. |
| Substantial integrated AI feature | [src/companion.py](../src/companion.py), [src/retrieval.py](../src/retrieval.py), [src/embeddings.py](../src/embeddings.py), [src/structured.py](../src/structured.py), [src/fusion.py](../src/fusion.py) | `python -m src.main "some jazz please"` | Natural language → guard → intent (genre/mood + directional numeric goals) → hybrid retrieval + structured-preference rank fusion → mood-diverse, grounded recommendations that honor the requested genre. |
| Mermaid architecture (matches code) | [diagrams/architecture.mmd](../diagrams/architecture.mmd) (implemented) + [diagrams/roadmap.mmd](../diagrams/roadmap.mmd) (target) | render either `.mmd` | Architecture shows only implemented nodes; roadmap holds planned ones. |
| End-to-end demonstration | [src/main.py](../src/main.py) + the three examples below | see "Reproducible examples" | Three real CLI transcripts (recommend / privacy / retrieval before-after). |
| Reliability / guardrail | [src/guard.py](../src/guard.py), [src/evaluator.py](../src/evaluator.py), fallback in [src/retrieval.py](../src/retrieval.py), opt-in receipt in [src/observability.py](../src/observability.py) | `python -m src.main "my email is a@b.com, find me melancholy piano" --log` | Email redacted; `mode: local` (never the provider); still recommends; `logs/events.jsonl` records ids/scores/decisions but **no query text**. |
| README & setup | [README.md](../README.md) | `python -m pytest -q` | `157 passed`; README setup/run/test commands work as written. |
| AI-collaboration reflection | [model_card.md](../model_card.md) §9, [ai_interactions.md](../ai_interactions.md) | read | Prompting/debugging, one useful + one flawed suggestion, limits. *(Owner to personalize.)* |

## Bonus (8 pts)

| Bonus | Evidence | Command | Expected |
|---|---|---|---|
| Multi-source RAG (+2) | [src/retrieval.py](../src/retrieval.py), [data/context_guides/](../data/context_guides/) | `python scripts/retrieval_demo.py "music to concentrate"` | Catalog + curated context guides; a guide expands a bridge query; before/after shown. |
| Agentic workflow (+2) | [src/companion.py](../src/companion.py), [ai_interactions.md](../ai_interactions.md) | `python -m src.main --trace "upbeat party music"` | Bounded actions + a privacy-safe `AgentTrace` (categories/ids/decisions, no raw text). |
| Specialized behavior (+2) | [src/voice.py](../src/voice.py), [docs/CADENCE_VOICE.md](CADENCE_VOICE.md) | `python -m src.main "clean chill beats for studying"` (with a key) vs template | Cadence's grounded voice card + framing vs the deterministic baseline. |
| Evaluation harness (+2) | [scripts/evaluate.py](../scripts/evaluate.py), [eval/cases.json](../eval/cases.json), [src/evaluation.py](../src/evaluation.py) | `python scripts/evaluate.py` | Labeled cases run across a scenario matrix; prints a pass/fail gate + metrics; writes versioned JSON + Markdown; results store no query text. |

## Reproducible end-to-end examples

**1 — Recommend with hard filters (hybrid, with a key; TF-IDF fallback without):**
```
python -m src.main "clean chill beats for studying, no vocals"
```
→ instrumental lofi/ambient study tracks, `[recommend]`, `mode: gemini` (or `local`
if no cache/key), the applied filters, and per-hit evidence.

**2 — Privacy guardrail (sensitive input stays local):**
```
python -m src.main "my email is alice@example.com, find me melancholy piano"
```
→ query shown as `[redacted]`, `[recommend]`, **`mode: local`** (never the
provider), melancholy blues/r&b/classical.

**3 — Retrieval before/after (semantic beats lexical on a paraphrase):**
```
python scripts/retrieval_demo.py "tunes for cramming before an exam"
```
→ TF-IDF finds weak/wrong matches; the hybrid surfaces lofi study tracks at high
semantic similarity with **zero shared words** — reproducible from the committed cache.

## Verify everything
```
python -m pytest -q     # -> 157 passed, fully offline
```
