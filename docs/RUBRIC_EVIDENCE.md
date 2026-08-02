# Project 4 Rubric Evidence Map

This map points from each grading criterion to code, an executable check, and an
honest expected result. Run commands from the repository root. Optional provider
tests use fakes; the normal suite does not require an API key or network.

## Required features — 21 points

| Criterion | Evidence | How to verify | Expected evidence |
|---|---|---|---|
| Base project and original scope (3) | [`README.md`](../README.md), [`src/recommender.py`](../src/recommender.py), [`src/main.py`](../src/main.py) | `python -m src.main` | Structured fictional taste profile → deterministic weighted top five. README identifies the original 20-track limitations and Phase 5 extension. |
| Substantial integrated AI feature (3) | [`src/companion.py`](../src/companion.py), [`src/retrieval.py`](../src/retrieval.py), [`src/fma_store.py`](../src/fma_store.py), [`src/modeling.py`](../src/modeling.py), [`src/research.py`](../src/research.py), [`ui/`](../ui/) | `streamlit run streamlit_app.py`; submit `some jazz please`; inspect tests named below | Guarded language → catalog-aware retrieval/structured ranking → evaluator → Cadence response. Specialized predictions are baked into FMA artifacts; research is a per-result post-ranking action, not a detached demo. |
| Mermaid architecture (3) | [`diagrams/architecture.mmd`](../diagrams/architecture.mmd) | Render the `.mmd` in Mermaid Live or `mmdc` | Source shows offline ETL/model path, FMA text + structured retrieval, fusion, evaluator/fallback, research/citation guard, annotation/human promotion, isolated DEAM, tests, and human review. |
| Functional end-to-end demonstration (3) | [`streamlit_app.py`](../streamlit_app.py), [`ui/`](../ui/), [`src/main.py`](../src/main.py), examples below | Run the UI or three CLI examples | Same application pipeline returns recommend/clarify/no-match/safe/degraded states with evidence. Result IDs come from the selected catalog. |
| Reliability/evaluation/guardrail (3) | [`src/guard.py`](../src/guard.py), [`src/evaluator.py`](../src/evaluator.py), [`src/catalog_artifacts.py`](../src/catalog_artifacts.py), [`src/observability.py`](../src/observability.py), [`scripts/evaluate.py`](../scripts/evaluate.py) | `python -m src.main --local-only --trace "my email is alice@example.com, find me melancholy piano"`; `python scripts/evaluate.py` | Email is redacted, provider use is blocked, local results remain; evaluator checks grounded IDs/evidence; corrupt artifacts fall back; report gate preserves the fictional `0.863` control. |
| README/setup (3) | [`README.md`](../README.md), [`docs/PROJECT_HANDBOOK.md`](PROJECT_HANDBOOK.md), [`docs/CATALOG_DATA_CARD.md`](CATALOG_DATA_CARD.md), [`docs/LICENSING.md`](LICENSING.md) | Follow Quick start; `python -m pytest -q` | Installation, UI/CLI/test/build commands, sample behavior, architecture, data/model boundaries, and known pending release evidence are documented. |
| Reflection on AI collaboration and design (3) | [`ai_interactions.md`](../ai_interactions.md), [`model_card.md`](../model_card.md) | Read Phase 5 action/observation/decision trace and reflection | Identifies prompts/uses, a helpful suggestion, flawed suggestions, fixes, limits, and future evidence gates. Owner should personalize before submission. |

## Stretch features — up to 8 points

| Bonus | Evidence | How to verify | Expected evidence |
|---|---|---|---|
| Multi-source/custom RAG (+2) | [`src/retrieval.py`](../src/retrieval.py), [`data/context_guides/`](../data/context_guides/), [`src/fma_store.py`](../src/fma_store.py) | `python scripts/retrieval_demo.py "music to concentrate"`; run FMA store tests | Fictional catalog + guide expansion and FMA's independently generated FTS5 + structured candidates demonstrate custom/multi-source retrieval with provenance. Data card explains why retrieval improves sparse numeric discovery. |
| Agentic workflow (+2) | [`src/companion.py`](../src/companion.py), [`src/research.py`](../src/research.py), [`ai_interactions.md`](../ai_interactions.md) | `python -m src.main --trace "upbeat party music"`; `python -m pytest tests/test_research.py -q` | Bounded recommendation actions and optional identity→grounded search→citation/fallback tool workflow. Trace logs observable steps/outcomes, never hidden chain-of-thought. |
| Specialized behavior (+2) | [`src/modeling.py`](../src/modeling.py), [`src/mood.py`](../src/mood.py), [`src/voice.py`](../src/voice.py), [`model_card.md`](../model_card.md) | `python -m pytest tests/test_modeling.py tests/test_mood.py -q`; inspect a generated real model report when available | Artist-split target models compare against Dummy/Ridge, quantify uncertainty, release or abstain, and feed a separate experimental mood profile. Model card states that real metrics are pending until a pinned build. |
| Evaluation harness (+2) | [`scripts/evaluate.py`](../scripts/evaluate.py), [`eval/cases.json`](../eval/cases.json), [`src/evaluation.py`](../src/evaluation.py), Phase 5 tests | `python scripts/evaluate.py`; `python -m pytest -q` | Scenario matrix prints a pass/fail summary; specialized tests cover unknowns, deterministic ETL, SQLite/resolver, model gates, mood, annotation, and guarded research. |

## Three end-to-end examples

### 1. Ordinary recommendation

```bash
python -m src.main "some jazz please"
```

Representative output:

```text
Here are a few picks for that:
1. After Midnight Set — East Ferry Trio [jazz · romantic]
2. Coffee Shop Stories — Slow Stereo [jazz · relaxed]
...
[degraded] · mode: degraded · voice: template
```

Without a live provider, `degraded` is an honest capability label, not a failure to
recommend.

### 2. Privacy guard and local fallback

```bash
python -m src.main --local-only --trace \
  "my email is alice@example.com, find me melancholy piano"
```

Expected behavior:

```text
You asked: "my email is [redacted], find me melancholy piano"
... grounded local recommendations ...
trace: guard_category=sensitive ... network_used=False ...
```

The exact raw email must not appear in output, receipts, or JSONL events.

### 3. Unsupported FMA hard capability

In the UI, select the FMA catalog and ask:

```text
clean instrumental music for focus
```

Expected behavior:

```text
I can't verify clean lyrics or an instrumental-only boolean in this catalog,
and I'd rather not guess. Remove that requirement or switch catalogs.
```

“More instrumental” may remain a soft preference when a trustworthy
instrumentalness number exists.

## Phase 5 focused verification

```bash
python -m pytest \
  tests/test_phase5_contracts.py \
  tests/test_phase5_companion.py \
  tests/test_fma_catalog_subsystem.py \
  tests/test_modeling.py \
  tests/test_mood.py \
  tests/test_annotation.py \
  tests/test_research.py -q
```

Run the full suite and generated evaluation report:

```bash
python -m pytest --collect-only -q
python -m pytest -q
python scripts/evaluate.py
```

Test count is deliberately generated, not copied into this file. The accepted
fictional evaluation control is `0.863` average genre satisfaction.

## Evidence that remains pending before a launch claim

The rubric can be demonstrated with the integrated code and fixture-backed tests,
but the larger Phase 5 product plan requires real-build evidence that must not be
fabricated:

- actual FMA accepted/quarantined counts and field coverage;
- generated target-by-target MAE, R², interval, retained-coverage, and release
  decisions;
- committed verified 300-track Lite database and manifest;
- published checksummed Full release asset;
- deterministic rebuild comparison and corrupt-release fallback smoke test;
- measured p95/open-time/memory/size/download report on a named machine;
- 300 primary + 60 independent human mood labels and a documented promotion
  decision;
- connected-browser accessibility and hosted deployment review.

Those are visible checklist items in the project handbook, not implied by unit
tests.
