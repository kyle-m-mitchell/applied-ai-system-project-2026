# Rubric Evidence Map

Each grading criterion → code, an executable check, and an honest expected result.
Run commands from the repository root; the normal suite needs no API key or network
(optional provider tests use fakes).

## Required features (21 pts)

| Criterion | Evidence | Verify | Expected |
|---|---|---|---|
| Base project & original scope | [`src/recommender.py`](../src/recommender.py), [`src/main.py`](../src/main.py), [README](../README.md) | `python -m src.main --structured-demo` | structured fictional profile → deterministic weighted top-five (lofi/chill, per-feature "Why", `Operating mode: local`); README states the original 20-track limits and the extension |
| Substantial integrated AI feature | [`companion`](../src/companion.py), [`retrieval`](../src/retrieval.py), [`fma_store`](../src/fma_store.py), [`modeling`](../src/modeling.py), [`research`](../src/research.py), [`ui/`](../ui/) | `streamlit run streamlit_app.py`; submit `some jazz please` | guarded language → catalog-aware retrieval + structured ranking → evaluator → response; predictions baked into FMA artifacts; research is a per-result post-ranking action |
| Mermaid architecture (matches code) | [`diagrams/architecture.mmd`](../diagrams/architecture.mmd) | render in Mermaid Live or `mmdc` | offline ETL/model path, FMA text + structured retrieval, fusion, evaluator/fallback, research/citation guard, isolated DEAM, tests, human review |
| End-to-end demonstration | [`streamlit_app.py`](../streamlit_app.py), [`ui/`](../ui/), [`src/main.py`](../src/main.py) | run the UI or the CLI examples below | one pipeline returns recommend/clarify/no-match/safe/degraded with evidence; ids come from the selected catalog |
| Reliability / guardrail | [`guard`](../src/guard.py), [`evaluator`](../src/evaluator.py), [`catalog_artifacts`](../src/catalog_artifacts.py), [`observability`](../src/observability.py), [`evaluate.py`](../scripts/evaluate.py) | example 2 below; `python scripts/evaluate.py` | email redacted, provider blocked, local results remain; evaluator checks grounded ids/evidence; corrupt artifacts fall back; gate holds the fictional `0.863` control |
| README & setup | [README](../README.md), [handbook](PROJECT_HANDBOOK.md), [data card](CATALOG_DATA_CARD.md), [licensing](LICENSING.md) | follow Quick start; `python -m pytest -q` | install/run/test/build commands, sample behavior, architecture, data/model boundaries, and pending release evidence documented |
| AI-collaboration reflection | [`ai_interactions.md`](../ai_interactions.md), [`model_card.md`](../model_card.md) | read | prompting/uses, one useful + flawed suggestions, fixes, and limits *(owner personalizes before submission)* |

## Stretch features (up to 8 pts)

| Bonus | Evidence | Verify | Expected |
|---|---|---|---|
| Multi-source RAG (+2) | [`retrieval`](../src/retrieval.py), [`data/context_guides/`](../data/context_guides/), [`fma_store`](../src/fma_store.py) | `python scripts/retrieval_demo.py "music to concentrate"` | fictional catalog + guide expansion, and FMA's independent FTS5 + structured candidates, with provenance |
| Agentic workflow (+2) | [`companion`](../src/companion.py), [`research`](../src/research.py) | `python -m src.main --trace "upbeat party music"`; `pytest tests/test_research.py -q` | bounded actions + an identity→grounded-search→citation/fallback tool workflow; the trace logs steps/outcomes, never hidden reasoning |
| Specialized behavior (+2) | [`modeling`](../src/modeling.py), [`mood`](../src/mood.py), [`voice`](../src/voice.py), [`model_card.md`](../model_card.md) | `pytest tests/test_modeling.py tests/test_mood.py -q` | artist-split models vs. Dummy/Ridge, uncertainty, release-or-abstain, feeding an experimental mood profile (real metrics pending a pinned build) |
| Evaluation harness (+2) | [`evaluate.py`](../scripts/evaluate.py), [`eval/cases.json`](../eval/cases.json), [`evaluation.py`](../src/evaluation.py) | `python scripts/evaluate.py`; `python -m pytest -q` | scenario-matrix pass/fail gate + metrics; results store no query text |

## Reproducible examples

```bash
# 1 — ordinary recommendation. The default catalog is FMA (real artists); the
#     fictional control gives deterministic, stable output:
python -m src.main --catalog fictional "some jazz please"
#   → After Midnight Set — East Ferry Trio [jazz · romantic] …

# 2 — privacy guard: a sensitive query is auto-forced local; the raw email must not
#     appear in output, receipts, or JSONL events
python -m src.main --trace "my email is alice@example.com, find me melancholy piano"
#   → "my email is [redacted], find me melancholy piano" · guard_category=sensitive · network_used=False

# 3 — unsupported FMA hard capability (in the UI, FMA catalog): "clean instrumental music"
#   → Cadence clarifies rather than guessing a clean/instrumental boolean FMA cannot prove.
#     ("more instrumental" remains a soft preference where a trustworthy value exists.)
```

## Verify everything

```bash
python -m pytest --collect-only -q   # the test count is generated, never hand-typed here
python -m pytest -q                   # full offline suite
python scripts/evaluate.py            # gate PASS; fictional control 0.863
```

## Pending before a launch claim (not implied by unit tests)

The rubric is demonstrable from the integrated code and fixture-backed tests, but the
larger product plan still needs *generated* real-build evidence: actual FMA
accepted/quarantined counts and coverage; per-target model metrics and release
decisions; a published, checksummed Full release with a deterministic-rebuild and
corrupt-asset fallback check; measured runtime performance on a named machine; 300
primary + 60 audit human mood labels with a documented promotion decision; and an
accessibility/deployment review. These are tracked in the project handbook.
