# Cadence — Project Handbook

A concise design-rationale companion to the [README](../README.md), the
[model card](../model_card.md), the [data card](CATALOG_DATA_CARD.md), and the
architecture diagram ([`../diagrams/architecture.mmd`](../diagrams/architecture.mmd)).
It records *why* the boundaries exist and how to reason about the system without
memorizing code.

## From a baseline to a system

The project began as **Music Recommender Simulation**: 20 fictional CSV rows, a
structured taste profile in, readable weighted points for genre/mood/energy/
acousticness/valence/danceability/tempo, and a printed top-five — no free text,
retrieval, LLM, agent, contracts, guardrails, provenance, evaluation, or UI. Its
value is that every decision is understandable. The extension strategy was to
**preserve that core as an immutable control** and add layers around it, so a new
"smart" feature can never quietly break the known baseline.

| Stage | Main addition | Learning goal |
|---|---|---|
| Foundation | Pydantic service boundary, immutable inputs | type hints express intent; runtime validation enforces it |
| Catalog | 200 fictional tracks with deterministic metadata | RAG quality starts with grounded data quality |
| Retrieval | TF-IDF + context-guide query expansion | retrieval finds evidence; it doesn't generate the answer |
| Semantic / agent | embeddings, guard, intent, bounded actions, voice | providers belong behind interfaces and fallbacks |
| Reliability / product | structured fusion, MMR, evaluator, harness, receipts, Streamlit | a feature is finished only when users can see and test it |
| Real data | FMA catalog v2, specialized prediction, experimental mood, annotation, research | real data requires unknowns, lineage, abstention, licenses, human gates |
| Personalization | session-only feedback re-rank; a warmer best-effort front door | adaptation must be bounded, reversible, and isolated |

## Core design principles

1. **Identity is contextual.** An integer id is unique only within its catalog, so
   receipts, research, feedback, and UI state use `TrackRef(catalog_id, track_id,
   external_id)` — otherwise a catalog switch could attach a rating to the wrong track.
2. **Requirements ≠ preferences.** A hard constraint defines eligibility ("clean
   only" removes ineligible tracks); a soft preference reorders ("more instrumental").
   Unknown FMA booleans cannot pass a hard filter, so Cadence clarifies.
3. **Unknown ≠ zero ≠ false ≠ estimated.** `None` = no evidence; `0.0`/`False` = known
   values; an estimate carries method, confidence, interval, and release status.
   Turning `None` into `0`/`False` fabricates evidence, so the whole pipeline keeps
   them distinct (missing numerics leave the score untouched; `None` never satisfies a
   boolean filter; two missing moods are not an MMR match; documents never index "None").
4. **Provenance travels with a value.** `FieldLineage` records origin (FMA metadata,
   Echo Nest-computed, model-estimated, deterministic-derived, …) so a polished UI
   cannot flatten all numbers into "facts."
5. **Abstention is a feature.** A model that ships passes a *global* gate (better than
   simple baselines, calibrated) and each prediction passes a *row* gate (in
   distribution, narrow enough); failing either yields `None`. "All tracks imported"
   never means "all numeric holes filled."

## RAG boundaries

**Retrieval** builds a bounded candidate pool (fictional: TF-IDF + context guides +
optional embeddings; FMA: independent FTS5 and structured SQL legs, ≤200 each, fused
by weighted reciprocal rank). Independence matters: a track with computed energy 0.94
but only a title in text would never be retrieved for "intense workout" by text
alone. **Augmentation** carries catalog-controlled evidence (identity, scope, value,
lineage, confidence, reason) into ranking and explanation; missing fields contribute
nothing, and raw prose never enters the voice prompt. **Generation** is deterministic
ranking; the optional voice model may only *select* approved fact-free framing, after
the evaluator runs. Research is downstream enrichment, never a hidden retrieval leg.

## The reliability chain

Reliability is a chain of independent checks, not one final function:

| Layer | Failure prevented |
|---|---|
| strict contracts | malformed/ambiguous values crossing boundaries |
| source checksum + safe-ZIP | a corrupt or hostile archive being parsed |
| explicit ETL + quarantine | fabricating sparse data or hiding a bad identity |
| lineage | estimates presented as source facts |
| model baselines / release gate | complexity shipping without measured value |
| row abstention | forcing every input through a model |
| catalog capability check | unknown booleans satisfying hard constraints |
| independent retrieval legs | sparse text blocking structured evidence |
| grounding evaluator | invalid ids/constraints/evidence reaching the user |
| citation validator | unsupported web claims being published |
| session-only, isolated feedback | one listener's taste leaking into another's |
| local fallback | a provider failure removing core utility |
| regression / evaluation suite | a feature looking good while breaking known behavior |
| human review | cultural/legal decisions treated as purely technical |

Privacy-safe receipts log ids, modes, latency, guard categories, and tool outcomes —
never raw prompt text or hidden reasoning.

## Evaluation

The fictional control must hold its accepted **0.863 average genre satisfaction** and
**100% hard-constraint adherence**; a new FMA metric never redefines it. The harness
(`scripts/evaluate.py`) runs labeled cases across a scenario matrix and reports a
pass/fail gate with metrics. Test and build counts are *generated* (collection, model
reports, manifests) rather than hand-typed in documentation.

## Tooling (free / low-cost, proportionate)

| Need | Choice | Why |
|---|---|---|
| runtime store / search | SQLite + FTS5 | built-in, portable, transactional, no server |
| UI | Streamlit + AppTest | Python-first, testable, free hosting |
| contracts | Pydantic | strict immutable runtime validation |
| offline ETL / models | pandas + scikit-learn (pinned) | explicit CSV handling; mature baselines/HGB/quantiles |
| full artifact | GitHub Release asset | checksummed distribution beyond repo blob limits |
| identity / research | MusicBrainz · optional Gemini grounding | open identity; structured grounding metadata |

No LangChain or vector database is used — add one only on a measured need.

## Repository map

| Path | Job |
|---|---|
| `src/contracts.py` | immutable public contracts, identity, provenance |
| `src/recommender.py`, `src/service.py` | original deterministic scoring control |
| `src/retrieval.py`, `src/structured.py`, `src/fusion.py`, `src/ranking.py` | retrieval, fusion, MMR, unknown-safe logic |
| `src/guard.py`, `src/intent.py`, `src/companion.py` | guarded bounded-agent workflow |
| `src/session_preference.py` | session-only feedback re-rank |
| `src/evaluator.py`, `src/evaluation.py` | result checks + report-card harness |
| `src/etl/`, `src/fma_store.py`, `src/catalog_artifacts.py` | FMA ingest, SQLite store, artifact resolution |
| `src/modeling.py`, `src/mood.py`, `src/annotation.py` | offline models, mood profile, human-label harness |
| `src/research.py` | post-ranking identity, grounded/catalog research |
| `ui/`, `streamlit_app.py` | interface and session state (no duplicated ranking) |
| `scripts/`, `tests/` | build/train/evaluate/demo; offline proof of behavior |

## Design decision log

| Decision | Reason |
|---|---|
| Preserve the fictional catalog as an immutable control | a real-data change needs a stable behavioral comparator |
| Dual catalogs (FMA + fictional), neither privileged | real data adds credibility; the control preserves reproducibility |
| Import all valid identities; never require full metadata | sparse source truth is more honest than a fabricated fuller catalog |
| Keep track/album/artist text separate | scope is evidence; flattening changes meaning |
| Independent text + structured retrieval | either evidence type can rescue a candidate the other misses |
| Rank fusion, not raw-score addition | BM25 and structured scores share no calibrated scale |
| Split model data by artist | avoid leakage across one creator's similar works |
| Require baselines, intervals, OOD, abstention | a prediction needs measured value and a safe refusal path |
| Derived mood in `mood_profile`, marked experimental | never overwrite or imply an authored mood |
| Harness first; no human-calibration claim | a label UI is not the same as credible evidence |
| Session-only, bounded, reversible feedback | adapt within a session without a persistent profile or cross-session effects |
| Best-effort front door with honest framing | serve real tracks for vague requests; put honesty in the framing, not a refusal |
| Research user-triggered and post-ranking; title/artist only | web uncertainty must not control eligibility, and listener context is unnecessary |

## Limits and the expert mindset

Cadence cannot know how a listener will personally feel, verify FMA lyrics, guarantee
every source field is current, or replace a licensed catalog. Its feature model
learns compatibility with Echo Nest computations, not objective truth; its mood model
is a transparent two-axis simplification; and citations support a claim without making
the whole web safe. The mindset is not to eliminate uncertainty but to **name it,
preserve its provenance, measure it when possible, abstain when evidence is weak, keep
a useful fallback, and put humans at the cultural, legal, and promotional decisions.**
