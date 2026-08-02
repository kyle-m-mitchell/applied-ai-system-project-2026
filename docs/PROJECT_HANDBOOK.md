# Cadence Project Handbook

This is the durable, beginner-friendly project memory. If a chat is lost, give
this file, the README, the architecture Mermaid source, the data card, and the
current `git status` to a new collaborator. It records what Cadence is, why the
boundaries exist, what Phase 5 implements, what remains an evidence gate, and how
to reason about the system without memorizing code.

Updated: **2026-08-02 / Phase 5 implementation**

## New-chat recovery prompt

```text
We are building Cadence, an evidence-first applied-AI music companion. Read:
README.md, docs/PROJECT_HANDBOOK.md, docs/CATALOG_DATA_CARD.md,
docs/LICENSING.md, model_card.md, ai_interactions.md, and
diagrams/architecture.mmd. Then inspect git status and run the offline tests.

The original control is a deterministic recommender over a fictional catalog.
Its existing retrieval/ranking behavior and 0.863 evaluation baseline must not
regress. Phase 5 adds unknown-safe catalog contracts; FMA source validation,
ETL, normalized SQLite/FTS5, manifests, Full→release→Lite resolution; offline
Librosa-to-Echo-Nest specialized models with baselines, intervals, OOD gates,
and abstention; an experimental four-quadrant mood profile; a local hidden-
prediction annotation harness; an isolated non-commercial DEAM benchmark; and
optional post-ranking MusicBrainz + citation-validated Gemini research.

Never claim that a real full artifact, model target, human calibration, latency,
or release is complete unless generated evidence exists. Never let web research
change rank or catalog truth. Keep unknown distinct from false and zero. Use the
best free/local tools: SQLite FTS5, Streamlit, pandas/scikit-learn offline,
pytest, GitHub Releases, MusicBrainz, and optional Gemini REST. Teach each change,
run focused/full tests, update the Mermaid source, and preserve unrelated work.
```

## 1. The project in one picture

Think of Cadence as a careful librarian with a personality:

- the **catalog** is the library inventory;
- the **retrievers** find plausible shelves;
- the **ranker** orders books for this request;
- the **evaluator** checks that every offered book really exists and obeys rules;
- the **voice** explains the checked result without inventing facts;
- the **research action** looks up one chosen book afterward, with citations;
- the **tests and humans** check whether the entire process deserves trust.

The language model is not the librarian's memory and not the recommendation
authority. It is an optional specialist at tightly controlled seams.

The authoritative data-flow drawing is
[`../diagrams/architecture.mmd`](../diagrams/architecture.mmd).

## 2. Where the project started

The base project was **Music Recommender Simulation**. It:

- loaded 20 fictional CSV rows;
- accepted a structured taste profile;
- assigned readable weighted points for genre, mood, energy, acousticness,
  valence, danceability, and tempo;
- sorted by score and printed the top five.

That small system is valuable because every decision is understandable. It had no
free-text input, retrieval, LLM, agent, validation contracts, guardrails,
provenance, evaluation report, logging, or UI.

The extension strategy was to preserve that core and add layers around it. This
makes regression measurable: a new “smart” feature is not allowed to quietly
break the known baseline.

## 3. Phase history

| Phase | Main addition | Learning goal |
|---|---|---|
| Foundation | Pydantic service boundary and immutable inputs | Type hints express intent; runtime validation enforces it |
| Catalog expansion | 200 fictional tracks with deterministic metadata | RAG quality starts with grounded data quality |
| Retrieval | TF-IDF catalog retrieval + context-guide query expansion | Retrieval is finding evidence, not generating an answer |
| Semantic/agent | cached/live embeddings, guard, intent, bounded actions, voice | AI providers belong behind interfaces and fallbacks |
| Reliability/product | structured fusion, MMR, evaluator, evaluation harness, receipts, Streamlit | A feature is finished only when users can see and test it |
| Phase 5 | FMA catalog v2, specialized prediction, experimental mood, annotation, post-rank research | Real data requires unknowns, lineage, abstention, licenses, and human gates |

## 4. The five ideas you must master

### 4.1 Identity is contextual

An integer ID is unique only inside its catalog. Fictional track `1` and FMA
track `1` are unrelated. `TrackRef` combines:

```text
catalog_id + local track_id + optional external_id
```

Receipts, research, feedback, and UI state use that namespace. Otherwise a catalog
switch could attach a rating or research brief to the wrong track.

### 4.2 Requirements and preferences are different

A **hard constraint** defines eligibility: “clean only” means an ineligible track
must never enter the result. A **soft preference** changes ordering: “more
instrumental” can reward trustworthy instrumentalness without claiming a boolean.

Unknown FMA lyric status cannot pass clean-only. Unknown FMA instrumental status
cannot pass instrumental-only. Cadence should clarify or offer a catalog switch.

### 4.3 Unknown, zero, false, and estimated are different

```text
None  = no evidence
0.0   = evidence of a numeric zero
False = evidence that a boolean statement is false
estimated = a value plus method, confidence, interval, and release status
```

The wrong shortcut is to turn `None` into `0` or `False`. That fabricates evidence.
Phase 5 fixes this through the whole pipeline:

- missing numeric fields are absent from numerator and denominator;
- `None` cannot satisfy a boolean filter;
- two missing moods do not look “the same” to MMR;
- retrieval never indexes the literal word `None`;
- cards conditionally render known fields;
- evidence reports only features actually used.

### 4.4 Provenance travels with a value

`FieldLineage` records where a value came from. Important origins include FMA
metadata, Echo Nest-computed, model-estimated, and deterministic-derived. This
prevents a polished UI from flattening all numbers into “facts.”

Echo Nest values are themselves machine-computed estimates. They serve as
training targets because they cover an overlap, not because they are human truth.

### 4.5 Abstention is a feature

An honest system sometimes says “I don't know.” Specialized models can be good on
some tracks and unreliable on others. Cadence uses:

- a global release gate: is this target model better than simple baselines and
  calibrated enough to use at all?
- a row gate: is this particular input in distribution and its interval narrow
  enough?

Failing either gate produces `None`. “All source tracks imported” never means “all
numeric holes filled.”

### Understanding check

Classify each statement before reading the answers:

1. `explicit=None` on a clean-only request.
2. `instrumentalness=0.0` from Echo Nest.
3. a released energy estimate with confidence `0.82`.
4. an album biography shown as a track description.
5. two tracks with no mood profile in MMR.

Answers:

1. unknown, so it cannot pass the hard constraint;
2. known numeric zero, so it is valid evidence;
3. estimated evidence, usable with its lineage/confidence;
4. invalid scope collapse—keep album and track text separate;
5. unknown/unknown is not a similarity match.

## 5. What RAG means in this project

RAG is **retrieval-augmented generation**.

### Retrieval

Retrieval builds a bounded candidate pool. It does not make the final claim.

Fictional retrieval:

- local TF-IDF over authored catalog documents;
- versioned context guides that can contribute controlled query-expansion terms;
- optional cached or live semantic embeddings;
- guides are evidence/bridges, never recommendable tracks.

FMA retrieval:

- FTS5 searches independently across weighted metadata scopes;
- structured SQL searches genre and trustworthy feature goals independently;
- each returns at most 200;
- weighted reciprocal-rank fusion combines the union.

Why separate legs? Suppose a track has computed energy `0.94` but only its title
and artist in text. A text-only pool may never retrieve it for “intense workout.”
Structured search gives the evidence a chance to enter.

### Augmentation

The candidate carries catalog-controlled evidence to ranking and explanation:
identity, matching scope, numeric value, lineage, confidence, and retrieval reason.
Missing fields contribute nothing. Raw biographies and arbitrary web text do not
enter the Cadence voice prompt.

### Generation

Deterministic code generates the ranked list. The optional voice model may select
only approved fact-free framing. The evaluator runs before that framing. Invalid
or failed model output uses a deterministic template.

The optional research agent is downstream enrichment, not a hidden third
retrieval leg. It cannot admit, remove, or reorder recommendations.

### Understanding check

Label each R, A, G, or separate reliability/enrichment:

1. SQLite FTS5 returns 200 IDs.
2. A result receives a confidence-bearing energy value.
3. Cadence selects an approved intro line.
4. The evaluator rejects an unknown ID.
5. A person clicks Research this track after the result exists.

Answers: **R, A, G, reliability, post-ranking enrichment**.

## 6. Phase 5 offline catalog pipeline

### 6.1 Source integrity

`src/etl/integrity.py` computes SHA-256 and validates ZIP members before
extraction. Defenses include path traversal, absolute paths, symbolic links,
member-count/size limits, duplicate names, and compression-ratio risk. The source
archive digest is pinned in the build script.

Lesson: file parsing is a security boundary. A public dataset is not automatically
safe merely because it is famous.

### 6.2 Explicit FMA parsing

FMA CSVs use multiple header rows. `src/etl/fma.py` names expected columns such as
`('track', 'title')` rather than flattening or guessing them. It normalizes
Unicode and plain text, safely parses list cells, validates links, keeps text
scopes separate, and processes track rows in chunks.

Valid identity rows are accepted even with sparse metadata. Bad identity rows are
quarantined with reason codes.

### 6.3 Numeric join and lineage

Echo Nest overlap supplies known computed target values. Released prediction JSONL
may supply otherwise-missing fields. Echo Nest takes precedence. The ETL records
the selected value's origin, method, confidence, and interval.

### 6.4 Mood derivation

`src/mood.py` turns trustworthy energy and valence into four scores. A low leader
margin yields no label. The profile is written separately from authored mood and
marked experimental.

### 6.5 SQLite artifact

`src/fma_store.py` writes normalized tables and a contentless FTS5 index in stable
track-ID order, then vacuums and atomically replaces the destination. Runtime opens
it read-only/immutable/query-only and materializes only requested tracks rather
than 106,000 Pydantic objects at startup.

### 6.6 Manifest and resolver

`src/etl/manifest.py` writes canonical JSON facts plus a checksum sidecar.
`src/catalog_artifacts.py` verifies the artifact and resolves local Full, release
cache/download, then bundled Lite. Any corrupt candidate is skipped with a
sanitized warning; if Lite is also invalid, startup fails closed.

The manifest is checksummed but not cryptographically signed.

### One-row trace

```text
tracks.csv identity and scoped metadata
  + genres.csv names
  + echonest.csv known computed features, when present
  + released prediction JSONL for eligible missing fields, when present
→ normalized FMA row
→ lineage per used feature
→ experimental mood profile, only if energy and valence exist
→ normalized SQLite row + genre/tag rows + weighted FTS columns
→ manifest coverage/count/checksum
→ read-only candidate retrieval
→ result card with conditional fields
```

## 7. Specialized feature modeling

`src/modeling.py` is offline-only. Runtime requirements do not include pandas or
scikit-learn and do not load pickle/joblib estimators.

### 7.1 Inputs and targets

Inputs are 518 Librosa-derived statistics. Six targets are trained separately:

- energy
- valence
- acousticness
- danceability
- tempo BPM
- instrumentalness

A separate target matters because one model may pass while another fails.

### 7.2 Artist-group split

Artists are deterministically assigned 70/15/15 to train, calibration, and locked
test. Random track splitting could put two songs by the same artist on both sides;
the model might memorize artist sound and appear more general than it is.

Train learns parameters. Calibration chooses uncertainty/OOD gates. Locked test
is opened once for an unbiased report. Tuning on test would turn it into another
calibration set.

### 7.3 Baselines and candidate model

- Median dummy: predicts one central value and proves the task has learnable
  signal beyond “always average.”
- Ridge: a simple regularized linear relationship.
- Histogram gradient boosting: captures nonlinear relationships for the point
  estimate.
- Quantile heads: estimate 10th and 90th percentiles.

The complex model must beat both simpler baselines by at least 5% MAE. Complexity
is earned, not assumed.

### 7.4 Intervals, OOD, and retained coverage

A point estimate alone hides uncertainty. The interval says how broad a plausible
range is. Calibration also measures standardized distance from the training
distribution. Wide, OOD, invalid, or unreleased values abstain.

Three metrics answer different questions:

- **MAE:** average absolute error when compared with target values;
- **interval coverage:** fraction of targets inside the predicted interval;
- **retained coverage:** fraction of otherwise-missing rows on which the system
  is willing to speak.

R² is reported for context but is not the release gate by itself.

### Understanding check

Why can a model with excellent retained MAE still be a poor product model?

Answer: it might retain only 1% of rows. That is precise but nearly useless. Why
can a model with 100% retained coverage be unsafe? It may be wrong or overconfident
on unfamiliar rows. Cadence gates both quality and coverage.

## 8. Human labeling and DEAM separation

`src/annotation.py` and the annotation scripts build a genre-stratified sample,
remove prediction fields, collect pseudonymous labels, reject duplicate
rater/track pairs, and report primary/audit counts and agreement statistics.

Humans label without seeing Cadence's guess because visible predictions can anchor
the judgment. The harness exists now; calibration does not become true merely
because a screen exists.

DEAM is a separate academic comparison under CC BY-NC. The benchmark requires an
explicit acknowledgement, never downloads data, restricts output to
`eval/noncommercial/`, and prints `production_effect: none`. This prevents a
convenient benchmark from quietly contaminating a product artifact.

## 9. Serving and ranking

### 9.1 Guard and intent

`InputGuard` handles size, PII/secret redaction, prompt-injection text, and crisis
routing. The deterministic parser turns safe text into typed intent and a bounded
action. Sensitive status remains sticky during refinements.

### 9.2 Capability validation

The selected `CatalogDescriptor` lists supported hard filters, numeric features,
retrieval methods, context-guide support, and research capability. FMA initially
does not support clean-only or instrumental-only hard filters; fictional does.

### 9.3 Candidate generation and fusion

For FMA, independent FTS5 and structured top-200 sets are unioned and fused by
rank. Confidence weights estimated numeric evidence. Missing numeric fields do not
alter numerator or denominator. MMR diversifies only among candidates above the
relevance floor.

### 9.4 Grounding evaluation and voice

The evaluator verifies IDs, uniqueness, requested count, hard constraints, and
evidence. Cadence's optional model can select only application-owned framing. The
track facts come from validated result objects. Provider failure uses local
framing.

### 9.5 UI state

Catalog switching must clear mix history, undo snapshots, ratings, and research
cache. The UI may display FMA Full/Lite source, lineage, confidence/interval,
experimental mood, scoped context, licenses, warnings, and citations.

No UI code reimplements ranking. Streamlit calls the same application services as
CLI/tests/evaluation.

## 10. Post-ranking research agent

`src/research.py` implements a deliberately small workflow:

1. user selects one already-ranked track;
2. only title and artist go to MusicBrainz;
3. exact normalized title + artist matching must produce one recording ID;
4. optional Gemini grounded search receives that resolved identity;
5. output is limited, scanned for instruction-like content, and mapped only from
   structured grounding supports to safe public citations;
6. any ambiguity, missing citation, unsafe/private URL, excessive size, timeout,
   provider error, or quota failure returns local fallback;
7. `ResearchBrief` lives in session memory only.

The trace records actions and outcomes:

```text
local recommendation complete
→ research requested
→ identity resolved / abstained
→ grounded search attempted
→ citations validated
→ brief published / local fallback
```

This is an action trace, not hidden chain-of-thought.

## 11. Reliability system

Reliability is not one final function. It is a chain:

| Layer | Failure prevented |
|---|---|
| strict contracts | malformed or ambiguous values cross boundaries |
| source checksum + safe ZIP | wrong/corrupt/hostile archive is parsed |
| explicit ETL + quarantine | sparse data is fabricated or malformed identity is hidden |
| lineage | estimates are presented as source facts |
| model baselines/release gate | complexity ships without measured value |
| row abstention | every input is forced through a model |
| catalog capability check | unknown booleans satisfy hard constraints |
| independent retrieval legs | sparse text blocks structured evidence |
| grounding evaluator | invalid IDs/constraints/evidence reach the user |
| citation validator | unsupported web claims are published |
| local fallback | optional-provider failure removes core utility |
| regression/evaluation suite | a feature looks good while breaking known behavior |
| human review | culturally/taste-dependent decisions are treated as purely technical |

Privacy-safe receipts log IDs, modes, latency, guard categories, and action/tool
outcomes. They do not log raw prompt text or hidden reasoning.

## 12. Evaluation plan

The fictional control must remain at its accepted `0.863` average genre
satisfaction and 100% required hard-constraint adherence.

Phase 5 evaluations must cover:

- byte/checksum deterministic ETL;
- database/manifest agreement and fallback on corrupt assets;
- no fabricated mood, prose, booleans, URL, or license;
- unknown-safe behavior in retriever/evaluator/UI/CLI;
- model MAE, R², interval coverage, retained coverage, genre/provenance slices;
- FMA genre and numeric satisfaction split by Echo Nest vs estimates;
- research exact/ambiguous/no match, injection, URL, citation, timeout, quota,
  no-key, and local-fallback fixtures;
- 100% citation coverage on published claims;
- catalog-switch state isolation;
- warm p95 under one second over a fixed 50-query Full benchmark;
- measured open time, memory, artifact size, and first-download time.

Never paste a test count into several files. Run collection/tests and let generated
results be the authority.

## 13. Free and low-cost tool choices

| Need | Choice | Why |
|---|---|---|
| Runtime database/search | SQLite + FTS5 | built-in, portable, transactional, no server or fee |
| UI | Streamlit | Python-first, fast iteration, free community deployment option |
| Contracts | Pydantic | strict immutable runtime validation |
| Offline ETL | pandas 3.0.3 | explicit multi-row CSV support and chunking |
| Offline models | scikit-learn 1.9.0 | mature baselines, HGB, quantile loss, metrics |
| Tests | pytest + Streamlit AppTest | offline unit/integration/UI coverage |
| Full artifact | GitHub Release asset | checksummed distribution outside normal repo blob limits |
| Identity | MusicBrainz | open music identity service with clear etiquette |
| Optional grounded research | Gemini Google Search grounding | structured grounding metadata and free/low-cost experimentation |
| Semantic/voice adapters | direct REST + local cache/fallback | small inspectable seam without framework lock-in |

No LangChain or vector database is required. Add one only after a measured need,
not because an AI project is expected to contain it.

## 14. File map

| File/folder | Job |
|---|---|
| `src/contracts.py` | shared immutable public contracts and provenance |
| `src/recommender.py`, `src/service.py` | original deterministic scoring control |
| `src/retrieval.py`, `src/structured.py`, `src/fusion.py`, `src/ranking.py` | fictional retrieval/fusion/MMR and unknown-safe common logic |
| `src/guard.py`, `src/intent.py`, `src/companion.py` | guarded bounded agent workflow |
| `src/evaluator.py`, `src/evaluation.py` | result checks and report-card harness |
| `src/etl/` | source integrity, FMA normalization, manifest |
| `src/fma_store.py` | SQLite build/read/search/materialization |
| `src/catalog_artifacts.py` | verified artifact resolution/download fallback |
| `src/modeling.py` | offline model training and exported predictions/report |
| `src/mood.py` | four-quadrant deterministic mood profile |
| `src/annotation.py` | human sample/label/readiness utilities |
| `src/research.py` | post-ranking identity, grounded research, citation guard |
| `ui/`, `streamlit_app.py` | product interface and state |
| `scripts/` | build, train, annotate, benchmark, evaluate, demonstrate |
| `tests/` | offline proof of behavior |
| `docs/CATALOG_DATA_CARD.md` | data provenance, schema, biases, update policy |
| `model_card.md` | specialized model/voice behavior and release evidence |
| `ai_interactions.md` | AI collaboration and sanitized action/decision traces |

## 15. Commands

Core runtime:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/evaluate.py
python -m src.main "some jazz please"
streamlit run streamlit_app.py
```

Offline FMA environment:

```bash
python3 -m venv .venv-ml
source .venv-ml/bin/activate
python -m pip install -r requirements-ml.txt
python scripts/build_fma_catalog.py --help
python scripts/train_fma_models.py --help
python scripts/build_mood_annotation_sample.py --help
CADENCE_LOCAL_ANNOTATION=1 python scripts/annotate_mood.py --help
python scripts/benchmark_deam.py --help
```

Use each script's current `--help` as the source of truth for build paths. A real
source build must generate and preserve manifests/reports rather than copying
planned counts into documentation.

## 16. Phase 5 completion checklist

### Code and fixture evidence

- [x] immutable namespaced identity and field-lineage contracts
- [x] optional unknown-safe catalog fields
- [x] unknown-safe hard filters, scoring denominator, MMR, documents, and evidence
- [x] safe explicit FMA ETL and deterministic Lite selection
- [x] normalized SQLite/FTS5 and read-only store
- [x] manifest/checksum and Full→release→Lite resolver
- [x] target-specific offline modeling framework and abstaining predictions
- [x] experimental mood profile
- [x] prediction-hidden local annotation harness
- [x] isolated DEAM benchmark
- [x] post-ranking exact-identity/citation-validated research agent
- [x] updated Mermaid source and documentation

### Real build/release evidence

- [ ] acquire and verify the pinned official archive
- [ ] prepare the actual Librosa/Echo Nest matrix
- [ ] run and review all six real target reports
- [ ] build Full and 300-track Lite artifacts
- [ ] rerun build and prove byte/checksum determinism
- [ ] commit Lite plus its manifest/checksum
- [ ] record Full coverage/counts/quarantine and performance
- [ ] publish compressed Full release asset plus verification metadata
- [ ] exercise a clean-machine release download and corrupt-asset fallback
- [ ] complete browser/mobile/accessibility and deployment review
- [ ] complete licensing/attribution/secrets review

### Human evidence

- [ ] collect/review 300 primary labels
- [ ] collect/review 60 independent audit pairs
- [ ] define agreement and promotion thresholds before inspecting promotion result
- [ ] record a human decision; keep experimental status if the gate is not met

## 17. Decision log

| Decision | Reason |
|---|---|
| Preserve fictional catalog as immutable control | A real-data change needs a stable behavioral comparator |
| Select FMA as intended initial product catalog; Lite is an edition fallback | Real independent music improves credibility while Lite preserves reliability |
| Import all valid identities; never require full metadata | Sparse source truth is more honest than a smaller fabricated catalog |
| Keep track/album/artist text separate | Scope is evidence; flattening changes meaning |
| Use independent text and structured retrieval | Either evidence type can rescue a candidate the other misses |
| Use rank fusion, not raw-score addition | BM25 and structured score are not calibrated to one common scale |
| Split model data by artist | Avoid leakage across similar works from one creator |
| Require baselines, intervals, OOD, and abstention | A prediction needs measured value and a safe refusal path |
| Put derived mood in `mood_profile` | Never overwrite or imply authored FMA mood |
| Harness first; no human-calibration claim | Building a label UI is not the same as collecting credible evidence |
| Isolate DEAM | CC BY-NC benchmark must not contaminate product artifacts |
| Make research user-triggered and post-ranking | Web uncertainty must not control eligibility or order |
| Send title/artist only to research | Listener context is unnecessary and privacy-sensitive |
| Use SQLite/FTS5 and direct adapters | Best free simple tool until measurements justify more infrastructure |

## 18. Known limits and next lesson

Cadence still cannot know how a listener will personally feel, verify FMA lyrics,
guarantee every source field is current, or replace a licensed streaming catalog.
Its feature model learns compatibility with Echo Nest computations, not objective
musical truth. Its mood model is a transparent two-axis simplification. Web
citations can support a claim but cannot make the whole web safe or complete.

The expert mindset is not “eliminate uncertainty.” It is:

1. name the uncertainty;
2. preserve its provenance;
3. measure when possible;
4. abstain when evidence is weak;
5. keep a useful fallback;
6. put humans at decisions that are cultural, legal, or promotional.

If you can explain why each Cadence component exists, what evidence it may use,
what it may not change, and how it fails, you understand the project at a
professional systems level.
