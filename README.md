# Cadence — an evidence-first AI music companion

Cadence turns a natural-language listening request into a small, explainable set
of music recommendations. It is deliberately local-first: ranking, validation,
fallbacks, and the fictional regression catalog work without an API key. Optional
AI providers can add semantic retrieval, bounded voice framing, or post-ranking
research, but they never become the authority for track eligibility.

The original classroom project was a deterministic content-based recommender over
20 fictional songs. It accepted one structured taste profile, compared genre,
mood, and numeric features, then printed a top-five list. It had no free-text
interface, retrieval, agent, guardrails, provenance, evaluation harness, or UI.
Cadence preserves that understandable scorer as a regression control and builds a
complete applied-AI system around it.

## What Phase 5 adds

Phase 5 moves Cadence toward an honest real-data product while preserving the
fictional system unchanged as a control:

- catalog-qualified identities (`TrackRef`) so `fictional:1` and `fma:1` cannot
  collide in receipts, feedback, UI keys, or research;
- optional, provenance-bearing catalog fields where unknown is different from
  zero or false;
- a deterministic FMA ETL with checksum and ZIP defenses, explicit multi-row CSV
  parsing, scoped metadata, quarantine output, normalized SQLite, FTS5, and a
  checksummed manifest;
- read-only FMA search with independent text and structured candidate legs;
- an offline specialized-model pipeline that learns Echo Nest-compatible audio
  character from FMA's Librosa statistics, evaluates against baselines, reports
  prediction intervals, and abstains on weak or out-of-domain rows;
- a visibly experimental valence/arousal mood profile: `upbeat`, `calm`,
  `intense`, and `somber`;
- a local, prediction-hidden human annotation harness. It measures readiness but
  cannot silently promote an experimental model;
- optional per-track research through exact MusicBrainz identity resolution and
  citation-validated Gemini grounded search. Research happens after ranking,
  leaves the list unchanged, and falls back to local FMA evidence;
- a catalog artifact policy: verified local Full → verified release cache →
  verified committed 300-track Lite.

This is not a Spotify-sized streaming service. Cadence does not supply audio,
accounts, collaborative filtering, persistent profiling, or commercial music
rights. Its product niche is transparent independent-music discovery.

## Release status — read this before making claims

The Phase 5 code, contracts, fixture-backed tests, and build tools live in this
repository. A real full-catalog release is a separate evidence gate because it
requires the pinned FMA source files, an offline training/build run, measured
model reports, a generated SQLite artifact, checksums, and publication.

| Item | Repository status | What is not implied |
|---|---|---|
| Fictional 200-track control | Implemented; historical evaluation gate is `0.863` average genre satisfaction | It is not real music data |
| Unknown-safe contracts and behavior | Implemented and test-covered | `None` is never treated as clean, instrumental, zero, or a match |
| FMA ETL, SQLite store, FTS5, manifests, resolver | Implemented and fixture-tested | Fixture success is not proof that a full public artifact has been built |
| Specialized model and abstention framework | Implemented and unit-tested | No target is called released until its real locked-test report passes every gate |
| Experimental mood math | Implemented and unit-tested | It is not an FMA-authored mood or a human-calibrated truth |
| Human-labeling harness | Implemented for local use | Calibration remains pending until the sample/agreement gates are met |
| Post-ranking research agent | Implemented and fixture-tested | It is optional enrichment, not catalog truth and not a ranking signal |
| Full release asset, real model metrics, production performance | Build/release evidence must be generated and recorded | Do not copy example placeholders into a launch claim |

Generated manifests and model reports are the source of truth for counts,
coverage, hashes, MAE, R², interval coverage, and retained coverage. Documentation
must not hand-type those results.

## Quick start

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Run the listener UI:

```bash
streamlit run streamlit_app.py
```

Run the CLI companion:

```bash
python -m src.main "some jazz please"
python -m src.main --trace "upbeat party music"
python -m src.main "my email is alice@example.com, find me melancholy piano"
```

The third request demonstrates a guardrail: the email is redacted, provider use
is blocked for the guarded turn, and the useful local pipeline still runs.

Optional Gemini features read `GEMINI_API_KEY`. Keep the UI's Local-only policy
enabled when a prompt should not leave the Cadence server. A hosted Streamlit app
still receives browser input; “local-only” means Cadence does not forward it to an
AI provider.

## Reproducible behavior examples

These are captured behaviors of the local control path; titles belong to the
clearly labeled fictional catalog.

```text
$ python -m src.main "some jazz please"
Here are a few picks for that:
1. After Midnight Set — East Ferry Trio [jazz · romantic]
2. Coffee Shop Stories — Slow Stereo [jazz · relaxed]
...
[degraded] · mode: degraded · voice: template
```

```text
$ python -m src.main "my email is alice@example.com, find me melancholy piano"
You asked: "my email is [redacted], find me melancholy piano"
... local grounded recommendations ...
mode: local/degraded · voice: template
```

```text
$ python scripts/evaluate.py
... evaluation report ...
gate: PASS
average genre satisfaction: 0.863
```

The exact test count is intentionally not copied here. Test collection is the
authority:

```bash
python -m pytest --collect-only -q
python -m pytest -q
```

## The foundational idea: evidence has types

For a beginner, this is the most important Phase 5 lesson. These values are not
interchangeable:

| Value | Meaning | Example | Ranking rule |
|---|---|---|---|
| `None` | We do not know | FMA cannot verify clean lyrics | No reward, penalty, or hard-filter pass |
| `0.0` | Known numeric zero | A computed feature is exactly zero | Use it as real evidence |
| `False` | Known boolean false | A curated fictional track is not instrumental | It fails an instrumental-only filter |
| Computed | An algorithm produced the value | Echo Nest audio features | Use it with `echonest_computed` lineage |
| Estimated | Cadence's released model predicted it | Energy inferred from Librosa statistics | Weight it by calibrated confidence |
| Authored/supplied | A person or source supplied it | FMA album information | Keep its source and scope visible |

Why this matters: if `explicit=None` passed “clean only,” Cadence would be saying
“clean” without evidence. If a missing energy became zero, a quiet request would
reward a track for data it never had. The contracts, scorer, retriever, evaluator,
CLI, and cards all enforce the same unknown-safe semantics.

Try the classification check before reading the answer:

1. FMA has no lyric-content field for a track. Is it clean, explicit, or unknown?
2. Echo Nest reports an instrumentalness of `0.0`. Is that unknown or known zero?
3. Cadence predicts energy with a wide interval and abstains. What enters ranking?

Answer: **unknown; known zero; nothing for that field**.

## Architecture in plain language

The authoritative diagram is [`diagrams/architecture.mmd`](diagrams/architecture.mmd).
It contains three separated systems:

1. **Offline evidence build.** Untrusted source archives are verified and parsed;
   models are trained and evaluated; accepted values and unknowns are baked into
   SQLite with lineage. Training libraries do not ship in the listener runtime.
2. **Serving.** A guarded request becomes typed intent. The selected catalog
   contributes candidates, deterministic code ranks them, an evaluator checks the
   output, and Cadence expresses only bounded claims.
3. **Post-ranking research.** A person explicitly chooses one result. Only its
   title and artist go to identity resolution/research. Any brief appears beside
   the existing recommendation; it cannot reorder or admit a track.

Human labeling and the DEAM benchmark sit outside serving. The first may support a
future promotion decision after its gates pass. DEAM is CC BY-NC and is isolated
to a non-commercial report that has no production effect.

## RAG: exactly what is retrieved, augmented, and generated

RAG means **retrieval-augmented generation**, but those words are boundaries, not
marketing labels.

### R — retrieval

For the fictional control, retrieval uses the existing catalog documents and
versioned context guides through local TF-IDF plus optional cached/live semantic
embeddings. Guides may expand a query but are never recommendable tracks.

For FMA, the runtime queries SQLite in two independent ways:

- FTS5 text retrieval searches title, artist, genres, separately scoped tags,
  track information, album information, artist biography, and deterministic
  feature terms. Fields have decreasing weight by scope.
- structured retrieval searches genres and trustworthy numeric features even
  when a track has weak text.

Each leg returns up to 200 candidates. Their union is combined using weighted
reciprocal-rank fusion. This independence is essential: text retrieval alone
could exclude a genuinely high-energy track merely because its biography never
uses the word “energy.”

### A — augmentation

The chosen candidates carry controlled evidence into scoring and explanation:
catalog identity, source scope, feature value, lineage, confidence, and retrieval
reason. Missing values are omitted. Raw FMA biographies or arbitrary web prose are
not handed to the Cadence voice model.

### G — generation

The recommendation list is generated by deterministic ranking code, not by a
chat model. Cadence's optional language model can only choose approved, fact-free
framing; invalid output falls back to a local template. Track cards are rendered
from validated evidence.

Post-ranking web research is deliberately separate. It enriches one card only
after the recommendation has been fixed.

## FMA catalog build

Install build dependencies in a separate environment. The normal app does not
need pandas or scikit-learn and never loads a serialized estimator.

```bash
python3 -m venv .venv-ml
source .venv-ml/bin/activate
python -m pip install -r requirements-ml.txt
```

The build tools require an official FMA metadata archive whose digest is pinned
in `scripts/build_fma_catalog.py`. They reject checksum mismatches, unsafe ZIP
members, schema drift, invalid identities, non-finite numbers, and unsafe URLs.

Build commands and source-to-model-matrix preparation are documented by their
CLI help so the checked-in interface—not a stale README copy—remains canonical:

```bash
python scripts/build_fma_catalog.py --help
python scripts/train_fma_models.py --help
```

The intended reproducible order is:

```text
verify/extract official FMA metadata
→ prepare the 518-feature model matrix
→ train and write model report + abstaining prediction JSONL
→ build FMA Full and deterministic FMA Lite SQLite files
→ validate database/manifests/checksums
→ measure determinism and runtime performance
→ publish only the verified Full artifact and its checksum
```

Until a real build report is committed, do not claim “106,574 imported,” “model
released,” or “sub-second p95.” The ETL accepts every row with a valid positive
ID, title, and artist; the generated manifest records the actual accepted and
quarantined counts.

### Why SQLite, not a vector database?

FMA's text and structured fields fit SQLite well. FTS5 provides fast ranked text
search, normal tables provide precise numeric/genre search, read-only mode reduces
mutation risk, and the artifact is one portable file. This keeps the product free,
inspectable, and easier to reproduce. A vector database would add operational
cost without solving a demonstrated Phase 5 need.

## Specialized feature prediction

The six separate targets are energy, valence, acousticness, danceability, tempo,
and instrumentalness. Echo Nest values are machine-computed reference estimates,
not human ground truth.

For each target:

1. Artists—not tracks—are deterministically split 70% train, 15% calibration,
   and 15% locked test. This prevents the sound of the same artist leaking into
   both training and evaluation.
2. A median dummy and Ridge establish simple baselines.
3. Histogram gradient boosting learns the point value; 10th/90th quantile heads
   estimate an interval.
4. Calibration chooses confidence/OOD/width behavior without touching the locked
   test set.
5. Release requires at least 5% MAE improvement over both baselines, 75–90%
   interval coverage, and at least 30% retained coverage on otherwise-missing
   tracks.
6. A retained row must also meet MAE-oriented width and OOD gates. Failure yields
   `None`, not a guessed default.

A model can therefore be useful without being universal. Coverage answers “how
often can it speak?” while MAE answers “how wrong is it when it speaks?” Cadence
needs both.

## Experimental mood

Mood is derived from trustworthy energy and valence values:

```text
high_arousal = sigmoid((energy - 0.5) / 0.15)
positive     = sigmoid((valence - 0.5) / 0.15)

upbeat  = high_arousal       × positive
calm    = (1 - high_arousal) × positive
intense = high_arousal       × (1 - positive)
somber  = (1 - high_arousal) × (1 - positive)
```

If the leading quadrant is less than `0.10` ahead of second place, Cadence says
balanced/uncertain and does not index a mood label. The result lives in
`mood_profile`; it never overwrites FMA's authored `mood` because FMA supplies no
such fact.

The local annotation workflow hides predictions while a human supplies valence,
arousal, quadrant, and confidence:

```bash
python scripts/build_mood_annotation_sample.py --help
CADENCE_LOCAL_ANNOTATION=1 python scripts/annotate_mood.py --help
```

The status remains experimental until 300 primary labels, 60 independent audit
pairs, and the future agreement/promotion review are genuinely complete.

## Optional research action

Research is intentionally user-triggered and downstream:

```text
local recommendation complete
→ research requested
→ MusicBrainz identity resolved or abstained
→ grounded search attempted only when configured
→ citations validated
→ brief published or deterministic local fallback
```

Only the selected title and artist leave the app. The listener prompt, history,
and preferences do not. Multiple exact MusicBrainz recording IDs are treated as
ambiguous. Published output is limited to three short claims with structured
citation coverage. Private/local URLs, prompt-like page instructions, missing
citations, oversized output, quota errors, and timeouts all fail closed.

## Reliability and testing

Run the complete offline suite and evaluation harness:

```bash
python -m pytest -q
python scripts/evaluate.py
```

Phase 5 test areas include:

- unknown vs false/zero behavior in retrieval, scoring, MMR, evidence, UI, and
  CLI;
- safe source parsing, deterministic builds, FTS5, read-only SQLite, manifest
  matching, corrupt artifact fallback, and stable Lite selection;
- artist-group splits, baseline/release gates, prediction intervals, OOD/width
  abstention, and mood-boundary behavior;
- prediction-hidden annotation samples and duplicate-rater protections;
- exact, ambiguous, and missing MusicBrainz identity; citation coverage;
  injection/URL/size/timeout/quota/no-key failures; and local fallback;
- preservation of the fictional regression control and its `0.863` evaluation
  baseline.

Before publishing a full artifact, record—not assume—byte determinism, database
open time, process memory, compressed size, first-download time, and warm p95 over
the fixed 50-query benchmark on a named machine.

## Repository map

| Path | Responsibility |
|---|---|
| `src/contracts.py` | Immutable identities, lineage, tracks, descriptors, research, requests, responses |
| `src/etl/` | FMA integrity checks, parsing, normalization, manifests, catalog orchestration |
| `src/fma_store.py` | Normalized SQLite schema, FTS5, read-only text/structured access |
| `src/catalog_artifacts.py` | Local Full → release cache → Lite resolution and verification |
| `src/modeling.py` | Offline target-specific training, metrics, release and abstention gates |
| `src/mood.py` | Deterministic four-quadrant mood profile |
| `src/annotation.py` | Prediction-hidden sample, local labels, readiness statistics |
| `src/research.py` | Exact identity, grounded search, citation guard, fallback trace |
| `src/companion.py` | Shared guarded recommendation workflow |
| `ui/` | Streamlit presentation and session behavior; no duplicate recommender |
| `scripts/` | Builds, model training, annotation, DEAM isolation, evaluation, demos |
| `tests/` | Offline unit, integration, reliability, regression, and UI tests |
| `diagrams/architecture.mmd` | Required Mermaid source matching the Phase 5 design |

## License and data boundaries

- Project code: MIT; see [`LICENSE`](LICENSE).
- Fictional catalog and guides: project-authored/synthetic, clearly labeled.
- FMA metadata: CC BY 4.0; individual track audio licenses are separate and must
  remain attached to track records. Cadence distributes metadata, not audio.
- DEAM: CC BY-NC; local non-commercial benchmark only, with outputs restricted to
  `eval/noncommercial/` and zero production effect.
- Gemini and MusicBrainz: optional network services subject to their current
  terms and usage policies. See [`docs/LICENSING.md`](docs/LICENSING.md).

Primary dataset references: [official FMA repository](https://github.com/mdeff/fma),
[FMA paper](https://arxiv.org/abs/1612.01840), and
[DEAM manual](https://cvml.unige.ch/databases/DEAM/manual.pdf).

## Honest limitations and next evidence gates

- FMA is broad independent-music metadata, not a licensed streaming catalog.
- Metadata, tags, biographies, descriptions, numeric features, and licenses have
  different coverage. Missing values remain missing.
- Echo Nest and Librosa values are machine-computed. Model estimates add another
  uncertainty layer.
- Mood quadrants are a transparent experiment, not an objective theory of music.
- No FMA clean/explicit or verified instrumental boolean exists; Cadence must
  clarify those hard requests or use a catalog that can prove them.
- Fixture tests validate code behavior, not real-source quality or model value.
- A full build, measured model report, signed/checksummed release, browser and
  accessibility review, and deployment smoke test remain launch evidence gates.

## AI collaboration reflection

AI assistance helped expand the design space, draft code, surface unknown-data
failures, and create adversarial tests. The most useful suggestion was to split
text retrieval from structured retrieval: it prevents sparse prose from deciding
which numeric tracks are even considered. A flawed early direction was to infer
missing mood/description fields or research tracks before ranking; that would make
generated material look like source truth and could change eligibility. The final
design instead carries lineage, abstains, and keeps web research downstream.

The project owner should personalize this reflection before submission. The full
observable action/decision trace is in [`ai_interactions.md`](ai_interactions.md),
and the long-form recovery and teaching guide is in
[`docs/PROJECT_HANDBOOK.md`](docs/PROJECT_HANDBOOK.md).
