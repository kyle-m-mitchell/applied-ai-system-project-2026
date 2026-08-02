# Cadence Model and AI System Card

Version: **Phase 5 / 2026-08-02**  
Status: **modeling framework implemented and fixture-tested; real FMA target reports pending a pinned-source build**

## 1. What this card covers

Cadence is a hybrid system, not one monolithic model. This card documents every
AI-like component and its authority:

| Component | Technique | May decide | Must not decide |
|---|---|---|---|
| Original scorer | deterministic weighted content score | fictional catalog ordering | invent fields or IDs |
| Fictional retriever | TF-IDF + optional cached/live embeddings | candidate evidence from authored catalog/guides | bypass hard constraints |
| FMA retriever | SQLite FTS5 + structured SQL + rank fusion | candidate pool from stored evidence | fabricate absent text/features |
| Specialized feature models | offline target-specific HGB + uncertainty gates | baked estimates for six missing numeric fields after release/row gates | force a value, overwrite Echo Nest, run in serving |
| Mood profile | deterministic sigmoid quadrant math | experimental scores/label when both axes exist | write an authored FMA mood or claim human truth |
| Cadence voice | deterministic template + optional bounded Gemini selection | fact-free response framing | create recommendation facts or rank tracks |
| Research agent | exact MusicBrainz identity + optional Gemini grounded search | up to three cited post-rank claims | change eligibility/order or persist catalog truth |

## 2. Intended use

Cadence is an educational and portfolio music-discovery companion. It demonstrates
natural-language intent, RAG, specialized modeling, agents/tool calls,
provenance, uncertainty, abstention, guardrails, evaluation, and human review in
one integrated product.

The FMA path is intended for transparent independent-music metadata discovery.
The fictional path is an immutable regression/demo control. Neither supplies
audio playback or a licensed commercial streaming service.

## 3. Original deterministic baseline

The base system compares a structured request with fictional track genre, mood,
energy, valence, danceability, acousticness, and tempo. It is intentionally
readable and local. The fictional catalog later grew from 20 to 200 tracks but
preserved the original records and behavior.

The accepted evaluation control is:

| Metric | Accepted value |
|---|---:|
| Required hard-constraint adherence | 100% |
| Average genre satisfaction | 0.863 |

Phase 5 must preserve this control exactly. FMA performance is reported separately
and can never redefine the baseline after the fact.

## 4. Specialized FMA feature models

### 4.1 Goal

FMA has 518 Librosa-derived statistics for nearly all tracks but Echo Nest audio
features for a much smaller overlap. Cadence learns compatibility mappings from
the Librosa statistics to six Echo Nest-computed targets:

- energy
- valence
- acousticness
- danceability
- tempo BPM
- instrumentalness

The goal is not to discover objective truth. It is to extend the availability of
a clearly labeled, Echo Nest-compatible audio-character estimate when evidence
supports doing so.

### 4.2 Data and leakage control

The offline prepared table contains a positive `track_id`, normalized artist,
Librosa input columns, and target columns with Echo Nest values or missing values.
Target columns and identity columns are forbidden as model inputs.

Artists are deterministically split:

- 70% training artists;
- 15% calibration artists;
- 15% locked-test artists.

All tracks by one normalized artist stay in one split. This reduces leakage from
creator-specific sound. It does not guarantee that related artists, releases, or
recording conditions are independent.

### 4.3 Models and baselines

Each target is independent:

- `DummyRegressor(strategy="median")` is the no-signal baseline;
- Ridge is the simple linear baseline;
- `HistGradientBoostingRegressor` predicts the point value;
- two histogram-gradient-boosting quantile models predict the 10th and 90th
  percentiles.

Input median/IQR statistics support OOD distance. Missing input fractions,
non-finite values, domain limits, prediction ranges, and interval width contribute
to row-level abstention.

### 4.4 Global release gates

A target is globally released only if the locked test shows:

1. MAE at least 5% better than the median dummy;
2. MAE at least 5% better than Ridge;
3. 10th–90th interval coverage between 75% and 90%;
4. at least 30% retained coverage for otherwise-missing rows; and
5. a calibrated interval-width threshold.

Calibration chooses the widest accepted interval compatible with retained MAE no
worse than `0.15` for unit targets or `15 BPM` for tempo. A failed target emits
null predictions with `released=false`.

### 4.5 Row-level abstention

Even a globally released target abstains when one row has:

- more than 20% missing model inputs;
- non-finite point/interval values;
- point or interval outside the allowed feature range;
- OOD score above the calibration threshold; or
- interval width above the calibrated threshold.

A retained estimate stores point, confidence, lower/upper interval, and model
version. Confidence is based on interval width and weights that feature's
structured ranking contribution. An abstention remains `None` and changes neither
score numerator nor denominator.

### 4.6 Real model result table

This table must be filled from the generated JSON report—not by hand. At the time
of this card, the real pinned FMA matrix has not been trained in this repository,
so no target is claimed as released.

| Target | Dummy MAE | Ridge MAE | HGB MAE | R² | Interval coverage | Retained coverage | Released |
|---|---:|---:|---:|---:|---:|---:|---|
| energy | pending | pending | pending | pending | pending | pending | **not claimed** |
| valence | pending | pending | pending | pending | pending | pending | **not claimed** |
| acousticness | pending | pending | pending | pending | pending | pending | **not claimed** |
| danceability | pending | pending | pending | pending | pending | pending | **not claimed** |
| tempo BPM | pending | pending | pending | pending | pending | pending | **not claimed** |
| instrumentalness | pending | pending | pending | pending | pending | pending | **not claimed** |

The synthetic tests prove deterministic artist grouping, baseline comparison,
gate shape, missing-target-only export, and strict deterministic JSON. They do not
measure real FMA quality.

### 4.7 Baseline vs specialized behavior

The measurable comparison is intentionally three-way:

| Case | Baseline catalog behavior | Specialized behavior | Ranking effect |
|---|---|---|---|
| Echo Nest target exists | use `echonest_computed` value | preserve it; never overwrite | full trustworthy feature contribution |
| Target missing; model and row pass | feature unavailable | bake `model_estimated` value + confidence + interval | contribution scaled by calibrated confidence |
| Target missing; model or row fails | feature unavailable | abstain (`None`, `released=false`) | no reward and no penalty |

A real report must add the quantitative before/after: target-specific MAE versus
both baselines and percentage of otherwise-missing tracks retained. That generated
comparison is the release evidence for the specialized-behavior stretch feature.

## 5. Experimental mood profile

FMA does not supply an authored track-level mood field. Cadence derives a separate
profile only when energy and valence exist:

```text
high_arousal = sigmoid((energy - 0.5) / 0.15)
positive     = sigmoid((valence - 0.5) / 0.15)

upbeat  = high_arousal       × positive
calm    = (1 - high_arousal) × positive
intense = high_arousal       × (1 - positive)
somber  = (1 - high_arousal) × (1 - positive)
```

The scores sum to one. If the highest score leads the second by less than `0.10`,
the label is omitted. Input evidence confidence propagates as the minimum known
axis confidence; a decisive quadrant does not manufacture confidence from
uncertain inputs.

Queries map transparently to underlying axes:

| Query | Energy direction | Valence direction |
|---|---|---|
| upbeat | high | high |
| calm | low | high |
| intense | high | low |
| somber | low | low |

Raw trustworthy axes drive ranking. The quadrant is an explanation/index aid and
is always marked experimental.

## 6. Human calibration status

The local harness creates a deterministic, genre-stratified 300-track sample with
prediction fields removed. A human labels valence, arousal, quadrant, and
confidence. Primary and independent-audit roles are distinct; duplicate
rater/track judgments are rejected.

Current status: **harness implemented; human calibration not claimed**.

The target is 300 primary-labeled tracks and 60 independent audit pairs. A future
promotion decision also needs predeclared agreement thresholds and human review.
Only bounded temperature/tag-weight calibration is eligible. Until then manifests
remain `experimental`.

DEAM is CC BY-NC and runs only through the isolated acknowledged benchmark. Its
data, metrics, weights, and thresholds have zero production effect.

## 7. Retrieval and ranking behavior

### Fictional control

TF-IDF/catalog + versioned context guides, optional cached/live semantic
embeddings, structured percentile fusion, and MMR preserve the accepted behavior.
Context guides expand controlled vocabulary and are evidence, not tracks.

### FMA

FTS5 and structured SQL each generate candidates independently. Weighted
reciprocal-rank fusion combines the union. Structured scoring:

- uses only populated evidence;
- weights estimates by confidence;
- does not let missing features weaken or strengthen a match;
- treats “more instrumental” as soft numeric instrumentalness;
- rejects unsupported clean-only/instrumental-only hard requirements.

MMR cannot count missing mood as a match. The grounding evaluator requires valid
catalog IDs, uniqueness, constraint satisfaction, and evidence actually used.

## 8. Cadence voice specialization

Cadence's personality is warm, observant, tactful, concise, and uncertainty-aware.
The deterministic renderer is the baseline. Optional Gemini does not freely
generate track facts; it can only select exact application-owned framing. Any
non-allowlisted, unsafe, malformed, or failed provider response falls back to the
template.

Baseline vs specialized voice:

```text
Baseline: "Here are a few picks for that:"
Specialized: a short approved Cadence line chosen for the bounded action/tone.
Track list and factual cards: identical in both cases.
```

Thus style may change while recommendation authority remains deterministic.

## 9. Post-ranking research model

Research starts only after a human selects one recommended track. MusicBrainz must
resolve exactly one normalized title+artist recording identity. Gemini grounded
search receives that resolved identity, not listener context.

Publication requires:

- no more than three short claims;
- structured grounding support for every claim;
- citation IDs that resolve to safe public HTTP(S) URLs;
- matching source-domain metadata;
- no prompt-like page instructions;
- bounded response size and provider time.

Ambiguity, unsafe/private URL, missing citations, injection-like text, quota,
timeout, missing key, or invalid response uses a deterministic local summary.
Research never changes rank, fields, mood, booleans, or stored catalog data.

## 10. Evaluation

Automated coverage includes:

- contracts, unknown semantics, scoring denominator, retrieval documents, MMR,
  evidence, and catalog capabilities;
- safe FMA parsing, deterministic Lite selection, SQLite/FTS5, read-only access,
  manifests, corrupt-asset fallback, and checksums;
- artist-group split, baselines, global/row gates, intervals, OOD, deterministic
  exports, and quadrant math;
- prediction-hidden annotation and readiness counts;
- exact/ambiguous/missing identity, prompt injection, unsafe URL, missing citation,
  timeout/quota/no-key, 100% citation coverage, and local fallback;
- fictional regression and application UI/CLI parity.

Real release reports must additionally publish model slices by genre/provenance,
FMA genre/numeric satisfaction split by Echo Nest vs estimate, warm p95, open
time, memory, artifact size, and first-download time.

## 11. Risks and limitations

- Echo Nest overlap may be selection-biased and is not ground truth.
- Librosa summaries cannot encode every musical quality or cultural meaning.
- Artist-group splitting reduces one leakage mode but not all dataset dependence.
- A prediction interval is an empirical model interval, not a guarantee.
- The four-quadrant mood theory is a coarse interpretation of two axes.
- FMA metadata richness differs across artists/genres and can bias text retrieval.
- Missing FMA lyric/instrumental booleans limit hard-filter use.
- Research sources can be incomplete, stale, or wrong even with valid citations.
- The UI supplies no audio, persistent profile, or collaborative signal.
- Fixture-backed code tests do not substitute for a real pinned build or human
  calibration.

## 12. Ethical and privacy boundaries

- No listener prompt/history/preferences enter track research.
- Sensitive guarded requests cannot call optional AI providers.
- Logs contain sanitized categories, IDs, scores, modes, and tool outcomes—not raw
  prompt text or hidden reasoning.
- FMA source scopes, license fields, and attribution are preserved.
- Model estimates are visible as estimates and may abstain.
- Human labels are pseudonymous/local and hidden from public deployment.
- DEAM remains non-commercial and isolated.

## 13. Reproducibility

Runtime is local-first. The model/ETL environment is pinned separately in
`requirements-ml.txt`. Serving reads deterministic SQLite and JSONL artifacts and
never deserializes training estimators. Source, artifact, compressed distribution,
schema, ETL, method, and model versions are recorded. Model and manifest JSON are
canonicalized; predictions are sorted by track ID and feature.

Reproducibility claims become valid only after two real builds produce matching
checksums and the reports are reviewed.

## 14. AI-collaboration reflection

The strongest AI-assisted design suggestion was to keep FMA text candidates and
structured candidates independent before fusion. That fixes a real recall
boundary: sparse prose should not prevent a numeric match from entering the pool.

Several attractive suggestions were rejected or corrected:

- generate missing descriptions/moods with an LLM;
- let web research score or reorder results;
- treat Echo Nest values as observed truth;
- let unknown clean/instrumental fields pass hard filters;
- call a built annotation harness “human calibrated”;
- use DEAM-derived thresholds in production despite its non-commercial boundary.

Those options would make the demo look fuller while weakening evidence. The final
design favors lineage, separate scopes, release gates, abstention, post-ranking
research, and explicit pending claims.

The project owner should revise this section in their own voice before submission,
including one concrete debugging moment they personally understand.
