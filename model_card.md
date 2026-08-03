# Cadence — Model and AI System Card

**Status:** modeling framework implemented and fixture-tested; real FMA target
reports pending a pinned-source build.

## 1. Components and authority

Cadence is a hybrid system, not one model. Each AI-like component has a bounded role:

| Component | Technique | May decide | Must not decide |
|---|---|---|---|
| Original scorer | deterministic weighted content score | fictional catalog ordering | invent fields or ids |
| Fictional retriever | TF-IDF + optional cached/live embeddings | candidate evidence | bypass hard constraints |
| FMA retriever | SQLite FTS5 + structured SQL + rank fusion | candidate pool from stored evidence | fabricate absent text/features |
| Specialized feature models | offline HGB + uncertainty gates | baked estimates for six missing numeric fields, after gates | overwrite Echo Nest, force a value, run in serving |
| Mood profile | deterministic sigmoid quadrant math | experimental scores/label when both axes exist | write an authored mood or claim human truth |
| Session personalization | bounded, session-only re-rank from feedback | nudge ordering within a browser session | override an explicit request or persist/cross sessions |
| Cadence voice | deterministic template + optional bounded Gemini selection | fact-free response framing | create track facts or rank tracks |
| Research agent | MusicBrainz identity + optional grounded search / catalog note | a post-ranking note | change eligibility, order, or catalog data |

## 2. Intended use

An educational and portfolio music-discovery companion demonstrating
natural-language intent, RAG, specialized modeling, agents, provenance,
uncertainty/abstention, guardrails, evaluation, and human review in one product.
The FMA path is transparent metadata discovery; the fictional path is an immutable
regression/demo control. Neither supplies audio or a licensed streaming service.

## 3. Deterministic baseline (control)

The base system scores a structured request against fictional-track genre, mood,
energy, valence, danceability, acousticness, and tempo. The fictional catalog grew
from 20 to 200 tracks while preserving the original records and behavior. Its
accepted evaluation control — **100% hard-constraint adherence** and **0.863 average
genre satisfaction** — must be preserved exactly; FMA performance is reported
separately and never redefines the baseline.

## 4. Specialized FMA feature models

FMA provides ~518 Librosa statistics for nearly all tracks but Echo Nest audio
features only for a smaller overlap. Cadence learns Echo Nest-compatible estimates
for six targets (energy, valence, acousticness, danceability, tempo, instrumentalness)
— extending the *availability* of a clearly-labeled estimate, not claiming truth.

- **Leakage control:** artists (not tracks) are split 70/15/15 train/calibration/
  locked-test, so one artist's sound cannot appear in both training and evaluation.
- **Models:** median-dummy and Ridge baselines; `HistGradientBoostingRegressor` for
  the point value; 10th/90th quantile heads for an interval.
- **Global release gates (locked test):** MAE ≥5% better than *both* baselines,
  interval coverage 75–90%, ≥30% retained coverage on otherwise-missing rows, and a
  calibrated width threshold (retained MAE ≤0.15 for unit targets / ≤15 BPM). A
  failed target emits nulls with `released=false`.
- **Row-level abstention:** even a released target returns `None` when inputs are
  >20% missing, values are non-finite or out of range, or the OOD/width thresholds
  are exceeded. An abstention changes neither score numerator nor denominator.

Real per-target metrics (MAE vs. baselines, R², interval and retained coverage) are
generated from the model report JSON, not hand-typed. At this card's date no real
pinned matrix has been trained in-repo, so **no target is claimed as released**; the
synthetic tests prove the grouping, gate shape, and deterministic export only.

## 5. Experimental mood profile

Derived only when energy and valence exist:

```text
high_arousal = sigmoid((energy - 0.5) / 0.15);  positive = sigmoid((valence - 0.5) / 0.15)
upbeat = high_arousal·positive       calm = (1-high_arousal)·positive
intense = high_arousal·(1-positive)  somber = (1-high_arousal)·(1-positive)
```

Scores sum to one; if the leading quadrant leads the second by <0.10 the label is
omitted. Confidence propagates the minimum axis confidence (a decisive quadrant does
not manufacture certainty). Queries map transparently to axes (e.g. *calm* → low
energy, high valence). The raw axes drive ranking; the quadrant is an always-labeled
experimental explanation aid. Human calibration (a prediction-hidden 300-track
sample, 60 audit pairs, predeclared agreement thresholds) is **implemented but not
claimed**; manifests remain `experimental`. DEAM (CC BY-NC) is isolated and has zero
production effect.

## 6. Retrieval, ranking, and voice

The fictional control uses TF-IDF + context guides + optional embeddings, structured
percentile fusion, and MMR. FMA generates text (FTS5) and structured (SQL)
candidates independently and fuses them by weighted reciprocal rank; structured
scoring uses only populated evidence, weights estimates by confidence, lets missing
features neither help nor hurt, and rejects unsupported clean-/instrumental-only
hard filters. A grounding evaluator requires valid unique ids, constraint
satisfaction, and evidence actually used. Session-only feedback adds a bounded,
reversible re-rank that never overrides an explicit request. The voice is
deterministic by default; optional Gemini may only *select* an approved framing line
(see [`docs/CADENCE_VOICE.md`](docs/CADENCE_VOICE.md)).

## 7. Post-ranking research

Research runs only after a user selects one recommended track, and only its title
and artist leave the app (never the prompt, history, or preferences). A tiered
fallback: (1) grounded, citation-validated web search when available — with an
*unverified-identity* web search when MusicBrainz finds no exact match, clearly
labeled; (2) otherwise a non-grounded creative note written *only* from the track's
own catalog attributes, labeled "not web-verified"; (3) otherwise a deterministic
local summary. Ambiguous identities abstain, and unsafe/private URLs, prompt-like
page text, missing citations, oversized output, quota/timeout, and a missing key all
fail closed. Research never changes rank, fields, or stored catalog data.

## 8. Evaluation

Offline coverage: contracts and unknown-vs-zero/false semantics; scoring, retrieval
documents, MMR, and evidence; safe FMA parsing, deterministic Lite selection,
read-only SQLite/FTS5, manifests, checksums, and corrupt-asset fallback; artist-group
splits, release/row gates, intervals, OOD, and mood math; prediction-hidden
annotation; exact/ambiguous/missing identity, injection, unsafe URL, missing
citation, timeout/quota/no-key, and local fallback; session isolation and
feedback-ordering; and fictional-regression + UI/CLI parity. A real release would add
model slices by genre/provenance, Echo-Nest-vs-estimate satisfaction, and measured
warm p95, open time, memory, and artifact/download sizes.

## 9. Risks and limitations

Echo Nest overlap may be selection-biased and is not ground truth; Librosa summaries
cannot encode every musical or cultural quality; artist-group splitting reduces one
leakage mode but not all dataset dependence; a prediction interval is empirical, not
a guarantee; the four-quadrant mood is a coarse two-axis interpretation; FMA metadata
richness varies and can bias text retrieval; missing clean/instrumental booleans
limit hard filters; research sources can be stale or wrong even when cited; and
fixture-backed tests do not substitute for a real pinned build or human calibration.

## 10. Ethics, privacy, and reproducibility

No listener prompt/history/preferences enter research; sensitive guarded requests
cannot call a provider; logs hold sanitized categories/ids/scores/modes — never raw
prompt text; FMA source scopes, licenses, and attribution are preserved; estimates
are visible as estimates and may abstain; human labels are local/pseudonymous. The
runtime is local-first and never deserializes training estimators; the ML/ETL
environment is pinned separately in `requirements-ml.txt`; model/manifest JSON is
canonicalized and predictions are sorted, so reproducibility becomes valid once two
real builds produce matching checksums.

## 11. AI-collaboration reflection

The strongest AI-assisted suggestion was to keep FMA text and structured candidates
independent before fusion, fixing a real recall boundary (sparse prose should not
gate a numeric match). Several attractive suggestions were rejected because they
would make the demo look fuller while weakening evidence: generating missing
descriptions/moods, letting web research reorder results, treating Echo Nest values
as observed truth, passing unknown clean/instrumental fields through hard filters,
and calling the annotation harness "human-calibrated." The final design favors
lineage, abstention, release gates, and strictly downstream research.
*(Owner: revise in your own voice before submission, including one concrete
debugging moment you personally understand.)*
