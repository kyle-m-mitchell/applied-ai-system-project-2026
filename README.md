# Cadence — an evidence-first AI music companion

Cadence turns a natural-language listening request into a small, explainable set
of music recommendations. It is local-first by design: guarding, retrieval,
ranking, evaluation, and fallbacks all run without an API key. Optional AI
providers add semantic retrieval, bounded voice framing, or post-ranking research,
but they never decide which tracks are eligible.

The project began as a deterministic content-based recommender over 20 fictional
songs: one structured taste profile in, a scored top-five list out — no free text,
retrieval, guardrails, evaluation, or interface. Cadence keeps that scorer as a
**regression control** and builds a complete applied-AI system around it, added one
tested feature at a time. It is not a streaming service: no audio, accounts,
persistent profiles, or licensed catalog. Its niche is *transparent* discovery.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q                       # full offline suite (no API key needed)
python scripts/evaluate.py                # evaluation report card (gate + 0.863 control)
python -m src.main --structured-demo      # the original deterministic scorer
python -m src.main "some jazz please"     # the NL companion (FMA catalog by default;
                                          #   add --catalog fictional for the control)
streamlit run streamlit_app.py            # the listener UI
```

Optional Gemini features read `GEMINI_API_KEY`. The UI's *Local-only* control blocks
provider calls for a request; a hosted app still receives browser input, so
"local-only" means Cadence does not forward it to an AI provider.

## System architecture

The authoritative diagram is [`diagrams/architecture.mmd`](diagrams/architecture.mmd).
Three subsystems are deliberately separated:

1. **Offline evidence build.** Untrusted source archives are checksum-verified and
   parsed; optional models are trained, evaluated against baselines, and either
   accepted or made to abstain; values and provenance are baked into read-only
   SQLite. Training libraries never ship in the serving runtime.
2. **Serving.** A guarded request becomes a typed intent; the selected catalog
   supplies candidates; deterministic code ranks them; a grounding evaluator checks
   the output; and Cadence expresses only bounded, evidence-backed claims.
3. **Post-ranking research.** An explicit user action researches one already-chosen
   result; the resulting note sits beside the recommendation and cannot reorder or
   admit a track.

## Retrieval-augmented generation

**Retrieval.** For the fictional control, local TF-IDF over catalog documents and
versioned context guides, plus optional cached/live semantic embeddings. For the
real (FMA) catalog, two *independent* legs query SQLite — FTS5 text search and
structured genre/feature search — each returning up to 200 candidates, fused by
weighted reciprocal rank. Independence matters: text search alone could exclude a
genuinely high-energy track whose description never says "energy."

**Augmentation.** Candidates carry controlled evidence into scoring and
explanation (identity, feature value, provenance, confidence, retrieval reason).
Missing values are omitted; raw biographies or web prose never reach the voice model.

**Generation.** The recommendation list is produced by deterministic ranking code,
not a chat model. The optional language model may only select approved, fact-free
framing (invalid output falls back to a template); track facts are rendered from
validated evidence.

## Core principle: evidence has types

Cadence never conflates these values, and the contracts, scorer, retriever,
evaluator, CLI, and UI all enforce the distinction:

| Value | Meaning | Ranking rule |
|---|---|---|
| `None` | unknown | no reward, penalty, or hard-filter pass |
| `0.0` | known numeric zero | used as real evidence |
| `False` | known boolean false | fails an "only" filter (e.g. instrumental-only) |
| computed / estimated | machine-produced (Echo Nest features; model predictions) | used with lineage; estimates weighted by calibrated confidence |
| authored / supplied | provided by a person or source | source and scope kept visible |

If a missing `energy` became `0.0`, a request for quiet music would reward a track
for data it never had; if `explicit=None` passed a "clean only" filter, Cadence
would claim "clean" without evidence. Treating unknown as unknown is the project's
central design commitment.

## Ranking, personalization, and guardrails

- **Structured-preference hybrid.** Directional cues ("high energy", "acoustic",
  a named genre) are scored against real track features and fused with the text
  leg by percentile rank; a diversity pass (MMR) avoids near-duplicates.
- **Session-only personalization.** Per-track feedback (more/fewer like this,
  didn't fit) accumulates a bounded, reversible re-ranking signal that lives only
  in the browser session — never on the shared engine, so sessions cannot influence
  one another. It nudges ordering; it never overrides an explicit request.
- **Guardrails.** An input guard redacts PII/secrets, strips prompt injection, and
  routes crisis language to a fixed safe response; sensitive input never reaches a
  provider. A grounding evaluator rejects any result that is unverifiable, has
  duplicate or invalid ids, or violates a hard constraint.

## Evaluation

A reproducible harness (`scripts/evaluate.py`) runs labeled cases across a scenario
matrix (local, cached-semantic, provider-outage) and reports a pass/fail gate plus
metrics, storing no query text. The fictional control's historical gate is **0.863
average genre satisfaction**, held fixed as a regression baseline; the real-catalog
path is evaluated separately on feature- and genre-appropriate cases.

```bash
python scripts/evaluate.py           # fictional control gate + metrics
python scripts/evaluate_fma.py       # real-catalog evaluation
python -m pytest --collect-only -q   # test count is generated, never hand-typed
```

## Catalogs and data provenance

Two first-class catalogs, neither privileged. The **fictional** 200-track catalog
is authored, complete, and clearly labeled — the immutable regression control. The
**FMA** catalog is ingested by a deterministic ETL (checksum and safe-ZIP defenses,
explicit parsing, unit conversion without clamping, quarantine of malformed rows,
normalized SQLite + FTS5 + a checksummed manifest). Real data is sparse, so unknown
fields stay unknown: FMA supplies no verified `clean`/`instrumental` boolean, so
Cadence clarifies those hard requests rather than guessing. A committed 300-track
"Lite" edition makes the real path reproducible offline; a full-catalog release is a
separate, evidence-gated build. Optional per-target models predict Echo Nest-style
audio character from FMA's Librosa statistics, split by *artist* to prevent leakage,
and abstain (returning `None`) when a prediction is weak or out of domain.

## Optional post-ranking research

A user may research one selected result. Only its title and artist leave the app —
never the prompt, history, or preferences. It follows a fallback ladder: grounded,
citation-validated web search when available; otherwise a non-grounded note written
*only* from the track's own catalog attributes (clearly labeled "not web-verified");
otherwise a deterministic local summary. Ambiguous identities abstain, and unsafe
URLs, prompt-like page content, missing citations, oversized output, and quota
errors all fail closed.

## Limitations

FMA is broad independent-music metadata, not a licensed streaming catalog; field
coverage is uneven and missing values remain missing. Echo Nest/Librosa values are
machine-computed, and model estimates add a further uncertainty layer. Mood
quadrants (`upbeat`/`calm`/`intense`/`somber`, derived from valence and arousal) are
a transparent experiment, not an objective theory, and remain labeled experimental
pending human calibration. Fixture tests validate code behavior, not real-source
quality; a full build, measured model report, and deployment review are outstanding
launch gates.

## License and data boundaries

- Code: MIT ([`LICENSE`](LICENSE)). Fictional catalog and guides: project-authored.
- FMA metadata: CC BY 4.0; per-track audio licenses are separate and kept attached
  to records — Cadence distributes metadata, not audio.
- DEAM (an isolated non-commercial mood benchmark): CC BY-NC, no production effect.
- Gemini and MusicBrainz are optional network services under their own terms; see
  [`docs/LICENSING.md`](docs/LICENSING.md). References:
  [FMA](https://github.com/mdeff/fma) · [FMA paper](https://arxiv.org/abs/1612.01840)
  · [DEAM](https://cvml.unige.ch/databases/DEAM/manual.pdf).

## AI collaboration reflection

AI assistance helped expand the design space, draft code, surface unknown-data
failure modes, and write adversarial tests. Its most useful suggestion was to split
text retrieval from structured retrieval, so sparse prose cannot decide which
numeric tracks are even considered. A flawed early direction was to infer missing
mood/description fields and to research tracks *before* ranking — which would let
generated material masquerade as source truth and could change eligibility; the
final design instead carries provenance, abstains on the unknown, and keeps research
strictly downstream. *(Owner: personalize before submission.)* The full action trace
is in [`ai_interactions.md`](ai_interactions.md).
