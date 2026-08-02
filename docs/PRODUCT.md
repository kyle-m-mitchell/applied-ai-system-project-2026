# Cadence — Product Brief

## Positioning

**Cadence is an evidence-first companion for discovering independent music.** A
listener describes a moment in ordinary language; Cadence returns a small,
diverse set, shows which evidence shaped it, admits what the catalog cannot prove,
and keeps useful local behavior when optional services fail.

Cadence competes on trust and control—not catalog scale, playback rights, or
surveillance-heavy personalization.

## The listener problem

Popular music products are excellent at playback and behavior-based discovery,
but their recommendation logic is difficult to inspect. A listener may not know:

- whether a request was treated as a requirement or a preference;
- which words, audio traits, or past behavior changed the result;
- whether a missing field was silently guessed;
- what data left the device/app;
- whether recommendations survive a provider outage.

Cadence makes those decisions visible. It is useful for a listener exploring real
FMA metadata and for a student learning how production-minded AI systems handle
retrieval, uncertainty, providers, evaluation, and human review.

## Product promise

Cadence should feel warm and alive without pretending to be a person or an
all-knowing music critic. Its personality is curious, concise, tactful, and
honest. A characteristic response is:

> I can't verify clean lyrics in this catalog, and I'd rather not guess. I can
> remove that requirement or switch catalogs.

That sentence is product behavior, not decoration: the FMA capability descriptor
cannot support a clean-only hard filter, so the bounded agent clarifies before
ranking.

## Core experience

1. The listener chooses FMA or the fictional control and types a request.
2. A privacy/input guard validates length, redacts sensitive material, removes
   injection-like instructions, and routes crisis content safely.
3. A deterministic parser produces typed intent: hard constraints stay distinct
   from soft preferences.
4. The catalog declares what it can support. Unsupported hard filters clarify.
5. FMA independently retrieves text and structured candidates, fuses their ranks,
   scores trustworthy values with provenance/confidence, and applies diversity.
6. A grounding evaluator checks IDs, constraints, evidence, and count.
7. Cadence renders a bounded response and evidence-rich cards. Local templates
   remain available on every provider failure.
8. The listener may explicitly research one result. Identity and citations are
   checked; the list never changes.

The fictional path remains a complete regression control with its existing
catalog, context guides, embedding cache, scorer, and `0.863` evaluation baseline.

## What makes Phase 5 product-like

### Honest real-data adaptation

FMA does contain track information, album information, artist biographies, tags,
genres, and some URLs/licenses—but not uniformly. Cadence preserves their scopes
and coverage instead of flattening them into an invented “track description.”

### Useful uncertainty

The specialized pipeline tries to fill numeric gaps from Librosa features, but a
prediction survives only after model-level and row-level gates. A partially
covered real catalog is better than a cosmetically complete fabricated one.

### Two-path discovery

Text search can find names, genres, tags, and supplied context. Structured search
can find high-energy, calm, acoustic, or tempo-relevant records even when prose is
sparse. Reciprocal-rank fusion lets both paths contribute without pretending that
their raw scores share one scale.

### Transparent cards

An FMA result card can distinguish:

- Full or Lite source edition;
- track, album, and artist context;
- known vs missing audio features;
- Echo Nest-computed vs model-estimated values;
- experimental mood and uncertainty;
- license/attribution;
- unsupported capabilities;
- optional research claims and citations.

### Reversible, privacy-aware interaction

Catalog switches reset mix history, undo snapshots, ratings, and research state so
IDs/evidence cannot cross catalog boundaries. Research is session-only. Provider
use is controlled by backend policy and surfaced in a request-local receipt.

## Target user and launch niche

Primary early user: an adventurous listener who values independent music,
explanations, explicit controls, and privacy more than seamless playback.

Secondary user: an AI/software student who wants a visible example of RAG,
specialized prediction, abstention, agents, guardrails, evaluation, lineage, and
human calibration in one coherent product.

A realistic first launch is a public metadata-discovery demo with FMA Lite always
available and a checksummed Full download for capable deployments. It should not
be marketed as a commercial streaming service.

## Competitive comparison

| Dimension | Mainstream streaming product | Cadence Phase 5 |
|---|---|---|
| Catalog/playback | Huge licensed catalog and playback | FMA metadata discovery; no audio |
| Personalization | Rich behavioral history and collaborative signals | Explicit request/session controls; no persistent profile |
| Explanation | Usually compact/opaque | Field-level evidence, lineage, confidence, source scope |
| Missing data | Product-specific/mostly hidden | Unknown is visible and score-neutral |
| Provider outage | Cloud feature may disappear | Deterministic local fallback |
| Research | General artist/track pages | Optional cited brief after ranking only |
| Evaluation | Internal and proprietary | Offline tests, public harness, regression control, generated reports |

Cadence cannot win on scale. It can be unusually strong on inspectability,
reproducibility, and respectful uncertainty.

## Non-goals

Phase 5 does not add:

- playback, downloads, or blanket audio rights;
- accounts, authentication, social sharing, or collaborative filtering;
- long-term listener profiles or ratings-driven ranking;
- Spotify/Apple/YouTube integration;
- automatic web research for every result;
- web-derived rank changes or persistent catalog mutations;
- a claim that experimental mood is human calibrated;
- DEAM-derived production thresholds or weights.

## Product-quality gates

Code completion and launch readiness are different. Public Full launch requires:

- a pinned-source real build with accepted/quarantined counts and field coverage;
- target-by-target model reports and honest unreleased/abstained outcomes;
- byte/checksum determinism and artifact/manifest agreement;
- a committed verified 300-track Lite fallback;
- 100% citation coverage on published research fixtures;
- the unchanged fictional evaluation control;
- measured Full warm p95 under one second on a named test machine;
- measured open time, memory, artifact size, and first-download time;
- catalog-switch state-isolation tests;
- connected-browser desktop/mobile/accessibility review;
- licensing, attribution, secrets, and deployment review.

Counts and metrics must be generated by builds/tests; they are never copied from
the roadmap as if already achieved.

## Next product investments after Phase 5

1. Complete and publish the real reproducible Full/Lite build evidence.
2. Run the prediction-hidden human annotation study and define an explicit
   agreement/promotion decision.
3. Add first-party playback links only where source rights and stable identifiers
   support them; do not synthesize links.
4. Evaluate session-only preference learning before allowing ratings to affect
   ranking.
5. Add production authentication/rate limits/monitoring only when the public usage
   model requires them.
6. Measure whether dense semantic retrieval improves FMA beyond SQLite FTS5 before
   introducing a vector database.

The product principle for every future feature remains: **more capability must
also add more evidence, a boundary, and a way to fail honestly.**
