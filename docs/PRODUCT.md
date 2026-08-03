# Cadence — Product Brief

## Positioning

**Cadence is an evidence-first companion for discovering independent music.** A
listener describes a moment in ordinary language; Cadence returns a small, diverse
set, shows the evidence behind it, admits what the catalog cannot prove, and keeps
useful local behavior when optional services fail. It competes on trust, control,
and inspectability — not catalog scale, playback rights, or behavioral profiling.

## The problem

Mainstream music products are strong at playback and behavior-based discovery, but
their recommendation logic is hard to inspect. A listener rarely knows whether a
request was treated as a requirement or a preference, which signals changed the
result, whether a missing field was silently guessed, what data left the app, or
whether the recommendations survive a provider outage. Cadence makes those
decisions visible — useful both to a listener exploring real (FMA) metadata and to
a student studying how production-minded AI systems handle retrieval, uncertainty,
providers, evaluation, and human review.

## Product promise

Cadence is warm and concise without pretending to be a person or an all-knowing
critic. A characteristic response is behavior, not decoration:

> I can't verify clean lyrics in this catalog, and I'd rather not guess. I can
> remove that requirement or switch catalogs.

FMA cannot support a clean-only hard filter, so the bounded agent clarifies before
ranking rather than fabricating certainty.

## Core experience

1. The listener picks a catalog (real FMA or the fictional control) and types a request.
2. A privacy/input guard checks length, redacts sensitive text, strips injection,
   and routes crisis content to a safe response.
3. A deterministic parser produces typed intent, keeping hard constraints distinct
   from soft preferences; unsupported hard filters trigger a clarification.
4. Retrieval, structured/text fusion, provenance-aware scoring, and a diversity
   pass produce candidates; a grounding evaluator checks ids, constraints, and evidence.
5. Cadence renders a bounded response with evidence-rich cards; local templates
   remain available on any provider failure.
6. The listener may explicitly research one result — identity and citations are
   checked, and the recommendation list never changes.

## What makes it different

- **Inspectability** — field-level evidence, provenance, confidence, and retrieval
  reason on every card, versus compact/opaque explanations elsewhere.
- **Honest uncertainty** — unknown data stays visible and score-neutral; a
  partially-covered real catalog is preferred over a cosmetically complete fabricated one.
- **Two-path retrieval** — text search finds names/genres/context; structured
  search finds high-energy/calm/acoustic/tempo records even when prose is sparse.
- **Resilience** — a deterministic local fallback replaces any cloud feature that fails.
- **Reversible, private interaction** — session-only feedback and research;
  provider use is backend-enforced and shown in a request-local receipt.

## Non-goals

No playback or audio rights; no accounts, social features, or collaborative
filtering; no persistent cross-session profiles; no third-party service
integration; no automatic web research; no web-derived rank changes; and no claim
that the experimental mood model is human-calibrated.

## Launch readiness

Code completion and launch readiness are distinct. A public release would require a
pinned-source real build (with generated accepted/quarantined counts and coverage),
per-target model reports, checksum determinism, the committed Lite fallback, the
unchanged fictional evaluation control, and an accessibility/deployment review — and
every count and metric must be *generated* by builds and tests, never copied as if
already achieved. The guiding principle for any future feature: **more capability
must also add more evidence, a boundary, and a way to fail honestly.**
