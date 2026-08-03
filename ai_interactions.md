# AI Interactions Log

## Agentic workflow (bonus)

**Task.** Extend a deterministic music recommender into an applied-AI companion,
built one tested feature at a time: multi-source retrieval (TF-IDF + versioned
context guides), Gemini embeddings with hybrid ranking, a natural-language
input/privacy guard and intent parser, and a bounded companion ("Cadence") with
structured-preference fusion, a grounding evaluator, MMR diversity, an optional
voice, an evaluation harness, and a Streamlit interface — later a real (FMA) catalog,
session-only personalization, and post-ranking research.

**Guiding prompts (approach, not verbatim):** work slowly and get approval per
feature; stay reproducible and free to run (local-first, every provider call behind a
fallback); teach each concept in plain language; and preserve the original scorer and
its 20 original catalog records.

**Runtime agent (the shipped artifact).** `MusicCompanion.respond()` is itself a
bounded agent — guard → intent → retrieve → structured/text fusion → session-taste
re-rank → diversify → evaluate → voice — choosing from an allowlist of actions
(`recommend / clarify / no_match / safe_response / degraded`) and emitting a
privacy-safe `AgentTrace` plus a request-local `PipelineReceipt`. A sensitive query
(e.g. one containing an email) yields `guard_category=sensitive`, redaction before
retrieval, and `voice_source=template`: the provider is never consulted and no raw
text appears in the trace. Reproduce with `python -m src.main --trace "<query>"`.

## Action / decision trace

An engineering audit trail of observable decisions, not private reasoning:

| Area | Decision and evidence |
|---|---|
| Privacy-safe queries | Guarded text stays only in per-browser session state (needed for refinement/undo); query params are discarded; local-only blocks onward provider calls, not the browser→server request. PII AppTests confirm raw addresses never appear in fields, output, params, or receipts. |
| Local-only enforcement | Enforced by a typed `ExecutionPolicy(force_local=…)` at the `MusicCompanion` boundary with a provider-free retriever, not a badge; sensitive routing is sticky. Exploding/counting provider doubles prove zero calls; receipts distinguish `CACHE`/`LIVE`/`LOCAL`. |
| Transactional controls | Taste Console runs once on **Remix**, not per widget; a typed `IntentPatch` validates one goal per feature; unsupported follow-ups clarify; no-op moves create no snapshot; undo restores the exact prior turn. |
| Provenance / diagnostics | `CompanionTurn(response, receipt)` records candidate/final ids, embedding source, network use, latency, guard category, and fingerprints; `LIVE` is preserved on a failed attempt before local fallback; the developer view reads the current receipt, never a shared log or prompt text. |
| Output guardrails | Cadence framing is separated from deterministic cards and constrained to exact membership in `APPROVED_FRAMINGS` (else template); a denylist could never enumerate invented claims (release year, nationality, awards). `N/A` (not evaluated) is kept distinct from `0.0` (evaluated, no match). |
| Human review | Headless AppTest proves functional behavior, not layout, keyboard flow, screen-reader quality, or contrast; connected-browser accessibility QA and a provider-disabled staging smoke test remain pending. |

## What I verified or fixed manually

- Caught a real API key accidentally used as a test fixture and replaced it with a
  fake token; confirmed no key is committed.
- Verified provider model names rather than trusting an assumed one (kept the working
  embedding model; corrected the text model to `gemini-flash-lite-latest`).
- Fixed real integration issues surfaced by running it: a failed local SDK install led
  to an inspectable stdlib-REST provider seam; TLS verification needed `certifi`; and a
  429 during the embedding build motivated throttling, backoff, and resumable caching.
- Rejected attractive-but-flawed suggestions: prompt-bearing share URLs, provider
  calls on every slider rerun, a "Familiar" control without popularity data, inferring
  missing mood/description fields, and letting web research reorder results — each would
  look more impressive while being less truthful.
- Verified the implemented paths with the offline test suite, AppTest, the evaluation
  gate (100% hard-constraint adherence, 0.863 average genre satisfaction), and CLI
  runs. Human review of AI-drafted catalog/guide prose and visual/accessibility QA
  remain explicit owner tasks.

## Design pattern (bonus)

**Strategy**, with a **Factory** helper. Retrieval, embedding, and text generation are
each an interface with interchangeable implementations, so a local deterministic
strategy and an optional provider strategy are swappable without touching callers, and
a fallback is always available — this is what makes "use real AI but stay
reproducible" achievable (tests and demos run on the deterministic strategies with no
key). In the code:

- `Retriever` → `TfidfRetriever` / `EmbeddingRetriever` / `HybridRetriever`;
- `Embedder` → `FakeEmbedder` / `CachedQueryEmbedder` / `GeminiEmbedder`;
- `TextGenerator` → `FakeTextGenerator` / `GeminiTextGenerator`;
- the voice picks template vs. Gemini-selected line per call, with a fallback;
- `build_default_retriever()` selects hybrid vs. TF-IDF from what is available.
