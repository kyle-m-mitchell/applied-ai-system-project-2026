# AI Interactions Log

> Stretch-feature documentation. This is a **draft for the project owner to
> review and personalize** — it records how the applied-AI features were built
> with an AI assistant and the runtime agentic workflow that shipped.

---

## Agentic Workflow (SF8)

**What task did you give the agent?**

Extend the deterministic music recommender with an applied-AI companion, built one
feature at a time with tests and honest documentation: a multi-source retrieval
index (TF-IDF + AI-drafted, versioned context guides), Gemini embeddings with hybrid ranking, a
natural-language input/privacy guard + intent parser, and a bounded companion
("Cadence") with structured-preference fusion, a grounding evaluator, MMR
diversity, an optional voice, an evaluation report card, and a fully integrated
Streamlit interface.

**Prompts used (approach, not verbatim):**

- Work slowly and controlled: plan each feature, get approval, implement, test, then update all docs.
- Keep everything reproducible and free to run: local-first, with any provider call behind a fallback.
- Teach each concept in plain language and pause for understanding checks.
- Preserve the original scorer and the original 20 catalog records.

**What did the agent generate or change?**

Feature by feature: `src/retrieval.py` (retriever interface, TF-IDF, context-guide
query expansion), `src/embeddings.py` + `scripts/build_embeddings.py` (Gemini
embeddings via stdlib REST + committed vector cache), `src/guard.py` /
`src/intent.py` / `src/companion.py` (guard, deterministic intent parser, bounded
agent), and `src/ranking.py` / `src/evaluator.py` / `src/generation.py` /
`src/voice.py` (MMR diversity, grounding evaluator, provider text, Cadence voice),
then `src/refine.py`, `ui/`, and `streamlit_app.py` (typed refinement, reversible
session state, evidence cards, privacy controls, and request-local diagnostics),
plus the offline test and evaluation suites and supporting documentation.

**Runtime agentic workflow (the shipped artifact).** `MusicCompanion.respond()` is
itself a bounded agent: guard → intent → retrieve → structured/text fusion →
diversify → evaluate → voice,
choosing from an allowlist of actions (`recommend / clarify / no_match /
safe_response / degraded`) and emitting a privacy-safe `AgentTrace` plus a
request-local `PipelineReceipt`. A representative controlled trace is:

```
guard_category=ok · intent(mood=chill, instrumental_only=True, clean=True)
candidate_ids=(controlled catalog ids) · final_ids=(grounded subset)
diversity=balanced · evaluation.ok=True · action=recommend
embedding_source=cache|live|local · network_used=<true|false>
voice_source=generated|template
```

A sensitive query (e.g. one containing an email) produces
`guard_category=sensitive`, the email redacted before retrieval, and
`voice_source=template` — the language provider is never consulted, and no raw
text appears in the trace. Reproduce with `python -m src.main --trace "<query>"`.

## Phase 4 Action / Observation / Decision Trace

This is an engineering audit trail of observable actions and decisions, not
private chain-of-thought.

| Area | Action | Observation | Decision / evidence |
|---|---|---|---|
| Privacy-safe query handling | Routed initial text and follow-ups through `InputGuard`; cleared submitted widgets; excluded prompt text from URLs, evolution summaries, receipts, and JSONL events. | Guarded text is still required inside the current session for retrieval, refinement, and exact undo. Claiming that no server ever handles it would be false on a hosted app. | Keep only guarded query text in per-browser session state; discard query parameters; explain that local-only blocks onward AI-provider calls, not the browser-to-app-server request. PII AppTests confirm raw addresses do not appear in fields, rendered output, query parameters, or receipts. |
| Local-only enforcement | Added typed `ExecutionPolicy(force_local=...)` at the `MusicCompanion` boundary and a separate provider-free retriever; made sensitive routing sticky across refinements. | A UI badge alone cannot prevent an embedding or generation call. Cached vectors are provider-free even though AI built them. | The backend selects the provider-free path and deterministic voice. Tests use exploding/counting provider doubles to prove zero calls; receipts distinguish `CACHE`, `LIVE`, and `LOCAL`. |
| Transactional controls | Put Taste Console controls inside a form and execute only on **Remix**; kept quick moves as explicit submitted actions; made tempo opt-in and preserved non-`near` tempo rules unless changed. | Streamlit reruns on widget interaction, which could otherwise create repeated provider calls and noisy history. Unsupported or repeated refinements could also create fake evolution. | Draft changes do not call the engine. Typed `IntentPatch` validates one goal per feature; unsupported follow-ups clarify; no-op quick moves do not create snapshots; undo restores the exact immutable prior turn. |
| Provenance and diagnostics | Added `CompanionTurn(response, receipt)` with candidate/final IDs, source, network use, latency, guard category, diversity, and fingerprints. | The old `gemini` mode could not distinguish an offline query-cache hit from a live request, and a failed live attempt initially lost its network provenance. | Record embedding source independently and preserve `LIVE` on a failed attempt before local fallback. Developer view reads the current request-local receipt rather than a shared log and never renders prompt text. |
| Output and explanation guardrails | Separated Cadence's framing from the deterministic track cards; constrained the model to selecting exact application-owned microcopy; audited score labels and evidence rules. | Denylists caught “slow acoustic instrumentals” but could never enumerate claims such as release year, nationality, duration, or awards. Also, semantic `0.0` means evaluated/no match, and MMR can reorder after fusion. | Require exact membership in `APPROVED_FRAMINGS` or use the template fallback; require positive evidence; empty rejected payloads; label **Fused relevance** as pre-diversity; keep `N/A` distinct from `0.0`. |
| Verification | Ran backend, evaluation, refinement, cache-failure, CLI-policy, and Streamlit AppTest paths through the same public services. | Presentation changes can appear correct while changing IDs, weakening privacy, retaining stale query cues, or rerunning the engine unexpectedly. | Current result: **216 tests pass offline**. AppTest covers backend/UI ID parity, bounded action states, privacy, cache/fallback labels, transactional controls, replacement searches, refinements, undo, sticky sensitivity, and action-aware developer view. The evaluation gate remains PASS with 100% hard-constraint adherence and 0.863 average genre satisfaction. |
| Human review | Deferred claims that require a real browser or deployment. | Headless AppTest proves functional behavior, not responsive layout, keyboard flow, screen-reader quality, reduced motion, contrast, or hosted-provider configuration. | Connected-browser desktop/mobile/accessibility QA and a provider-disabled staging smoke test remain pending and must be recorded before launch claims. |

**What did you verify or fix manually?**

- Caught a real API key accidentally used as a test fixture and replaced it with a fake token; confirmed no key is committed.
- Verified provider model names instead of trusting an assumed name: retained the
  available embedding model and replaced the unsupported text-model assumption
  with `gemini-flash-lite-latest`.
- Fixed real integration issues surfaced by running it: an early local SDK install
  failed, so the small provider seam used inspectable stdlib REST; TLS certificate
  verification needed `certifi`; and the embedding build hit a 429 rate limit, so
  throttle, backoff, resumable caching, and complete-cache validation were added.
  Current `google-genai` releases support Python 3.14, so the SDK remains a viable
  future choice rather than being ruled out by that historical failure.
- Rejected several attractive but flawed UI suggestions: prompt-bearing share
  URLs, provider calls on every slider rerun, a "Familiar" control without
  popularity/history data, and an ablation chart reconstructed from final cards.
  Each would have made the interface more impressive-looking but less truthful.
- Verified the implemented paths with **216 offline tests**, AppTest, the
  evaluation gate, and end-to-end CLI runs. Human review of AI-drafted catalog
  and context-guide prose and connected-browser visual/accessibility QA remain
  explicit owner tasks.

---

## Design Pattern (SF10)

**Which design pattern did you use?**

**Strategy** (with a **Factory** helper). Retrieval, embedding, and text generation
are each defined as an interface with interchangeable implementations, so a local
deterministic strategy and an optional provider strategy are swappable without
touching callers — and a fallback is always available.

**How did AI help you brainstorm or implement it?**

The assistant proposed the interface-first, local-first shape early and reused it
for every provider touchpoint, which is what made "use real AI but stay
reproducible" achievable: tests and demos run on the deterministic strategies with
no key, while the provider strategies plug in behind the same interface.

**How does the pattern appear in your final code?**

- `Retriever` (`src/retrieval.py`) → `TfidfRetriever`, `EmbeddingRetriever`, `HybridRetriever`.
- `Embedder` (`src/embeddings.py`) → `FakeEmbedder`, `CachedQueryEmbedder`, `GeminiEmbedder`.
- `TextGenerator` (`src/generation.py`) → `FakeTextGenerator`, `GeminiTextGenerator`.
- Cadence's voice picks a strategy per call (template vs Gemini-selected approved line) with a fallback.
- Factory: `build_default_retriever()` (`src/retrieval.py`) selects hybrid vs TF-IDF from what's available.
