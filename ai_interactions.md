# AI Interactions Log

> Stretch-feature documentation. This is a **draft for the project owner to
> review and personalize** — it records how the applied-AI features were built
> with an AI assistant and the runtime agentic workflow that shipped.

---

## Agentic Workflow (SF8)

**What task did you give the agent?**

Extend the deterministic music recommender with an applied-AI companion, built one
feature at a time with tests and honest documentation: a multi-source retrieval
index (TF-IDF + curated context guides), Gemini embeddings with hybrid ranking, a
natural-language input/privacy guard + intent parser, and a bounded companion
("Cadence") with a grounding evaluator, MMR diversity, and an optional voice.

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
plus the full offline test suite and every doc.

**Runtime agentic workflow (the shipped artifact).** `MusicCompanion.respond()` is
itself a bounded agent: guard → intent → retrieve → diversify → evaluate → voice,
choosing from an allowlist of actions (`recommend / clarify / no_match /
safe_response / degraded`) and emitting a privacy-safe `AgentTrace`. Example
trace for `"clean chill beats for studying, no vocals"`:

```
guard_category=ok · intent(instrumental_only=True, clean=True)
retrieved_ids=(...) · diversity_applied=True · evaluation.ok=True
action=recommend · voice_source=gemini
```

**What did you verify or fix manually?**

- Caught a real API key accidentally used as a test fixture and replaced it with a fake token; confirmed no key is committed.
- Corrected model names the assistant assumed: the real API had no `gemini-embedding-2`… wait, it did, but had no `gemini-3.5-flash-lite` (used `gemini-flash-lite-latest`).
- Fixed real integration issues surfaced by running it: the `google-genai` SDK would not build on Python 3.14 (switched to stdlib REST), TLS cert verification needed `certifi`, and the embedding build hit a 429 rate limit (added throttle, backoff, and resumable caching).
- Verified every feature with tests and end-to-end CLI runs; reviewed all AI-drafted prose (catalog, context guides, this log) for accuracy.

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
- Cadence's voice picks a strategy per call (template vs grounded Gemini) with a fallback.
- Factory: `build_default_retriever()` (`src/retrieval.py`) selects hybrid vs TF-IDF from what's available.
