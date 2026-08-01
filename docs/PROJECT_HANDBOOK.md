# Project Handbook: Applied AI Music Companion

This is the durable source of truth for the project. It is written for two
audiences: a beginner learning the system from first principles and a future AI
assistant that may have none of the conversation history. Update it whenever an
implemented feature, major decision, dependency, test result, or limitation
changes.

> **Status language matters:** “Implemented” describes code that exists and has
> been verified. “In progress” describes uncommitted or incomplete work.
> “Planned” describes intent only. Never present planned architecture as working
> software.

## Recover this project in a new chat

Copy and paste this prompt into a new chat while the repository is open:

```text
Read docs/PROJECT_HANDBOOK.md completely, then inspect git status, the current
commit, repository files, and the latest test result. Treat “Implemented” as
fact and “Planned” as intent. Preserve the original scoring core and original
20 catalog records unless the handbook explicitly says otherwise. Continue
from “Next action,” verify each feature end to end, and update the handbook,
architecture diagram, tests, and evidence after completing it. Teach every
concept in beginner-friendly language and give me short understanding checks.
```

Do not trust a test count, commit hash, provider model name, free-tier rule, or
working-tree statement merely because it appears below. Re-run the associated
check first; those facts can become stale.

## Live project snapshot

Last updated: **2026-07-30**

| Item | Current state |
|---|---|
| Branch | `main`; the retrieval, guard/intent, and agent/Cadence work is committed. The current working tree holds the Step-1 stabilization fixes (cache resume, cache/cosine validation, query-key normalization, no raw-PII echo, tighter response/trace contracts, bounded retries, MMR relevance floor, docs). Inspect with `git status --short`. |
| Feature | **Feature 6 (bounded agent + grounding evaluator + MMR diversity + Cadence's voice) implemented and tested; live voice validated against the real API. A Step-1 stabilization pass (from a code review) then hardened caches, contracts, retries, MMR, and docs. Human sign-off pending** |
| Working tree | Uncommitted Step-1 stabilization fixes across `src/{embeddings,retrieval,contracts,voice,companion,generation,ranking,evaluator,main}.py`, `scripts/build_embeddings.py`, tests, and docs (see the Branch row) |
| Last verified regression check | `115 passed` from `.venv/bin/python -m pytest -q` on 2026-07-31 (fully offline; no key) |
| Implemented | Original scorer, contracts, service, 200-track catalog, catalog data card, TF-IDF + context-guide retrieval, Gemini embeddings + hybrid ranking (committed cache + fake + fallback), input/privacy guard + deterministic intent parser, **a bounded `MusicCompanion` agent with a privacy-safe trace, MMR diversity (relevance-floored), a grounding evaluator, and Cadence's voice (deterministic baseline + optional grounded generated renderer, output-guarded) — wired through the CLI**, full offline test suite, before/after demos, Mermaid architecture |
| In progress | Human review of catalog records, AI-drafted context-guide wording, and the `ai_interactions.md`/voice-card drafts |
| Not implemented yet | Gemini structured intent; evaluation harness (Feature 7); session feedback + web UI (Feature 8); privacy-safe JSONL logging + presentation (Feature 9) |
| Next action | Feature 7 (evaluation harness — build before the UI), then Feature 8 (session feedback + stdlib web UI); optional Gemini structured-intent parser; complete human sign-off |

### Current implementation boundary

Two validated entry points share the one catalog:

```text
structured path:  RecommendationRequest → RecommendationService → scorer → RecommendationResult
language path:     text → MusicCompanion → InputGuard → IntentParser → Retriever → CompanionResponse
```

The natural-language path is honest *because* the guard and deterministic intent
parser now exist: free text is size-checked, PII/secrets are redacted (and kept
local, never sent to the provider), injection directives are stripped, crisis
language gets a safe response, and only then does the parsed intent drive
retrieval. Sensitive input never reaches Gemini. `RecommendationRequest` remains
structured-only by design; language enters through `MusicCompanion`, not a
`query` field bolted onto the scorer request.

## Assignment and definition of success

The assignment extends an existing project with at least one substantial,
integrated Applied AI feature. The extension must change real application
behavior, work end to end, include responsible-design protections and evidence,
and be understandable to another developer.

Our goal is the **full 21 required points plus 8 stretch points**, not a bare
pass. That means the finished product should demonstrate RAG, an agentic
workflow, specialized companion behavior, and an evaluation harness while
remaining small enough to explain and operate reliably.

### Rubric-to-evidence map

| Rubric area | Points | Planned evidence | Status |
|---|---:|---|---|
| Original project and scope | 3 | README and model card describe the 20-track deterministic baseline | Implemented; wording needs final refresh |
| Substantial integrated AI feature | 3 | Natural-language request changes retrieval, ranking, and grounded response through the shared service | Implemented (guard → intent → hybrid retrieval → grounded Cadence response via the CLI) |
| Mermaid architecture | 3 | `diagrams/architecture.mmd`, plus optional rendered preview | Target source implemented; synchronize with final code |
| End-to-end demonstration | 3 | Streamlit or CLI plus 2–3 reproducible README runs | CLI baseline implemented; AI examples planned |
| Reliability or guardrail | 3 | Contracts, input/privacy guard, grounded output evaluator, fallback | Implemented (contracts, guard, grounding evaluator, and local fallbacks for retrieval and voice) |
| README and setup | 3 | Goals, installation, run/test commands, sample output | Partially implemented |
| AI collaboration reflection | 3 | Model card records prompting, debugging, one useful and one flawed suggestion, limits | Planned refresh |
| Multi-source RAG bonus | +2 | Song records + context guides + session feedback, with before/after evidence | Implemented (two sources: catalog + curated context guides, with guide-driven query expansion and a before/after demo; session feedback is the 3rd source, added in Feature 8) |
| Agentic workflow bonus | +2 | Bounded steps/tool calls and structured trace in `ai_interactions.md` | Implemented (bounded `MusicCompanion` with an `AgentTrace`; `ai_interactions.md` drafted) |
| Specialized behavior bonus | +2 | Cadence voice card/few-shot examples and baseline comparison | Implemented (`docs/CADENCE_VOICE.md` voice card + few-shot; grounded Gemini voice with a deterministic baseline comparison) |
| Evaluation harness bonus | +2 | Predefined cases and a pass/fail metric summary | Planned |

Required final artifacts also include a 5–7 minute presentation and portfolio
reflection. Those are documentation/delivery tasks, not runtime features, but
they must remain on the final checklist.

## Original project: what we are extending

The base project is **Music Recommender Simulation**, called **TasteTether 1.0**
in the original model card. It is a deterministic content-based recommender.

The original system:

- loads 20 fictional songs from a CSV file;
- accepts genre, mood, energy, acousticness, valence, danceability, and tempo;
- compares each song with those preferences using a readable weighted formula;
- gives partial credit to manually defined genre and mood “families”;
- sorts by score descending and uses song ID as a stable tie-breaker;
- prints top recommendations and feature-level reasons.

“Deterministic” means the same data and request always produce the same answer.
There is no trained model, vector search, external provider, listener history,
or natural-language understanding in the baseline.

The scoring core is intentionally preserved as a trusted local baseline. New AI
capabilities should wrap or combine with it through `RecommendationService`, not
silently replace a known-good algorithm. The README contains the full formula
and earlier scoring experiments.

## Product vision: give the application life responsibly

The target product is an explainable music companion with the working name
**Cadence**. Cadence should feel like a warm, observant fictional DJ: concise,
curious, tasteful, and willing to ask one useful question when the request is
ambiguous. Personality is not permission to invent facts.

Cadence may:

- recommend tracks from the validated catalog;
- explain recommendations using retrieved metadata and scorer evidence;
- ask for clarification when preferences conflict or are too vague;
- acknowledge when the catalog has no strong match;
- respond gracefully to irrelevant, unsafe, or sensitive input;
- disclose when the external provider is unavailable and local mode is active;
- remember preferences and feedback only within the current session.

Cadence must not:

- claim consciousness, feelings, a human identity, or a real personal history;
- pretend to have heard a fictional song;
- invent songs, artists, catalog fields, citations, or match confidence;
- act as a therapist, crisis counselor, doctor, or emergency authority;
- manipulate a listener into emotional dependence;
- retain a long-term personal profile without a future explicit consent design;
- hide degraded mode or present a fallback result as provider-generated.

The companion is a **presentation and decision-policy layer**. Retrieval,
ranking, validation, and evidence remain authoritative.

## Target architecture and data flow

Canonical source: [`diagrams/architecture.mmd`](../diagrams/architecture.mmd)

Rendered preview: [`assets/architecture.png`](../assets/architecture.png)

Planned normal path:

```text
Listener
  → input/privacy guard
  → structured Cadence intent and allowlisted plan
  → multi-source retrieval with provenance
  → hard filters + hybrid ranking + diversity
  → runtime evaluator
  → situation policy
  → companion renderer
  → output guard
  → recommendations, reasons, evidence, and operating mode
```

Planned failure path:

```text
Gemini unavailable, times out, or remains invalid after one repair
  → deterministic rule parser + local TF-IDF + original scorer
  → the same evaluator and output guard
  → explicit degraded/local-mode response
```

Humans are part of the system:

- a curator reviews catalog records and context guides;
- a developer reviews model/tool decisions and privacy boundaries;
- a tester evaluates normal, edge, adversarial, and outage behavior;
- the listener can correct preferences, reject a suggestion, and reset memory.

The Mermaid file currently shows a **target architecture**. The assignment
requires the final diagram to match actual code, so remove or relabel any
component that remains unimplemented at submission time.

## Foundational concepts

### Validation

Validation checks whether data obeys a contract before the system relies on it.
Examples include an energy value between 0 and 1, a real boolean rather than the
string `"yes"`, and a recommendation ID that exists in the catalog. Validation
protects the request and the result as they pass through multiple components.

### Retrieval-Augmented Generation (RAG)

RAG means “retrieve trusted information first, then let a language model use
that information.” The language model is not the catalog. It receives a small,
relevant evidence packet and must ground its answer in that packet. In this
project, RAG prevents Cadence from relying on generic music knowledge or making
up fictional catalog facts.

### Embeddings

An embedding converts text into a vector: a list of numbers representing
semantic meaning. Similar requests and track descriptions tend to have vectors
pointing in similar directions. Cosine similarity measures that closeness. An
embedding does not prove relevance; metadata quality, filters, ranking, and
evaluation still matter.

### Agentic workflow

An agent is not “an LLM with unlimited freedom.” Our agent is a bounded workflow
that chooses from a small allowlist of actions such as `clarify`, `retrieve`,
`recommend`, `no_match`, or `safe_response`. Its intermediate record stores
structured action summaries, retrieved IDs, tool outcomes, and validation
results—not private hidden chain-of-thought.

### Reliability harness

A reliability harness repeatedly sends predefined cases through the same public
service used by the UI and reports measurable outcomes. Unit tests prove small
code rules. An evaluation harness measures system behaviors such as retrieval
recall, catalog faithfulness, constraint adherence, fallback success, and tone.

### Local fallback

The local fallback is a complete, deterministic path that does not require an
API call. It is more than a canned apology: it should still parse supported
preferences, retrieve candidates with local TF-IDF, rank them with the original
scorer, validate results, and clearly label the operating mode. This keeps the
product useful, testable, free to demo, and honest during provider failures.

## Design decisions

| Decision | Why | Status |
|---|---|---|
| Build for the full 29 points | The project should teach several modern AI patterns and be portfolio-worthy | Active |
| Use Cadence as a fictional DJ companion | Personality makes interaction coherent without pretending the system is human | Working decision; final name still open |
| Keep memory session-only | Personalization without long-term profiling or a database | Active |
| Preserve the original scorer | It is explainable, deterministic, and provides a reliable fallback/baseline | Implemented |
| Put one service boundary around all interfaces | CLI, UI, agent, and tests must exercise the same application logic | Implemented |
| Add schemas before natural language | The system should not accept inputs it cannot honestly process | Implemented |
| Validate both input and output | Protecting only the request would still allow invalid or invented results downstream | Partially implemented |
| Use one repair attempt | A bounded retry can correct formatting; repeated self-repair adds cost and unpredictability | Planned |
| Fall back locally after failure | The app should remain functional and disclose degraded mode | Local baseline exists; provider switch planned |
| Use specialized prompting, not call it fine-tuning | Few-shot examples and a voice card are honest, cheap, and measurable | Planned |
| Use in-memory vectors for 200 songs | A vector database adds complexity without useful scale benefits | Implemented (TF-IDF, in-memory Python dicts) |
| Use pure-Python stdlib TF-IDF instead of scikit-learn | Zero new dependencies, fully inspectable math, no wheel/compat risk on Python 3.14.5, and trivially fast at 200 short docs; scikit-learn/NumPy were not installed | Implemented |
| Build the catalog-only retriever first; defer context guides to Feature 3b | Keeps one controlled, fully testable step; the `Retriever` interface and `SourceType` already leave room for a second source | Implemented (3b done) |
| Use context guides as query expansion + evidence, not as recommendable items | A guide is not a track; expanding the query with a guide's catalog-vocabulary terms improves track retrieval while keeping "tracks are the only recommendable items" intact | Implemented |
| Gate guide expansion with a dominance threshold (≥ 0.5 × top guide score) | Drops weak, spurious guide matches that would otherwise inject off-topic expansion terms | Implemented |
| Use real Gemini embeddings but keep the system reproducible via a committed cache + deterministic fake + TF-IDF fallback | Gemini is the tool that *builds* a reproducible artifact; the committed vectors and offline fallback are what make it portable and testable with no key | Implemented |
| Feature 4 hybrid blends semantic + lexical (not yet the 55/35/10) | The numeric-scorer and feedback weights need the Feature 5 intent parser and Feature 8 memory; dense+sparse is the buildable blend now, with configurable weights | Implemented |
| Call the Gemini REST embeddings endpoint with the standard library (`urllib`), not the `google-genai` SDK | Zero third-party dependencies; the SDK's `cryptography` dep has no Python-3.14 wheel and needs Rust/OpenSSL to build. REST is a plain HTTPS POST, fits the project's zero-dependency ethos, and runs on 3.14 | Implemented |
| Read the API key only from `GEMINI_API_KEY` (git-ignored `.env`); never log or commit it | A live provider must not turn observability or version control into a secret leak | Implemented |
| Keep the `Retriever` standalone — no natural-language `query` in the public request yet | The app must not accept inputs it cannot responsibly process; NL entry waits for the Feature 5 privacy guard + intent parser | Active |
| Start hybrid ranking at 55/35/10 | Semantic relevance leads, original content score anchors behavior, session feedback personalizes modestly | Hypothesis to evaluate, not a final fact |
| Use MMR-style diversity | Prevent five near-duplicate results while retaining relevance | Planned |
| Never log raw sensitive prompts | Observability must not become a privacy leak | Planned |

## Free-tool research and implementation choices

The governing rule is **zero-cost-first, not provider-dependent**. External AI
improves language understanding and presentation, but the application must be
demonstrable without a paid API or network access.

| Need | Preferred free tool | Why it fits | State |
|---|---|---|---|
| Language/runtime | Python and standard library | Already used; sufficient for CSV, JSONL, hashing, and orchestration | Implemented |
| Runtime contracts | Pydantic 2 | Strict schemas and useful validation errors | Implemented |
| Unit/integration tests | pytest | Small, readable tests and fixtures | Implemented |
| Web UI | Streamlit | Fast Python-only interactive demo with session state | Declared; UI planned |
| Gemini access | Direct REST call with the Python standard library (`urllib`) | Zero third-party dependency; the `google-genai` SDK's `cryptography` dep has no Python-3.14 wheel. Embeddings are a simple HTTPS POST | Implemented for embeddings (stdlib REST); structured output/function calling will be revisited when a later feature needs them |
| Structured intent/voice | `gemini-3.5-flash-lite` | Current stable low-cost/free-tier candidate for structured tasks | Planned |
| Hosted embeddings | `gemini-embedding-2` reduced to 768 dimensions | Current stable semantic model; 768 is an officially recommended dimension | Implemented (committed vector cache + deterministic fake for tests + TF-IDF fallback) |
| Offline retrieval | **Pure-Python standard-library TF-IDF + cosine** (scikit-learn not used) | Deterministic, inspectable, no API/vector database, no new dependency, no Python-3.14 wheel risk; scikit-learn was overkill for 200 short docs | Implemented |
| Vector storage | In-memory Python dicts (sparse TF-IDF) + index fingerprint | 200 records do not justify NumPy arrays or a database; fingerprint is the cache seam for future embeddings | Implemented (TF-IDF); NumPy deferred to embeddings |
| Logs | Python JSON Lines | Appendable, diffable, no telemetry vendor | Planned |
| Diagram | Mermaid source; Kroki/Mermaid renderer for preview | Rubric-compliant text source with free rendering | Implemented |
| Versioning | Git and GitHub | Reproducible history and portfolio evidence | Implemented |

Official research references:

- [Gemini Flash Lite model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
- [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- [Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output)
- [Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings)
- [Gemini latest-model guidance](https://ai.google.dev/gemini-api/docs/latest-model)
- [Gemini deprecation schedule](https://ai.google.dev/gemini-api/docs/deprecations)
- [Gemini API terms](https://ai.google.dev/gemini-api/terms)
- [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/)
- [Google Gen AI SDK migration guide](https://ai.google.dev/gemini-api/docs/migrate)
- [scikit-learn text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)
- [pytest documentation](https://docs.pytest.org/)
- [Streamlit documentation](https://docs.streamlit.io/)
- [Mermaid documentation](https://mermaid.js.org/)
- [Kroki documentation](https://docs.kroki.io/kroki/)

Research verified on **2026-07-26**:

- `gemini-3.5-flash-lite` is currently a stable GA model with structured output
  and function calling. The free tier lists token input/output at no charge, but
  it is quota-limited and free-tier prompts/responses are marked as used to
  improve Google products.
- `gemini-embedding-2` currently supports 128–3072 dimensions; 768 is an
  officially recommended size. Index tracks separately rather than passing many
  tracks as one input, because multiple inputs produce one aggregated embedding.
- Standard free-tier text embeddings are currently listed at no charge, but
  batch embedding is not. Build a throttled indexing command and cache vectors.
- For Embedding 2, retrieval instructions belong in the text (for example,
  `task: search result | query: ...`); do not use the older `task_type` field.
- Structured JSON can still be semantically wrong even when it matches a JSON
  Schema. Pydantic and application-level validation remain mandatory.
- The model selects function names/arguments; our Python code executes only
  allowlisted tools after validating arguments. We will use a manually bounded
  orchestrator instead of open-ended automatic dispatch.
- Gemini 3.5 Flash-Lite currently ignores or deprecates temperature-style
  controls. Cadence’s voice should come from a voice card, system instruction,
  few-shot examples, and output evaluation—not a temperature setting.
- The official `google-genai` SDK is the future path for structured output and
  function calling, but for **embeddings** we call the REST endpoint directly with
  the standard library. The SDK pulls in `cryptography`, which has no Python-3.14
  wheel and fails to build without Rust/OpenSSL on this machine; the REST API
  needs no third-party package. Revisit the SDK (in a Python-3.12 env) only if a
  later feature needs its structured-output/function-calling helpers.

Current package snapshot from official package indexes on the research date:

| Package | Researched stable version | Minimum Python | Role |
|---|---:|---:|---|
| `google-genai` | 2.13.0 | 3.10 | Gemini SDK — **not installed**; embeddings use stdlib REST (its `cryptography` dep has no 3.14 wheel) |
| `scikit-learn` | 1.9.0 | 3.11 | TF-IDF and cosine similarity |
| `pydantic` | 2.13.4 | 3.9 | Contracts and validation |
| `pytest` | 9.1.1 | 3.10 | Tests and evaluation harness |
| `streamlit` | 1.60.0 | 3.10 | UI |

The current local virtual environment uses Python 3.14.5. Streamlit Community
Cloud currently defaults to Python 3.12, so Python 3.12 is the recommended
deployment baseline; verify all pins together when Feature 3 adds dependencies.

Model IDs, rate limits, prices, free-tier availability, SDK APIs, and data-use
terms are volatile. Re-check the official pages immediately before provider
implementation and again before deployment. Under the current unpaid-service
terms, submitted content and responses may be used to improve products and may
be reviewed by humans; Google says not to submit sensitive, confidential, or
personal information. Show a concise cloud-AI disclosure, keep secrets and
direct identifiers local, sanitize telemetry, and prefer the local path for
sensitive-looking input. The current terms also include age and regional
deployment constraints that must be reviewed before making the app public.

Streamlit Community Cloud is currently free and suitable for this educational
demo, but apps may hibernate, public apps may be indexed, and resource policies
can change. Kroki’s public renderer is a best-effort demonstration service, so
the committed Mermaid source remains authoritative.

## Completed engineering foundation

### `src/contracts.py`

Strict, immutable Pydantic models reject empty requests, unknown fields,
out-of-range values, booleans masquerading as numbers, NaN/infinity, malformed
catalog records, duplicate output IDs, and oversized result sets.

### `src/service.py`

`RecommendationService` validates the catalog once, protects an immutable copy,
adapts modern request names to the original scorer, and returns structured
results. Match strength is the raw score divided by the maximum score available
for the fields the user actually supplied. It is a fit indicator, not a
probability or statistical confidence.

### `src/main.py`

The CLI now exercises the shared request/service/result path rather than calling
the scoring function directly.

### Tests

The foundation suite proves compatibility with the original ranking, numeric
and schema validation, category normalization, request-relative match strength,
non-mutation, duplicate handling, structured reasons, and stable fallback
behavior.

### Architecture artifact

The canonical Mermaid source includes the retriever, agent, evaluator, tester,
human curator, data flow, fallback, privacy-safe logs, and provider boundary. It
must be revised from “target” to “implemented” as features land.

## Feature 2 specification: retrieval-ready catalog

### Purpose

The 20-song baseline is too sparse for useful retrieval or diversity. Feature 2
creates a trustworthy grounding corpus before adding embeddings. This is a data
and reliability feature, not yet RAG by itself.

### Required shape

- exactly 200 fictional tracks;
- exactly 20 genres with 10 tracks per genre;
- retain all original ten values for IDs 1–20;
- add `house`, `soul`, and `punk` to the original 17 genres;
- keep IDs unique and contiguous from 1 through 200;
- keep normalized title/artist pairs unique.

### Rich metadata

| Field | Meaning | Future use |
|---|---|---|
| `description` | Natural-language sound and use-case summary | Primary embedding text and grounded explanation |
| `tags` | Compact sonic/style terms | Keyword retrieval and evidence |
| `contexts` | Activities or situations | Requests such as “for studying” or “for a rainy drive” |
| `instruments` | Salient fictional instrumentation | Sound-specific retrieval and explanation |
| `instrumental` | Whether vocals are absent | Hard constraint/filter |
| `explicit` | Fictional content rating flag | Hard constraint/filter |
| `era` | Decade-style aesthetic such as `2020s` | Filter and retrieval clue |

List-valued CSV fields use `|` as the delimiter and become immutable tuples at
load time. Booleans must be exactly lowercase `true` or `false` in the CSV and
real booleans inside the service.

### Feature 2 acceptance gate

- [x] `data/songs.csv` contains 200 valid rows and the exact schema.
- [x] Every expected genre has exactly 10 rows.
- [x] IDs are exactly 1–200 and title/artist identities are unique.
- [x] Every row passes `CatalogTrack` validation.
- [x] Original fields for IDs 1–20 exactly match `data/legacy_songs.csv`.
- [x] Loader rejects schema drift, malformed list cells, and noncanonical booleans.
- [x] House, soul, and punk requests return exact-genre results through the real service.
- [x] The full CLI still runs.
- [x] Automated tests pass (`30 passed` on 2026-07-26).
- [x] Catalog data card explains provenance, generation, human review, and limitations.
- [ ] A human signs off on the representative sample and flagged outliers.

Human review should inspect at least one representative track per genre plus all
records flagged by automated outlier checks. Do not mark that review complete
until a person has actually signed off.

## Implementation roadmap

Features are listed by capability, in build order. Features 1–6 are implemented;
the remaining order is **7 evaluation harness → 8 session feedback + UI → 9
logging, evidence, and presentation** (measure quality before adding UI polish).

### Feature 1 — validated service foundation

Status: **Implemented**

Purpose: make one reliable seam that every future interface and AI component
must use.

Done when: invalid input cannot enter ranking unnoticed, results are typed and
immutable, CLI uses the service, and legacy ranking remains unchanged.

### Feature 2 — 200-track retrieval-ready catalog

Status: **Implemented; human content sign-off pending**

Purpose: give future retrieval enough balanced, descriptive, validated evidence.

Primary files: `data/songs.csv`, `data/legacy_songs.csv`, catalog generator,
`src/recommender.py`, `src/contracts.py`, catalog tests, catalog data card.

Technical gate: complete. Final content gate: a person completes the review
table and flagged-outlier check in `docs/CATALOG_DATA_CARD.md`.

### Features 3–4 — multi-source retrieval index and embeddings

Status: **Implemented** (Feature 3 TF-IDF, Feature 3b context guides, Feature 4 Gemini embeddings + hybrid)

Build one canonical retrieval document per track from its rich fields. Add
curated context guides as a second source with provenance. Implement local
TF-IDF first, then a provider-backed embedding strategy behind the same
interface. Cache the index using catalog content hash, embedding model ID,
dimension, and schema version.

Done when: a structured or normalized query retrieves relevant IDs, returns
source/provenance metadata, passes retrieval unit tests, and demonstrates a
before/after improvement over the original scorer for context-rich requests.

**Feature 3 acceptance gate (local TF-IDF retriever):**

- [x] `Retriever` interface + `TfidfRetriever` exist in `src/retrieval.py`, pure standard library.
- [x] One canonical retrieval document per track from `genre`, `mood`, `era`, `description`, `tags`, `contexts`, `instruments`.
- [x] Deterministic TF-IDF + cosine; results ordered by score with stable `id` tie-break.
- [x] Provenance contracts (`SourceType`, `RetrievalHit`, `RetrievalResult`) carry source type, source id, content hash, fields used, score, matched terms.
- [x] Hard filters (`instrumental_only`, `exclude_explicit`) run before ranking.
- [x] Index fingerprint covers both sources' content (catalog + guides) and the expansion settings (cache/rebuild seam for future embeddings).
- [x] No-signal (empty / out-of-vocabulary) queries return no hits rather than inventing relevance.
- [x] `tests/test_retrieval.py` passes.
- [x] `scripts/retrieval_demo.py` shows a before/after vs the numeric scorer for a context-rich phrase.
- [x] `recommend()` path and the public request contract are unchanged (no NL `query` field added).

**Feature 3b acceptance gate (curated context guides — second source):**

- [x] Curated guides live in `data/context_guides/*.md` (human-readable, one file per situation), loaded and validated as `ContextGuide`.
- [x] Guides are indexed as a second source (`SourceType.CONTEXT_GUIDE`) with content hashes.
- [x] Guide-driven **query expansion**: a matching guide folds its distinctive catalog-vocabulary terms into the track query; a dominance threshold drops weak matches.
- [x] Guides are **evidence, never recommendations** — they appear only in `guides_used`/`expanded_query_terms`, never in `hits`.
- [x] `GuideEvidence` records source type, source id, content hash, title, score, matched terms, and contributed expansion terms.
- [x] Before/after demonstrated: bridge queries (e.g. "music to concentrate") return **no** track-only hits but relevant hits once a guide expands the query.
- [x] `tests/test_context_guides.py` passes (incl. fingerprint coverage of guide content + expansion settings); full suite `70 passed` on 2026-07-29.

**Feature 4 acceptance gate (Gemini embeddings + hybrid ranking):**

- [x] `Embedder` interface with a deterministic offline `FakeEmbedder` and a lazy `GeminiEmbedder` (`gemini-embedding-2`, 768-d, truncate→renormalize).
- [x] `EmbeddingRetriever` (semantic) and `HybridRetriever` (semantic+lexical blend, configurable weights) behind the same `Retriever` interface.
- [x] Committed embedding cache keyed on catalog content + model + dimension; loader detects a stale/mismatched cache.
- [x] Honest fallback: missing/stale cache or a provider error → TF-IDF, `operating_mode=DEGRADED`; real semantic path is `GEMINI`.
- [x] Reproducibility: no third-party dependency (Gemini via stdlib REST); the full suite (`70 passed` on 2026-07-29) and the fallback run with **no key**; the demo reproduces offline from committed track + query caches.
- [x] Secret handling: key only from `GEMINI_API_KEY` (git-ignored `.env`), never logged or committed; `.env.example` provided.
- [x] `recommend()` path and the public request contract unchanged (still no NL `query` field).
- [x] Real embedding cache generated and committed (`data/embeddings/`, 200 track vectors + 5 example queries, `gemini-embedding-2` @ 768-d). Real paraphrase before/after recorded: `"tunes for cramming before an exam"` returns lofi study tracks at `sem 0.70+` with `lex 0.000` (zero shared words) — a match TF-IDF and guides both miss.

**Still pending here:** session feedback as a third source (Feature 8); and a
quantitative before/after retrieval metric in the evaluation harness (Feature 7).

### Feature 5 — input/privacy guard and structured intent

Status: **Implemented (deterministic); Gemini structured intent deferred**

Add honest natural-language `query` support. Validate size/type, detect likely
secrets and direct identifiers, identify prompt-injection patterns, and convert
ordinary music language into a typed intent. Try structured provider output
once, repair once, then use a deterministic rule parser.

Done when: valid language changes retrieval; unsafe/sensitive input follows a
safe path; unsupported constraints are not invented; raw private text is not
logged.

**Feature 5 acceptance gate:**

- [x] `InputGuard` (`src/guard.py`): size limit; PII/secret **redaction** (email, phone, card/SSN-like, key-like/high-entropy); prompt-injection stripping; conservative crisis → safe response. Deterministic, offline, auditable regexes.
- [x] Deterministic `IntentParser` (`src/intent.py`) reusing the scorer's genre/mood vocabulary: hard-filter cues (`no vocals` → instrumental-only, `clean` → exclude-explicit), genre/mood detection (incl. multi-word), one clarifying question when nothing is recognized.
- [x] `MusicCompanion` (`src/companion.py`): bounded actions `recommend / clarify / no_match / safe_response / degraded`; valid language drives the hybrid retriever; **sensitive input is routed to the local retriever and never reaches Gemini**.
- [x] Wired through the public CLI: `python -m src.main "clean chill beats for studying, no vocals"` returns instrumental study tracks (mode `gemini`); a PII query returns results at mode `local`; injection/vague → clarify; no-arg → the unchanged structured scorer.
- [x] Raw private text is never logged or sent to the provider (redacted before retrieval).
- [x] `tests/test_{guard,intent,companion}.py` pass; full suite `95 passed` on 2026-07-30, fully offline.
- [ ] Gemini structured-intent parser behind the same `IntentParser.parse` shape (deferred).

### Feature 6 — bounded agent, evaluator, and Cadence

Status: **Implemented** (deterministic + optional grounded Gemini voice)

Implement an explicit state machine with allowlisted actions. Hybrid rank
candidates using the initial 55% semantic, 35% original content, 10% session
feedback hypothesis; apply hard constraints and MMR diversity. Evaluate IDs,
duplicates, constraints, and evidence before Cadence renders a response.

Done when: recommend, clarify, no-match, safe-response, and degraded paths are
all reachable and tested; output claims are grounded; provider failure still
returns a useful labeled result.

**Feature 6 acceptance gate:**

- [x] Bounded agent: `MusicCompanion.respond()` chooses from an allowlist (`recommend / clarify / no_match / safe_response / degraded`) and emits a privacy-safe `AgentTrace` (categories, ids, decisions — never raw sensitive text).
- [x] MMR diversity (`src/ranking.py`): deterministic re-rank so the top-k isn't near-duplicates; relevance still leads.
- [x] Grounding evaluator (`src/evaluator.py`): validates ids/dupes/constraints/evidence, and that a generated message quotes only retrieved tracks.
- [x] Cadence voice (`src/voice.py` + `docs/CADENCE_VOICE.md`): deterministic baseline + optional grounded Gemini renderer that names no songs and passes the grounding check, degrading to the template voice on failure/no-key.
- [x] Provider text via stdlib REST (`src/generation.py`, `gemini-flash-lite-latest`); sensitive input reaches **neither** the retrieval nor the language provider.
- [x] Live voice validated: `python -m src.main "clean chill beats for studying, no vocals"` → a grounded Cadence framing + diverse set, `mode: gemini`, `voice: gemini`; a PII query → `mode: local`, `voice: template`.
- [x] `tests/test_{ranking,evaluator,voice}.py` + companion trace tests pass; full suite `113 passed` on 2026-07-30, fully offline.
- [x] `ai_interactions.md` drafted (SF8 agentic workflow + SF10 Strategy/Factory pattern).

The 55/35/10 hybrid with the numeric scorer + session feedback remains partial:
Feature 4's semantic+lexical blend is in place; the scorer/feedback weights await
structured prefs (this needs the intent parser to emit them) and Feature 8 memory.

### Feature 7 — evaluation harness (next)

Status: **Planned** — built before the UI so quality is measurable, not hidden
behind polish.

Create normal, edge, adversarial, missing-context, and provider-outage cases and
run them through the same public path. Produce a machine-readable report plus a
readable pass/fail summary; provider calls can be faked so the harness runs
offline and deterministically.

Suggested metrics:

- intent parse success;
- hard-constraint adherence;
- catalog faithfulness (grounding);
- retrieval recall@k;
- duplicate result rate;
- fallback success rate;
- latency;
- human-rated tone and helpfulness.

Done when: one command runs the evaluation set and prints pass/fail totals,
provider calls can be faked, and failures are actionable rather than hidden.

### Feature 8 — session feedback and web UI

Status: **Planned** (stdlib `http.server` UI — Streamlit's `pyarrow` chain will
not build on Python 3.14).

Create the working companion interface with chat input, recommendation cards,
reason/evidence display, operating-mode/voice badges, like/dislike controls, and
a visible “reset session memory” action. Session feedback is the third RAG
source: 👍 nudges later results toward a liked genre/mood, 👎 suppresses a track,
all session-only.

Done when: UI and CLI call the same `MusicCompanion`; feedback changes a later
ranking only within the session; a refresh/reset removes memory; three demo
inputs are reproducible.

### Feature 9 — privacy-safe logging, evidence, and presentation

Status: **Planned**

Log structured events (never API keys, raw PII prompts, full session memory, or
private reasoning; define a retention/deletion policy). Suggested event fields:

```text
request_id, timestamp, operating_mode, sanitized_intent,
retrieved_track_ids, retrieval_scores, validation_result,
selected_action, fallback_reason, latency_ms
```

Then finish the evidence: README setup + 2–3 end-to-end runs, a guardrail/fallback
example, the model card and AI-collaboration reflection, a synchronized Mermaid +
regenerated PNG, and the 5–7 minute presentation and portfolio entry.

Done when: a new reviewer can install, run, test, understand, and verify every
rubric claim from committed artifacts.

## Planned RAG design

### Sources

1. **Song catalog:** authoritative IDs and metadata. *Implemented* (TF-IDF index).
2. **Curated context guides:** human-written explanations of activity/mood
   relationships and responsible recommendation rules. *Implemented* as
   `data/context_guides/*.md`, used for guide-driven query expansion + evidence.
3. **Session feedback:** ephemeral likes, dislikes, and recent preferences.
   *Planned* (Feature 8).

Every retrieved item carries source type, source ID, content hash, and fields
used. Session feedback is context, not a permanent document.

### Retrieval stages

```text
validated intent
  → apply hard filters
  → create/search semantic and TF-IDF queries
  → fuse candidate scores
  → original content score
  → modest session-feedback adjustment
  → MMR diversity
  → evidence packet
```

Hard filters run before soft ranking for requirements such as instrumental-only
or clean-only. A high semantic score must never override a hard constraint.

### Why two retrieval strategies?

Provider embeddings understand paraphrases better; TF-IDF is deterministic,
free, inspectable, and available offline. Putting both behind one interface lets
the same agent and evaluator work in Gemini and local modes.

## Planned agent states

| State/action | When used | Required evidence |
|---|---|---|
| `clarify` | Request is too vague or conflicting | Missing/contradictory intent fields |
| `recommend` | Enough evidence and valid candidates exist | Catalog IDs, scores, reasons, provenance |
| `no_match` | Hard constraints leave no valid candidate | Filter counts and failed constraint summary |
| `safe_response` | Sensitive/high-risk/blocked input | Policy category, without retaining sensitive text |
| `degraded` | Provider unavailable or invalid after repair | Failure category and successful local result |

The action trace may record: normalized intent, chosen action, called tool name,
retrieved IDs, score summaries, validation decisions, and fallback reason. It
must not expose hidden chain-of-thought.

## Reliability and guardrail layers

1. **Schema guard:** correct fields, types, ranges, sizes, and enums.
2. **Privacy guard:** keep direct identifiers, secrets, and unnecessary personal
   information away from provider calls and logs.
3. **Injection guard:** user content cannot redefine system policy or invent
   tools; retrieved text is data, not instruction.
4. **Hard-constraint filter:** `instrumental`, `explicit`, and future constraints
   are enforced before ranking.
5. **Grounding evaluator:** every recommended ID and factual claim must trace to
   the evidence packet.
6. **Result guard:** reject duplicates, missing IDs, malformed schema, unsupported
   claims, and over-limit lists.
7. **Bounded repair:** one structured repair attempt.
8. **Local fallback:** deterministic useful response with explicit mode.
9. **Evaluation harness:** regression evidence across normal and hostile cases.
10. **Human review:** qualitative tone, relevance, bias, and surprising failures.

## Documentation and evidence map

| Artifact | Purpose |
|---|---|
| `README.md` | Reviewer-facing project summary, setup, architecture, commands, real outputs, guardrail example |
| `docs/PROJECT_HANDBOOK.md` | Durable project state, research, decisions, teaching guide, recovery context |
| `docs/CATALOG_DATA_CARD.md` | Catalog provenance, schema, validation, review, risks, limitations |
| `model_card.md` | Intended use, system behavior, bias, evaluation, AI collaboration, specialized comparison |
| `ai_interactions.md` | Structured agent workflow trace and human verification/corrections |
| `diagrams/architecture.mmd` | Rubric-required architecture and data flow source |
| Evaluation report/log | Quantitative system results and pass/fail gates |

README is the polished front door. The handbook is the engineering memory. The
model card is the responsible-AI reflection. Do not force one file to do all
three jobs.

## Known limitations and open decisions

Current limitations:

- The scoring families are subjective human groupings.
- Genre dominance reduces cross-genre discovery.
- Match strength is a request-relative score, not calibrated confidence.
- An unknown/all-miss genre can still yield zero-score stable ID order.
- Catalog records are fictional and manually/synthetically authored, not audio
  measurements from real tracks.
- Rich metadata can encode author bias and does not become objective because it
  is embedded.
- The target diagram contains components not yet implemented.
- `model_card.md` and older README experiments include historical 20-track
  observations that must be clearly labeled or refreshed.
- `ai_interactions.md` is drafted (SF8 agentic workflow + SF10 design pattern),
  pending the owner's personalization.
- TF-IDF retrieval is lexical (word forms); embeddings add meaning, but semantic
  quality needs the committed cache and retrieval similarity is not a calibrated
  probability.
- The input guard is a coarse, deterministic regex net: it can over-redact
  (safe) or miss novel PII/injection phrasings. The crisis detector is
  conservative and is **not** a substitute for real help.
- The intent parser is rule-based, so it understands only the genre/mood
  vocabulary and filter cues it knows; broader phrasing is handled by retrieval,
  not by structured parsing, until a Gemini structured-intent parser is added.
- Natural language reaches retrieval through `MusicCompanion`; the structured
  scorer path (`RecommendationService.recommend`) is intentionally separate.
- Semantic quality needs the real committed embedding cache (a rotated key runs
  `scripts/build_embeddings.py`). Without it, the hybrid honestly degrades to
  TF-IDF and labels the result `DEGRADED`. The `FakeEmbedder` exercises the
  plumbing only — it captures no real meaning.
- Embedding spaces are model- and dimension-specific; changing either requires
  re-embedding (the cache key enforces this).
- The context guides are AI-drafted fictional prose pending curator review; their
  wording encodes judgment about what music is "for" and can carry bias.
- Guide-driven query expansion is still lexical: a guide only helps when the
  listener's words overlap the guide, and it contributes only terms the catalog
  already uses.
- No provider adapter, `.env.example`, UI, AI logger, or evaluation report exists yet.

Open decisions:

- final product name and whether Cadence is final;
- exact Gemini model IDs and pinned SDK versions;
- whether the project owner accepts the current free-tier data-use terms;
- JSONL retention/deletion policy;
- metric pass thresholds;
- exact UI controls and session-reset language;
- final human evaluator(s), presentation date, and portfolio destination.

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-07-25 | Preserve the original scorer behind a validated service | Gives every interface a tested baseline and local fallback |
| 2026-07-25 | Do not add natural-language `query` prematurely | An accepted-but-ignored field would misrepresent system capability |
| 2026-07-25 | Report request-relative match strength, not “confidence” | The score is not a calibrated probability |
| 2026-07-25 | Use strict validation and immutable result contracts | Protect results as they move through the system |
| 2026-07-25 | Use a local fallback after bounded provider failure | Preserves utility, testability, and honest operation |
| 2026-07-25 | Make Cadence a warm fictional DJ with session-only memory | Adds life while avoiding deceptive identity and long-term profiling |
| 2026-07-26 | Expand and validate grounding data before RAG | Retrieval quality cannot exceed source-data quality |
| 2026-07-26 | Treat Feature 2 as data/reliability infrastructure, not claim it is RAG | Honest rubric mapping and architecture |
| 2026-07-26 | Recommend Python 3.12 for deployment while local development is 3.14.5 | Matches current Streamlit Community Cloud default and all planned package minimums |
| 2026-07-27 | Implement retrieval with pure-Python stdlib TF-IDF instead of scikit-learn | Zero new dependencies, inspectable math, no Python-3.14 wheel risk, trivial at 200 short docs |
| 2026-07-27 | Build a catalog-only retriever first; defer context guides to Feature 3b | One controlled, testable step; interface (`SourceType`, `Retriever`) already leaves room for the second source and the +2 multi-source bonus |
| 2026-07-27 | Keep the retriever standalone; add no natural-language `query` to the public request | Honest interface — NL entry belongs with the Feature 5 privacy guard + intent parser |
| 2026-07-27 | Return no hits for no-signal queries rather than zero-score filler | Retrieval must not claim relevance it cannot justify from matched terms |
| 2026-07-27 | Feature 3b: context guides act via query expansion + evidence, not as recommendable items | A guide is not a track; expanding the query with a guide's catalog terms improves retrieval while keeping tracks the only recommendable items, and yields a clean before/after |
| 2026-07-27 | Store context guides as one Markdown file per situation | Human-authored, diffable, curator-reviewable; the file stem is the guide id and the first heading is the title |
| 2026-07-29 | Feature 4: real Gemini embeddings, made reproducible by a committed vector cache + deterministic fake + TF-IDF fallback | Lets the project use a live AI without depending on it — tests and demos run with no key, results stay portable |
| 2026-07-29 | Hybrid ranking blends semantic + lexical only for now (configurable weights) | The numeric-scorer/feedback weights await Feature 5/8; dense+sparse is the honest blend the current inputs support |
| 2026-07-29 | Embeddings call the Gemini REST API via stdlib `urllib` (no `google-genai` SDK); key from a git-ignored `.env` only | The SDK's `cryptography` dep has no Python-3.14 wheel and fails to build without Rust/OpenSSL; REST needs no third-party package. Keeps secrets out of code, logs, and version control |
| 2026-07-30 | Add `certifi` (pure-Python CA bundle) with a system-trust fallback | The python.org 3.14 build has no usable system trust store, so `urllib` TLS verification fails; `certifi` fixes it with no compilation |
| 2026-07-30 | Embed one document per `embedContent` call (not sync batch); throttle + backoff + resumable incremental cache | `gemini-embedding-2` exposes single `embedContent` and `asyncBatchEmbedContent`, not sync `batchEmbedContents`; free-tier RPM returns 429, so the build throttles, backs off, and saves progress to resume |
| 2026-07-30 | Feature 5: natural language enters through a `MusicCompanion`, not a `query` field on `RecommendationRequest` | Keeps the trusted structured-scorer request pure and sets up Feature 6's bounded agent + Cadence voice; two validated entry points, one catalog |
| 2026-07-30 | Deterministic rule-based intent parser first (Gemini structured intent deferred behind the same interface) | Reproducible and key-free; the rule parser is the required fallback regardless — same local-first pattern as retrieval |
| 2026-07-30 | Guard redacts PII/secrets and routes sensitive queries to the local retriever | Sensitive text must never reach the provider or logs; redaction happens before retrieval, and a sensitive query is answered at operating mode `local` |
| 2026-07-30 | Feature 6 Cadence voice: deterministic renderer + optional grounded Gemini framing; the model never supplies song facts | Earns the specialized-behavior bonus while staying reproducible and hallucination-safe — the app lists the tracks; Cadence only frames them, and the framing is grounding-checked |
| 2026-07-30 | Sensitive queries reach neither the retrieval nor the language provider | Extends the Feature 5 guarantee: a redacted/local query uses the local retriever and the deterministic voice, never Gemini |
| 2026-07-30 | MMR diversity via a genre-family similarity proxy (no vectors) | Deterministic, cheap, reuses the scorer's families; keeps the top-k from being five near-duplicates without a heavy re-embedding step |

## Commands for the next developer

```bash
# Inspect state before touching files
git status --short
git log -5 --oneline --decorate

# Activate the existing environment
source .venv/bin/activate

# Run all tests
python -m pytest -q

# Run the application end to end
python -m src.main

# Regenerate the catalog after intentional seed changes
python scripts/generate_catalog.py
```

Do not commit automatically. Inspect `git diff --check`, `git diff --stat`, and
the substantive diff first. Preserve unrelated user changes.

## Teaching track and knowledge checks

Use this learning loop for every feature:

1. Explain the concept in plain English.
2. Point to where it lives in code.
3. Show one normal and one failure example.
4. Ask a short prediction question.
5. Have the learner explain the trade-off back in their own words.

Feature 2 check:

> Why validate metadata before creating embeddings?

Expected idea: embeddings make source data searchable; they do not make it
correct. Malformed, vague, or biased metadata would be retrieved and amplified,
so grounding quality starts with catalog quality.

RAG check:

> If the language model writes a beautiful explanation for a song ID that was
> not retrieved, should the output pass?

Expected answer: no. Eloquence cannot replace provenance; the grounding
evaluator must reject or repair the unsupported claim.

Fallback check:

> Why is a local fallback better than only showing “the API is down”?

Expected idea: the original scorer and local retrieval can still provide useful,
deterministic, testable results, while an operating-mode label keeps the system
honest.

## Next action

The full local + provider stack is built and tested offline (`115 passed`):
retrieval (TF-IDF + context guides), embeddings + hybrid ranking, the
natural-language guard + intent parser, and the bounded `MusicCompanion` agent —
MMR diversity, grounding evaluator, and Cadence's voice (deterministic + grounded
Gemini), wired through the CLI (Feature 6), then hardened by a Step-1 stabilization
pass. Next, in order:

1. **Feature 7 — evaluation harness.** Golden cases across
   normal/edge/adversarial/missing-context/outage through the same public path,
   with a pass/fail summary; provider calls faked so it runs offline. Built
   **before** the UI so quality is measured, not hidden behind polish.
2. **Feature 8 — session feedback + stdlib web UI.** A clickable companion (chat
   input, recommendation cards, evidence, operating-mode/voice badges,
   like/dislike, visible "reset memory"); feedback nudges a later ranking within
   the session. Stdlib `http.server` (Streamlit won't build on Python 3.14).
3. **Feature 9 — privacy-safe logging, evidence, and presentation.**
4. **Optional:** a Gemini structured-intent parser behind `IntentParser.parse`.
5. **Human sign-off** remains outstanding — the catalog representative/outlier
   review, the AI-drafted context guides, and the `ai_interactions.md`/voice-card
   drafts.
