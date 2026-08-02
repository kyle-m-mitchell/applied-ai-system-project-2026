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

Last updated: **2026-08-02**

| Item | Current state |
|---|---|
| Branch | `main`; Phase 3 is committed at `cc0d349`. The working tree contains the Phase 4 flagship UI, its backend policy/refinement seams, tests, and synchronized product documentation. Inspect with `git status --short`. |
| Feature | **Phase 4 — flagship Streamlit listening room implemented:** evidence cards, truthful source/network badges, backend-enforced local-only policy, typed Taste Console, guarded follow-ups, sticky privacy, reversible snapshots, set evolution, a session-only fit rating with no ranking effect, every bounded action state, and a request-local developer receipt. |
| Working tree | Uncommitted Phase 4 changes across `streamlit_app.py`, `ui/`, `src/{contracts,companion,refine,...}.py`, Streamlit configuration/dependencies, UI/refinement tests, README, product/handbook/evidence docs, and Mermaid sources. Inspect before editing. |
| Last verified regression check | `224 passed` from `.venv/bin/python -m pytest -q` on 2026-08-02 (fully offline; provider disabled); evaluation gate PASS with 100% hard-constraint compliance and genre satisfaction `0.863`. |
| Implemented | Original scorer and validated service; 200-track catalog; catalog + context-guide TF-IDF; cached/live Gemini embeddings and semantic/lexical hybrid with fallback; input/privacy guard and typed intent; bounded `MusicCompanion`; structured-preference fusion; relevance-floored MMR; grounding evaluator; deterministic voice plus optional model selection from approved microcopy; privacy-safe trace and JSONL event; evaluation report card; **Streamlit product UI with explicit execution/diversity policy, reversible refinements, and AppTest coverage**; offline tests, before/after demos, and authoritative Mermaid source. |
| In progress | Human review of catalog records, AI-drafted context-guide wording, and the `ai_interactions.md`/voice-card drafts |
| Not implemented yet | Provider structured intent; feedback-informed ranking; real licensed/public dataset ingestion; accounts/auth/rate limits/retention and deployment monitoring; correct pre-fusion ablation; human catalog/guide sign-off; final presentation/portfolio capture. |
| Next action | Run a connected-browser desktop/mobile/accessibility review, deploy a provider-disabled staging build, then implement provenance-first real-data ingestion before evaluated personalization. |

### Current implementation boundary

Two validated entry points share the one catalog:

```text
structured path:  RecommendationRequest → RecommendationService → scorer → RecommendationResult
language/UI path: text or typed patch → MusicCompanion → guard/intent → retrieval
                  → fusion/MMR → evaluators/voice → CompanionTurn + PipelineReceipt
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

| Rubric area | Points | Evidence | Status |
|---|---:|---|---|
| Original project and scope | 3 | README and model card describe the 20-track deterministic baseline and the applied-AI extension | Implemented |
| Substantial integrated AI feature | 3 | Natural-language requests change retrieval, structured/text fusion, diversity, and grounded presentation through `MusicCompanion` | Implemented in the CLI and Streamlit UI |
| Mermaid architecture | 3 | `diagrams/architecture.mmd` (authoritative source) | Implemented and synchronized with Phase 4; the historical PNG still needs regeneration |
| End-to-end demonstration | 3 | Streamlit listening room, CLI, and reproducible README runs | Implemented; final hosted-demo capture remains a delivery task |
| Reliability or guardrail | 3 | Contracts, privacy guard, hard filters, result/framing evaluators, local fallback, and report-card gate | Implemented |
| README and setup | 3 | Goals, installation, run/test commands, sample output, and UI walkthrough | Implemented; deployment URL is still pending |
| AI collaboration reflection | 3 | Model card and `ai_interactions.md` record prompting, debugging, useful/flawed suggestions, and limits | Implemented as a draft; owner personalization/final review pending |
| Multi-source RAG bonus | +2 | Song records + context guides, with query-expansion provenance and before/after evidence | Implemented with exactly two retrieval sources; session feedback is **not** a RAG source |
| Agentic workflow bonus | +2 | Bounded steps/tool calls and structured trace in `ai_interactions.md` | Implemented (bounded `MusicCompanion` with an `AgentTrace`; `ai_interactions.md` drafted) |
| Specialized behavior bonus | +2 | Cadence voice card, approved microcopy palette, and baseline comparison | Implemented (Gemini selects from a finite application-owned palette; deterministic baseline remains the fallback) |
| Evaluation harness bonus | +2 | `scripts/evaluate.py`, predefined cases, machine-readable results, and pass/fail summary | Implemented; current gate PASS |

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
- remember the current interpreted intent and reversible refinement history only
  within the current session;
- collect a session-only fit rating for human reflection. The rating currently
  does **not** change ranking, become retrieval context, or train a model.

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

## Implemented architecture and data flow

Canonical source: [`diagrams/architecture.mmd`](../diagrams/architecture.mmd)

Historical preview (predates Phase 4; do not use until regenerated):
[`assets/architecture.png`](../assets/architecture.png)

Implemented normal path:

```text
Listener through Streamlit or CLI
  → typed execution policy (local-only and diversity preset)
  → input/privacy guard
  → deterministic typed intent or validated refinement patch
  → catalog + context-guide retrieval with provenance
  → hard filters
  → semantic/lexical text ranking + structured-preference percentile fusion
  → relevance-floored MMR diversity
  → result evaluator
  → deterministic Cadence voice or optional Gemini framing
  → generated-framing guard
  → validated response + request-local receipt
  → recommendations, reasons, evidence, source/network badges, and mode
```

Implemented provider-free/failure paths:

```text
Local-only policy or sensitive input
  → no provider call
  → committed-cache/local TF-IDF retrieval + deterministic Cadence voice
  → the same evaluator and output contracts

Live embedding request fails
  → local TF-IDF fallback
  → explicit degraded mode, with attempted network use preserved in the receipt

Generated framing fails or violates the voice contract
  → discard it
  → deterministic template framing over the unchanged validated track set
```

Humans are part of the system:

- a curator reviews catalog records and context guides;
- a developer reviews model/tool decisions and privacy boundaries;
- a tester evaluates normal, edge, adversarial, and outage behavior;
- the listener can refine preferences, undo changes, reset the session, and rate
  a set. That rating is currently human feedback only; it does not feed ranking.

The Mermaid source now describes the **implemented Phase 4 architecture**. Keep
it synchronized whenever a runtime component, data source, policy, evaluator, or
human-review boundary changes.

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
preferences, retrieve candidates with local TF-IDF (or an exact committed query
vector), apply the same structured fusion/MMR when applicable, validate results,
and clearly label the source and operating mode. This keeps the product useful,
testable, free to demo, and honest during provider failures.

## Design decisions

| Decision | Why | Status |
|---|---|---|
| Build for the full 29 points | The project should teach several modern AI patterns and be portfolio-worthy | Active |
| Use Cadence as a fictional DJ companion | Personality makes interaction coherent without pretending the system is human | Working decision; final name still open |
| Keep product state session-only | Reversible intent/refinement state without accounts or a long-term profile; the fit rating does not personalize ranking | Active |
| Preserve the original scorer | It is explainable, deterministic, and provides a reliable fallback/baseline | Implemented |
| Put one service boundary around all interfaces | CLI, UI, agent, and tests must exercise the same application logic | Implemented |
| Add schemas before natural language | The system should not accept inputs it cannot honestly process | Implemented |
| Validate both input and output | Protecting only the request would still allow invalid or invented results downstream | Implemented with guards, Pydantic contracts, result evaluation, and framing evaluation |
| Bound provider retries, then fall back | Repeated retries add cost and unpredictability; a failed optional provider must not break the product | Implemented in the REST adapters; the UI uses an even stricter zero-retry interactive policy |
| Fall back locally after failure | The app should remain functional and disclose degraded mode | Implemented for embeddings/retrieval and model-selected voice |
| Use specialized prompting, not call it fine-tuning | The provider selects one exact application-owned Cadence line; a finite output space is honest, cheap, and measurable | Implemented for Cadence's optional microcopy selection |
| Use in-memory indexes for 200 songs | A vector database adds complexity without useful scale benefits | Implemented with in-memory sparse/dense data and committed JSON embedding caches |
| Use pure-Python stdlib TF-IDF instead of scikit-learn | Zero new dependencies, fully inspectable math, no wheel/compat risk on Python 3.14.5, and trivially fast at 200 short docs; scikit-learn/NumPy were not installed | Implemented |
| Build the catalog-only retriever first; defer context guides to Feature 3b | Keeps one controlled, fully testable step; the `Retriever` interface and `SourceType` already leave room for a second source | Implemented (3b done) |
| Use context guides as query expansion + evidence, not as recommendable items | A guide is not a track; expanding the query with a guide's catalog-vocabulary terms improves track retrieval while keeping "tracks are the only recommendable items" intact | Implemented |
| Gate guide expansion with a dominance threshold (≥ 0.5 × top guide score) | Drops weak, spurious guide matches that would otherwise inject off-topic expansion terms | Implemented |
| Use real Gemini embeddings but keep the system reproducible via a committed cache + deterministic fake + TF-IDF fallback | Gemini is the tool that *builds* a reproducible artifact; the committed vectors and offline fallback are what make it portable and testable with no key | Implemented |
| Blend the text retrievers at 60% semantic / 40% lexical | Dense semantics catches paraphrases while sparse TF-IDF preserves inspectable word overlap | Implemented; falls back honestly when semantic evidence is unavailable |
| Call the Gemini REST endpoint with the standard library (`urllib`) for this small adapter | Minimal dependencies and an inspectable HTTPS seam. Current `google-genai` supports Python 3.14 and remains a viable future SDK choice; an early local install failure is not a current incompatibility claim. | Implemented |
| Read the API key only from `GEMINI_API_KEY` (git-ignored `.env`); never log or commit it | A live provider must not turn observability or version control into a secret leak | Implemented |
| Keep the structured scorer and natural-language companion as separate validated entry points | A typed `RecommendationRequest` serves the deterministic baseline; free text enters only through the guard and intent parser | Implemented |
| Fuse text and structured rankings as unit-free percentiles | Cosine/text scores and structured scores use different units, so raw-number addition would let one scale dominate | Implemented at calibrated 0.4 text / 0.6 structured when structured intent exists; text-only order is unchanged |
| Use relevance-floored MMR diversity | Prevent near-duplicate sets without admitting weak, off-topic tracks | Implemented with Focused, Balanced, and Exploratory presets and a fixed relevance floor |
| Treat the UI fit rating as evaluation-only for now | A feedback widget is not evidence that personalization works; ranking changes need a defined signal, consent boundary, and evaluation first | Active; no feedback-informed ranking exists |
| Never log raw or sanitized prompt text | Observability must not become a privacy leak | Implemented with allowlisted `CompanionEvent` JSONL receipts; the UI defaults to `NullEventSink` |

## Free-tool research and implementation choices

The governing rule is **zero-cost-first, not provider-dependent**. External AI
improves language understanding and presentation, but the application must be
demonstrable without a paid API or network access.

| Need | Preferred free tool | Why it fits | State |
|---|---|---|---|
| Language/runtime | Python and standard library | Already used; sufficient for CSV, JSONL, hashing, and orchestration | Implemented |
| Runtime contracts | Pydantic 2 | Strict schemas and useful validation errors | Implemented |
| Unit/integration tests | pytest | Small, readable tests and fixtures | Implemented |
| Web UI | Streamlit 1.60 + `AppTest` | Fast Python-only interactive product with session state and testable UI flows | Implemented in `streamlit_app.py` and `ui/` |
| Gemini access | Direct REST calls with the Python standard library (`urllib`) | Keeps the small adapter inspectable and provider use optional; the official SDK is also viable | Implemented for query embeddings and approved-line selection; provider structured intent is not implemented |
| Structured intent and voice | Deterministic typed parser + `gemini-flash-lite-latest` optional bounded selector | Rules keep intent reproducible; Gemini selects only application-owned microcopy, never track facts or ranking | Implemented; non-allowlisted output falls back to the template |
| Hosted embeddings | `gemini-embedding-2` reduced to 768 dimensions | Current stable semantic model; 768 is an officially recommended dimension | Implemented (committed vector cache + deterministic fake for tests + TF-IDF fallback) |
| Offline retrieval | **Pure-Python standard-library TF-IDF + cosine** (scikit-learn not used) | Deterministic, inspectable, no API/vector database, no new dependency, no Python-3.14 wheel risk; scikit-learn was overkill for 200 short docs | Implemented |
| Vector storage | In-memory Python structures + fingerprinted JSON caches | 200 records do not justify a vector database; committed vectors make the semantic demo reproducible | Implemented for TF-IDF and dense embeddings, without NumPy |
| Logs | Python JSON Lines + request-local typed receipts | Appendable, inspectable, and privacy-safe without a telemetry vendor or a shared-log read in the UI | Implemented as opt-in JSONL; UI persistence is off by default |
| System evaluation | Python + JSON/Markdown fixtures | Deterministic cases, outage doubles, and a nonzero exit code make regressions actionable | Implemented in `scripts/evaluate.py` |
| Diagram | Mermaid source; Kroki/Mermaid renderer for preview | Rubric-compliant text source with free rendering | Implemented |
| Versioning | Git and GitHub | Reproducible history and portfolio evidence | Implemented |

Official research references (model/provider facts are volatile; this list is a
dated research record, not proof that an alias is still available):

- [Gemini model documentation](https://ai.google.dev/gemini-api/docs/models)
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

Research verified on **2026-07-26**; implementation choices made later are
called out explicitly:

- The researched Flash-Lite option supported structured output and function
  calling, with quota/data-use trade-offs on the free tier. That model-name
  assumption was later rejected during integration: the implemented optional
  text adapter uses `gemini-flash-lite-latest` and must be reverified before
  deployment. Provider structured intent/function calling is not implemented.
- `gemini-embedding-2` currently supports 128–3072 dimensions; 768 is an
  officially recommended size. Index tracks separately rather than passing many
  tracks as one input, because multiple inputs produce one aggregated embedding.
- Standard free-tier text embeddings are currently listed at no charge, but
  batch embedding is not. Build a throttled indexing command and cache vectors.
- For Embedding 2, retrieval instructions belong in the text (for example,
  `task: search result | query: ...`); do not use the older `task_type` field.
- Structured JSON can still be semantically wrong even when it matches a JSON
  Schema. Pydantic and application-level validation remain mandatory.
- Provider function calling was researched but not used. `MusicCompanion` is a
  manually bounded Python workflow, so no model can invent or dispatch a tool.
- Cadence’s specialized voice comes from a voice card, system instruction,
  approved microcopy, examples, and exact output validation—not a temperature setting.
- Both implemented Gemini adapters call REST directly with the standard library.
  An early local SDK install failed, but current `google-genai` and
  `cryptography` releases provide Python 3.14-compatible packages. The SDK is a
  viable future choice when its higher-level response handling adds enough value.

Current package snapshot from official package indexes on the research date:

| Package | Researched stable version | Minimum Python | Role |
|---|---:|---:|---|
| `google-genai` | 2.13.0 | 3.10 | Researched alternative — **not installed**; both provider adapters use stdlib REST |
| `scikit-learn` | 1.9.0 | 3.11 | Researched alternative — **not installed**; retrieval uses stdlib math |
| `pydantic` | 2.13.4 | 3.9 | Contracts and validation |
| `pytest` | 9.1.1 | 3.10 | Tests and evaluation harness |
| `streamlit` | 1.60.0 | 3.10 | UI |

The current local virtual environment uses Python 3.14.5. Streamlit Community
Cloud currently defaults to Python 3.12, so Python 3.12 is the recommended
deployment baseline; verify all pins together before deployment and whenever a
provider model or dependency changes.

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

The canonical Mermaid source includes the implemented Streamlit/CLI entry
points, execution policy, guard, retriever, fusion/MMR ranker, evaluators,
Cadence voice, tester, human curator, fallback, privacy-safe events, request-
local receipt, and provider boundary. It is synchronized with Phase 4.

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

Features are listed by capability in their historical build order. The runtime
work through Phase 4 is implemented: Features 1–6, the Feature 7 evaluation
harness, the Feature 8 Streamlit product, and Feature 9's observability
foundation. Remaining work is human sign-off, browser/deployment verification,
real-data ingestion, evaluated personalization, and final presentation capture.

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
versioned AI-drafted context guides as a second source with provenance. Implement local
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

**Feature 3b acceptance gate (versioned context guides — second source):**

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

**Deliberately outside this feature:** the UI's session-only fit rating is not a
retrieval source and does not change candidate scores. The evaluation harness is
implemented, but a correct pre-fusion retrieval ablation is still pending; do
not reconstruct one from final MMR-reordered cards.

### Feature 5 — input/privacy guard and structured intent

Status: **Implemented (deterministic); Gemini structured intent deferred**

Add honest natural-language `query` support. Validate size/type, detect likely
secrets and direct identifiers, identify prompt-injection patterns, and convert
ordinary music language into a typed intent with a deterministic rule parser.
The parser interface leaves room for a future provider implementation, but no
provider currently decides structured intent.

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

Status: **Implemented** (deterministic + optional Gemini-selected approved microcopy)

Implement an explicit state machine with allowlisted actions. Hybrid rank
candidates in two stages: the text leg blends semantic/lexical retrieval at
60/40, then structured preferences (when present) are percentile-fused with the
text ordering at 60/40 structured/text. Apply hard constraints and relevance-
floored MMR diversity. Evaluate IDs, duplicates, constraints, and evidence
before Cadence renders a response.

Done when: recommend, clarify, no-match, safe-response, and degraded paths are
all reachable and tested; output claims are grounded; provider failure still
returns a useful labeled result.

**Feature 6 acceptance gate:**

- [x] Bounded agent: `MusicCompanion.respond()` chooses from an allowlist (`recommend / clarify / no_match / safe_response / degraded`) and emits a privacy-safe `AgentTrace` (categories, ids, decisions — never raw sensitive text).
- [x] MMR diversity (`src/ranking.py`): deterministic re-rank so the top-k isn't near-duplicates; relevance still leads.
- [x] Grounding evaluator (`src/evaluator.py`): validates ids/dupes/constraints/evidence, empties rejected payloads, and accepts model framing only when it exactly matches approved fact-free microcopy.
- [x] Cadence voice (`src/voice.py` + `docs/CADENCE_VOICE.md`): deterministic baseline + optional Gemini selection from an exact application-owned palette, degrading to the template on deviation/failure/no-key.
- [x] Provider text via stdlib REST (`src/generation.py`, `gemini-flash-lite-latest`); sensitive input reaches **neither** the retrieval nor the language provider.
- [x] Live voice path validated: an ordinary request may use one allowlisted Cadence line with `mode: gemini`, `voice: generated`; a PII query remains `mode: local`, `voice: template`.
- [x] `tests/test_{ranking,evaluator,voice}.py` + companion trace tests pass; current full suite `224 passed` (2026-08-02, including evaluation, observability, cache failure paths, CLI provider policy, structured refinement, and Streamlit AppTest), fully offline.
- [x] `ai_interactions.md` drafted (SF8 agentic workflow + SF10 Strategy/Factory pattern).

The earlier 55/35/10 semantic/scorer/feedback idea was **not** implemented. It
mixed unlike score units and assumed a feedback signal that the product does not
yet have. The implemented rank-percentile fusion makes the text and structured
orders comparable; feedback-informed ranking remains future work.

### Feature 7 — evaluation harness

Status: **Implemented**

Purpose: measure whole-system behavior through the same public
`MusicCompanion` path used by the product. This catches failures that isolated
unit tests cannot, such as a guard working correctly while a later fallback is
labeled incorrectly.

Implemented behavior:

- `eval/cases.json` defines 15 required, planned, development, and holdout cases;
- four deterministic scenarios exercise local TF-IDF, fake hybrid plumbing,
  embedding outage, and generation outage (15 cases × 4 = 60 runs);
- `scripts/evaluate.py` writes transient JSON and Markdown reports, prints a
  summary, and exits nonzero when the quality gate fails;
- `eval/results/baseline.json` is the accepted, committed comparison point;
- report rows store case IDs, categories, actions, track IDs, scores, and
  outcomes—never raw or sanitized query text;
- the gate checks required cases, 100% hard-constraint adherence, catalog
  faithfulness, embedding/generation fallback, and a 0.75 genre-satisfaction
  floor. The accepted result passes at `0.863` genre satisfaction.

Run it with:

```bash
python scripts/evaluate.py
```

The harness is a regression gate, not proof that listeners love the product.
Human tone/relevance review and a true pre-fusion retrieval ablation remain
separate work.

### Feature 8 — flagship Streamlit UI and session controls

Status: **Implemented in Phase 4; connected-browser and hosted staging review
pending**

Purpose: turn the working pipeline into a real product flow without copying AI
logic into presentation code. `streamlit_app.py` and `ui/` call the same
`MusicCompanion` boundary as the CLI and evaluator.

Implemented behavior:

- evidence-first track cards and interpreted-intent/context-guide evidence;
- honest local, cached semantic, live semantic, degraded, network-use, and voice
  badges;
- backend-enforced `ExecutionPolicy(force_local, diversity)`, including
  Focused/Balanced/Exploratory MMR presets;
- a transactional Taste Console, quick moves, and guarded free-text follow-ups;
- sticky sensitive routing, immutable snapshots, set-evolution summaries, exact
  undo, and session reset;
- graceful recommend, clarify, no-match, safe-response, and outage states;
- a developer view backed by the current turn's privacy-safe `PipelineReceipt`,
  not a shared log; and
- Streamlit `AppTest` coverage for normal, privacy, fallback, refinement, undo,
  state, and developer flows.

The “Did this set fit?” control stores only a **session-scoped rating** in
Streamlit state and displays an acknowledgment. It does **not** alter later
ranking, suppress tracks, expand a query, create a third RAG source, or train a
model. Those behaviors require a separately designed and evaluated
personalization feature.

### Feature 9 — privacy-safe logging, evidence, and presentation

Status: **Runtime observability implemented; final human review, deployment
evidence, presentation, and portfolio capture pending**

The logging principle is “a receipt, not a diary.” `CompanionEvent` contains an
allowlist of decisions, IDs, component scores, modes, source/network facts,
latency, and fingerprints. It excludes raw/sanitized query text, prompts, API
keys, persistent listener identity, free-form memory, and hidden reasoning.

```text
request_id, timestamp, guard_category, allowlisted intent facets,
candidate_ids, final_ids, score components, selected action,
operating/embedding/voice source, network_used, fallback_reason,
latency_ms, index fingerprint, config fingerprint
```

`JsonlEventSink` provides opt-in append-only persistence; the CLI enables it with
`--log`. `NullEventSink` is the default, and the Streamlit UI intentionally does
not write or read a shared event file. The UI receives a request-local
`PipelineReceipt` instead. Logging is best-effort: an I/O failure cannot change
the recommendation.

The README, model card, AI-interaction draft, Mermaid source, test suite,
and evaluation baseline exist. Still required before final submission: owner
review/personalization, catalog/context-guide human sign-off, connected-browser
and staging evidence, a retention/deletion policy if persistent logs will be
used, and the 5–7 minute presentation/portfolio entry.

## Implemented RAG design

### Sources

1. **Song catalog:** the authoritative 200 track IDs and validated metadata. One
   canonical searchable document is built per track for TF-IDF and embedding
   retrieval. Only catalog tracks can become recommendations.
2. **Versioned context guides:** AI-drafted Markdown explanations of activity/mood
   relationships intended for human curation. They currently remain AI-drafted
   pending sign-off. They live in `data/context_guides/*.md` and are searched as
   a second source. Strong matches contribute controlled catalog vocabulary to
   the track query and appear as provenance; a guide can never be returned as a
   song.

Every retrieved item carries source type, source ID, content hash, and fields
used. Guide evidence separately records the guide ID/hash, match score, matched
terms, and expansion terms.

The Streamlit fit rating is **not a third source**. It is session-only UI state
that currently changes nothing downstream. The original numeric scorer is also
not a document source; its structured feature comparisons form a ranking leg
after text retrieval has produced candidates.

### Retrieval stages

```text
guarded text → deterministic typed intent
  → choose standard or provider-free retriever from execution/privacy policy
  → apply instrumental/clean hard filters
  → retrieve strong context-guide evidence and expand with controlled terms
  → search track TF-IDF plus cached/live semantic embeddings
  → blend semantic/lexical text scores (60/40)
  → percentile-fuse structured preferences when present (40/60 text/structured)
  → Focused/Balanced/Exploratory MMR with a fixed relevance floor
  → evaluate IDs, constraints, duplicates, count, and evidence
  → render deterministic track cards + optional Gemini-selected approved line
```

Hard filters run before soft ranking for requirements such as instrumental-only
or clean-only. A high semantic score must never override a hard constraint.

### Why two retrieval strategies?

Provider embeddings understand paraphrases better; TF-IDF is deterministic,
free, inspectable, and available offline. Putting both behind one interface lets
the same agent and evaluator work in Gemini and local modes.

## Implemented bounded-agent states

| State/action | When used | Required evidence |
|---|---|---|
| `clarify` | Initial request is too vague, or a follow-up has no supported musical change | Typed intent/guard outcome without retrieval |
| `recommend` | Enough evidence and valid candidates exist | Catalog IDs, scores, reasons, provenance |
| `no_match` | Retrieval/MMR yields no candidate, or the result evaluator rejects the set | Empty/rejected retrieval result, applied filters/provenance, and evaluator/fallback category |
| `safe_response` | High-risk/crisis language requires a fixed non-clinical response | Guard category; no retrieval or provider call |
| `degraded` | A live semantic attempt fails but local retrieval still succeeds | Attempt/source fact, fallback reason/mode, and validated local result |

Sensitive-but-ordinary music requests are not automatically `safe_response`.
They are redacted, locked to provider-free execution, and can still produce a
normal recommendation. That distinction preserves usefulness without sending
direct identifiers to Gemini.

The `AgentTrace` records only the guard category, allowlisted intent facets,
retrieved IDs, diversity/evaluation result, chosen action, voice source, network
use, and fallback reason. `PipelineReceipt` adds candidate/final IDs, timings,
source, policy, and fingerprints. Neither contains query text, provider prompts,
or hidden chain-of-thought.

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
7. **Generated-framing guard:** reject long/multi-sentence text, names, quotes,
   links/markup, persona violations, and unsafe role claims; use the template.
8. **Bounded provider retry:** adapters have a small retry budget; the interactive
   UI config uses zero retries, then the local/template fallback takes over.
9. **Execution-policy guard:** local-only and sticky sensitive routing are
   enforced at the backend, not merely shown as badges.
10. **Local fallback:** deterministic useful response with explicit source/mode.
11. **Privacy-safe observability:** request receipts and opt-in events use an
    allowlist and never contain query/prompt text.
12. **Evaluation harness:** reproducible regression evidence across normal,
    privacy, adversarial, and outage cases.
13. **Human review:** qualitative tone, relevance, bias, accessibility, and
    surprising failures. Automated tests cannot replace this layer.

## Documentation and evidence map

| Artifact | Purpose |
|---|---|
| `README.md` | Reviewer-facing project summary, setup, architecture, commands, real outputs, guardrail example |
| `docs/PROJECT_HANDBOOK.md` | Durable project state, research, decisions, teaching guide, recovery context |
| `docs/CATALOG_DATA_CARD.md` | Catalog provenance, schema, validation, review, risks, limitations |
| `model_card.md` | Intended use, system behavior, bias, evaluation, AI collaboration, specialized comparison |
| `ai_interactions.md` | Structured agent workflow trace and human verification/corrections |
| `diagrams/architecture.mmd` | Rubric-required architecture and data flow source |
| `eval/cases.json` + `eval/results/baseline.json` | Quantitative system cases, accepted results, and pass/fail gates |

README is the polished front door. The handbook is the engineering memory. The
model card is the responsible-AI reflection. Do not force one file to do all
three jobs.

## Known limitations and open decisions

Current limitations:

- The 200 catalog records are fictional and manually/synthetically authored.
  There is no licensed real catalog, audio analysis, preview playback, live
  availability, popularity, or collaborative-listener signal yet.
- Genre/mood families and rich descriptions are subjective human groupings.
  Embedding authored metadata makes it searchable; it does not make it factual,
  neutral, or culturally complete.
- The original scorer deliberately makes genre dominant, which protects an
  explicit genre request but can reduce cross-genre discovery. Its match
  strength is request-relative fit, not calibrated probability or confidence.
- TF-IDF recognizes word overlap. Embeddings add paraphrase sensitivity, but
  cosine similarity is still not calibrated confidence, and semantic quality
  depends on the model-specific committed cache or an allowed live query call.
  `FakeEmbedder` tests plumbing only and has no semantic meaning.
- Embedding spaces are model- and dimension-specific; changing either requires
  rebuilding the cache. Fingerprints detect a mismatch but cannot repair it.
- Context-guide expansion is lexical and curator-dependent. The guides are
  AI-drafted fictional prose pending human review, and their assumptions about
  what music is “for” can introduce bias.
- The intent parser is deterministic and vocabulary-bounded. Retrieval can
  understand broader text, but only recognized genres, moods, filters, and
  numeric cues become structured preferences. Provider structured intent is not
  implemented.
- The input guard is a coarse regex safety net. It can over-redact benign text
  or miss novel PII/injection phrasing; its crisis route is conservative and is
  **not** a substitute for professional or emergency help.
- MMR uses genre/mood-family similarity as an explainable proxy for diversity.
  It does not know novelty, popularity, familiarity, or the listener's history.
- The UI's fit rating is session-only and currently has **zero ranking effect**.
  There is no learned preference profile, dislike suppression, collaborative
  filtering, account, or durable memory.
- Streamlit session state is per browser session, not an identity system. The UI
  deliberately clears query parameters and does not offer prompt-bearing share
  URLs.
- Persistent JSONL events are opt-in and privacy-minimized, while the UI uses
  `NullEventSink`. Retention, rotation, deletion automation, access control, and
  production monitoring are not implemented.
- The evaluation matrix has 15 authored cases across four offline scenarios.
  It is a useful regression gate, not representative user research. Fake hybrid
  validates wiring, not semantic quality; a correct pre-fusion retrieval
  ablation and human tone/helpfulness study remain pending.
- Headless `AppTest` proves flows and state behavior, not responsive visual
  quality, keyboard/screen-reader usability, reduced motion, contrast, or hosted
  configuration. Connected-browser accessibility review and staging are pending.
- `ai_interactions.md`, the voice card, context guides, and catalog sample still
  need the owner's human review/sign-off before submission claims are final.

Open decisions:

- final product name and whether Cadence is final;
- exact stable Gemini model IDs/pinning for a deployed release (the optional
  text adapter currently uses the rolling `gemini-flash-lite-latest` alias);
- whether the project owner accepts the current free-tier data-use terms;
- source/license choice and field-provenance rules for the real-data catalog;
- JSONL retention/deletion/access policy if persistent events are enabled;
- metric gates for real-data retrieval and any future personalization;
- what feedback signal may affect ranking, how consent/reset works, and how its
  value will be tested before release;
- deployment host, accessibility acceptance criteria, and monitoring boundary;
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
| 2026-07-27 | Store context guides as one Markdown file per situation | AI-drafted, diffable, and pending curator review; the file stem is the guide id and the first heading is the title |
| 2026-07-29 | Feature 4: real Gemini embeddings, made reproducible by a committed vector cache + deterministic fake + TF-IDF fallback | Lets the project use a live AI without depending on it — tests and demos run with no key, results stay portable |
| 2026-07-29 | Hybrid ranking blends semantic + lexical only for now (configurable weights) | The numeric-scorer/feedback weights await Feature 5/8; dense+sparse is the honest blend the current inputs support |
| 2026-07-29 | Embeddings call the Gemini REST API via stdlib `urllib`; key from a git-ignored `.env` only | The minimal REST seam was easiest to inspect after an early install failure. Current `google-genai` supports Python 3.14 and is a valid future option. Either path must keep secrets out of code, logs, and version control. |
| 2026-07-30 | Add `certifi` (pure-Python CA bundle) with a system-trust fallback | The python.org 3.14 build has no usable system trust store, so `urllib` TLS verification fails; `certifi` fixes it with no compilation |
| 2026-07-30 | Embed one document per `embedContent` call (not sync batch); throttle + backoff + resumable incremental cache | `gemini-embedding-2` exposes single `embedContent` and `asyncBatchEmbedContent`, not sync `batchEmbedContents`; free-tier RPM returns 429, so the build throttles, backs off, and saves progress to resume |
| 2026-07-30 | Feature 5: natural language enters through a `MusicCompanion`, not a `query` field on `RecommendationRequest` | Keeps the trusted structured-scorer request pure and sets up Feature 6's bounded agent + Cadence voice; two validated entry points, one catalog |
| 2026-07-30 | Deterministic rule-based intent parser first (Gemini structured intent deferred behind the same interface) | Reproducible and key-free; the rule parser is the required fallback regardless — same local-first pattern as retrieval |
| 2026-07-30 | Guard redacts PII/secrets and routes sensitive queries to the local retriever | Sensitive text must never reach the provider or logs; redaction happens before retrieval, and a sensitive query is answered at operating mode `local` |
| 2026-07-30 | Feature 6 Cadence voice: deterministic renderer + optional Gemini selection from approved microcopy; the model never supplies song facts | Earns the specialized-behavior bonus while staying reproducible and hallucination-safe; exact membership makes the publishable output finite and auditable |
| 2026-07-30 | Sensitive queries reach neither the retrieval nor the language provider | Extends the Feature 5 guarantee: a redacted/local query uses the local retriever and the deterministic voice, never Gemini |
| 2026-07-30 | MMR diversity via a genre-family similarity proxy (no vectors) | Deterministic, cheap, reuses the scorer's families; keeps the top-k from being five near-duplicates without a heavy re-embedding step |
| 2026-08-01 | Gate system quality before adding the product UI | The offline scenario matrix makes hard constraints, grounding, fallback, and genre satisfaction measurable instead of hiding regressions behind polish |
| 2026-08-01 | Fuse text and structured rankings as percentiles at 0.4/0.6 | The two legs have unlike raw units; percentile fusion makes their ordering comparable and achieved 0.863 genre satisfaction without regressing required cases |
| 2026-08-01 | Observability is an allowlisted receipt, not a prompt diary | `PipelineReceipt` and optional `CompanionEvent` preserve decisions/provenance for debugging while excluding query text and persistent identity |
| 2026-08-02 | Keep Phase 4 UI thin and enforce privacy/diversity through typed backend policy | Streamlit controls must re-enter the same guard → retrieve → fuse → MMR → evaluate pipeline; a badge alone cannot block provider use |
| 2026-08-02 | Treat the session fit rating as evaluation-only | Showing a feedback control does not prove personalization; it must not alter ranking until a signal, consent model, and evaluation gate are designed |

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
python -m src.main "clean chill beats for studying, no vocals"

# Run the reproducible whole-system evaluation gate
python scripts/evaluate.py

# Run the Phase 4 listening room
streamlit run streamlit_app.py

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

Phase 4 is built and tested offline (`224 passed`). The working product now
includes guarded natural language, two-source RAG, cached/live Gemini embeddings,
semantic/lexical plus structured-preference fusion, relevance-floored MMR,
grounding and framing evaluation, specialized Cadence voice, local fallbacks,
privacy-safe receipts/events, the evaluation report card, and the Streamlit
listening room. The rating control is present, but personalization is not.

Proceed in this order:

1. **Verify the product in a real browser.** Review desktop/mobile layout,
   keyboard navigation, screen-reader names, focus, contrast, reduced motion, and
   all bounded states. Record defects instead of treating headless AppTest as
   visual/accessibility proof.
2. **Deploy a provider-disabled staging build.** Smoke-test setup, startup
   failure, local-only behavior, privacy copy, and resource limits without
   exposing an API key. Decide retention/monitoring policy before enabling logs
   or provider access in a public deployment.
3. **Build provenance-first real-data ingestion.** Choose a legally usable
   source, retain field-level source/license/unit metadata, distinguish missing
   from false/zero, quarantine invalid rows, rebuild fingerprints/embeddings, and
   compare retrieval quality against the fictional baseline. Do not fabricate
   fields merely to satisfy the current schema.
4. **Only then design evaluated personalization.** Define what a rating means,
   how it decays/resets, what ranking component it changes, how to prevent one
   click from overpowering explicit constraints, and which offline/human metrics
   prove improvement. Until that work lands, feedback remains session-only UI
   data with no ranking effect.
5. **Close the submission evidence.** Complete catalog/context-guide and draft
   document sign-offs, update any artifacts changed by the real-data decision,
   capture reproducible UI/output evidence, and prepare the 5–7 minute
   presentation and portfolio reflection.
