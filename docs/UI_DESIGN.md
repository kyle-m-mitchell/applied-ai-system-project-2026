# Cadence UI: Product Design and Teaching Guide

This document records what Phase 4 actually built, why the original proposal
was revised, and how to reason about the UI as both a listener and an engineer.
It is recovery context for a future developer or AI assistant; the executable
truth remains the code and tests.

## The short version

Cadence is no longer a command-line demonstration. `streamlit_app.py` provides
an evidence-forward listening room where a person can:

1. describe a musical moment in everyday language;
2. see what Cadence interpreted and how the request was processed;
3. inspect grounded recommendations and their ranking signals;
4. reshape the set with controlled preferences, hard filters, variety, or a
   guarded follow-up;
5. undo a change and see which tracks entered or left; and
6. verify privacy, provider, fallback, and provenance claims in developer view.

The UI is intentionally a **thin client**. It does not contain a second
recommendation algorithm. Every submitted transaction calls the same
`MusicCompanion` used by the CLI and evaluation harness.

## The foundational mental model

Think of Cadence as a restaurant with a careful kitchen:

- The **UI is the dining room**. It accepts an order and presents the meal.
- The **input guard is the host**. It removes private details, rejects unsafe
  situations, and decides whether cloud assistance is allowed.
- The **intent parser is the order ticket**. It turns supported words into typed
  musical preferences and hard constraints.
- The **retrievers are the pantry search**. TF-IDF finds matching words;
  embeddings can find related meaning; context guides bridge vocabulary.
- The **structured scorer and fusion step are the chef**. They combine text fit
  with preferences such as lower energy or more acoustic texture.
- **MMR is the plating rule**. It keeps the set varied without admitting weak,
  off-topic tracks.
- The **evaluator is quality control**. It checks IDs, duplicates, constraints,
  evidence, and exact approved framing before anything is served.
- **Cadence's voice is the server**. It can frame a grounded result warmly, but
  it never invents the food—the validated code chooses the songs.

The UI can ask for a different plate, but it never walks around the kitchen's
validation and guardrails.

## What changed from the first proposal

The original plan had a strong product idea, but several details would have
overpromised what the system could prove. The implemented plan makes these
changes:

| Original idea | Implemented decision | Why |
|---|---|---|
| Put the current intent in the URL | Clear all query parameters; no prompt-sharing URL | Even sanitized text is user-authored and can leak through history, referrers, analytics, or screenshots. |
| Re-rank live whenever a control moves | Draft controls inside a form; run once on **Remix this set** | Streamlit reruns on widget changes. A transaction button avoids accidental provider calls, noisy history, and hard-to-reproduce states. |
| Familiar ↔ Adventurous dial | Focused / Balanced / Exploratory presets | The system measures MMR variety, not familiarity or popularity. Named presets are honest and reproducible. |
| Let variety lower relevance freely | Keep one fixed relevance floor; change only MMR lambda | Exploration must never become permission to show off-topic filler. |
| Always-active tempo slider | Explicit **Set a tempo target** toggle | A default slider value would silently add a preference the listener never requested. |
| Raw prompt “never rendered or retained” | Clear it after submit; keep guarded text only in session memory for refinement/undo | The input must render while typed, and the current intent needs a query. It is not written to URLs, UI logs, or disk. |
| Local-only as a display toggle | Typed `ExecutionPolicy(force_local=True)` enforced inside `MusicCompanion` | A privacy badge is not a guardrail unless the backend actually blocks both provider paths. |
| One `gemini` mode for cache and live calls | Separate `EmbeddingSource.CACHE`, `LIVE`, and `LOCAL` | A committed cache is not a network request. The product must distinguish them. |
| Re-run trusted intents without another guard | Re-inspect `MusicIntent.query` and require the previous guard category | A public intent method must not become a way around the input guard. Sensitivity stays sticky across refinements. |
| Show a quick ablation chart from final cards | Defer ablation | A correct text-only/structured-only/fused comparison needs the original pre-fusion pool. Guessing from final cards would be false evidence. |
| Removable interpreted-intent chips | Read-only intent badges plus explicit console controls | Removal semantics need a complete, tested patch for every facet. The current controls make supported changes explicit. |

## Runtime data flow

```text
typed text
  → Streamlit form captures and clears the widget
  → MusicCompanion.respond_detailed(text, ExecutionPolicy)
  → guard → typed intent → retrieve → fuse → diversify → evaluate → voice
  → CompanionTurn(response + request-local receipt)
  → immutable MixSnapshot in st.session_state
  → evidence cards, badges, and developer receipt

submitted refinement
  → controlled IntentPatch OR guarded follow-up parser
  → MusicCompanion.respond_with_intent_detailed/refine_detailed
  → the same retrieve/fuse/diversify/evaluate/voice pipeline
  → one new snapshot + controlled change summary + entered/left track IDs
```

There are two kinds of state:

- **Shared cached resources:** the immutable catalog index and stateless provider
  clients, created once with `@st.cache_resource`.
- **Per-listener session state:** turns, guarded intents, undo snapshots,
  controlled change descriptions, and feedback. This is never placed in the
  shared engine or event log.

This separation prevents one visitor's mix from influencing another visitor.

## The Taste Console

The console distinguishes three concepts that beginners often combine:

1. **Soft preference:** changes order but does not remove eligibility. “Lower
   energy” can lift calmer tracks without banning energetic tracks.
2. **Hard constraint:** removes tracks before ranking. “Instrumental only” means
   a vocal track cannot win no matter how relevant it otherwise is.
3. **Diversity policy:** changes how much near-duplicate penalty MMR applies
   after relevance scoring. It does not change popularity or learn taste.

Energy, mood tone, movement, texture, and tempo become typed `FeatureGoal`
objects. Instrumental and clean become hard booleans. Variety becomes a typed
`DiversityLevel`. An `IntentPatch` validates the whole proposed change before a
new turn is allowed to run.

## Privacy boundaries

“Local-only” has a precise definition:

- On a laptop, the app server and browser can both be local.
- On a hosted deployment, text still travels from the browser to the hosted
  Cadence server.
- In both cases, local-only prevents Cadence from calling the Gemini embedding
  and generation providers for that submitted turn.
- Sensitive input forces that policy even if the visible toggle is off, and the
  sensitive category remains sticky for every refinement of that mix.
- Raw prompt text is cleared from widgets after submission and never placed in
  URLs, request receipts, evolution summaries, JSONL events, or developer traces.
- Guarded query text remains only in the current Streamlit session because
  retrieval refinements and exact undo need it. Reset or session expiry removes
  it.

This is **privacy-first**, not an absolute claim that no server ever handles the
request.

## Honest UI states

The interface renders every bounded action instead of assuming success:

- `recommend`: evidence-backed cards;
- `degraded`: cards plus a visible local-fallback warning;
- `clarify`: a small question instead of arbitrary results;
- `no_match`: an honest failure instead of zero-score filler;
- `safe_response`: a fixed, non-clinical response with no retrieval.

During refinement, clarify, no-match, and safe-response outcomes are transient:
they do not destroy the last working set. This makes experimentation reversible.

## Evidence and score literacy

Each track card displays:

- validated title, artist, genre, mood, and era;
- confirmation of any hard constraints;
- one human-readable grounded reason;
- semantic, keyword, structured, and final ranking signals;
- an explicit distinction between `N/A` (not evaluated) and `0.0` (evaluated,
  no match); and
- a fictional/no-playback label.

These numbers are ranking signals, **not confidence, probability, predicted
enjoyment, or a percentage chance that the listener will like the song**.

Developer view shows a request-local pipeline receipt with candidate/final IDs,
latency, guard category, execution policy, embedding source, network use,
fingerprints, operating mode, and voice source. It never reads a shared log.

## Verification contract

The Phase 4 gate is:

```bash
python -m pytest -q tests/test_refine.py tests/test_ui.py
python -m pytest -q
python scripts/evaluate.py
streamlit run streamlit_app.py
```

Automated coverage checks normal, clarify, no-match, crisis-safe, PII,
provider-outage, cached-semantic, controlled-refinement, undo, transactional
console, sticky privacy, developer-view, and backend/UI-ID parity paths. The
evaluation report must remain PASS; UI work is not allowed to weaken ranking
quality to gain visual polish.

## Launch-oriented next steps

Phase 4 is a portfolio-quality product surface, not yet a commercial music
service. The highest-value next work is:

1. capture and review desktop and narrow-screen screenshots in a connected
   browser, including keyboard-only and reduced-motion checks;
2. deploy to a staging URL with provider use disabled first, then run privacy
   and outage smoke tests;
3. add real, licensed/public catalog ingestion with field-level provenance and
   data-quality quarantine;
4. evaluate session personalization before allowing feedback to alter ranking;
5. add authentication, rate limiting, abuse controls, retention policy, and
   operational monitoring before any public launch; and
6. add ablation only after the backend preserves the original pre-fusion pool.

Do not add fake playback, fabricated popularity, social proof, or unsupported
“AI confidence” merely to resemble a commercial streaming application.

## Understanding check

You should now be able to explain:

1. Why does a hard constraint filter candidates while a soft preference only
   changes their order?
2. Why does the console submit one transaction instead of calling the backend
   on every widget movement?
3. Why can cached semantic retrieval say “AI-built index” and “no network call”
   at the same time?
4. Why is local-only enforced in `MusicCompanion`, not just in Streamlit?
5. Why is a correct ablation view deferred even though it would look impressive?
