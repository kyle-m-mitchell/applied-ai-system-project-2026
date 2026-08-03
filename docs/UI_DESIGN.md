# Cadence UI — Design Notes

How the Streamlit interface is built and why. The executable truth remains the code
and tests ([`tests/test_ui.py`](../tests/test_ui.py)).

## A thin, evidence-forward client

`streamlit_app.py` is an evidence-forward "listening room" where a person describes a
musical moment, sees what Cadence interpreted, inspects grounded recommendations and
their ranking signals, reshapes the set with controlled preferences / hard filters /
variety / a guarded follow-up / per-track feedback, undoes changes, and verifies
privacy, provider, fallback, and provenance claims in a developer view.

Crucially, the UI is a **thin client**: it contains no second recommendation
algorithm. Every submitted transaction calls the same `MusicCompanion` used by the
CLI and the evaluation harness, so backend/UI result-id parity is testable.

## Design decisions worth recording

Several first-proposal ideas were revised so the UI never overpromises what the
system can prove:

| Idea | Decision | Why |
|---|---|---|
| Put the current intent in the URL | clear all query params; no share URL | even sanitized text is user-authored and leaks via history/referrers/screenshots |
| Re-rank live on every widget move | draft controls in a form; run once on **Remix** | Streamlit reruns on change; a transaction avoids accidental provider calls and unreproducible states |
| Familiar ↔ Adventurous dial | Focused / Balanced / Exploratory presets | the system measures MMR variety, not familiarity or popularity |
| Let variety lower relevance freely | fixed relevance floor; change only MMR λ | exploration must never become permission for off-topic filler |
| Local-only as a display toggle | typed `ExecutionPolicy(force_local=True)` enforced in `MusicCompanion` | a badge is not a guardrail unless the backend blocks both provider paths |
| One `gemini` mode for cache and live | separate `EmbeddingSource.CACHE` / `LIVE` / `LOCAL` | a committed cache is not a network request |
| Re-run trusted intents without a guard | re-inspect the query; carry the previous guard category | a public intent method must not bypass the guard; sensitivity stays sticky |

Two decisions were later revisited as the backend matured: **removable interpreted
chips** are now wired (each drops one facet and re-runs the guarded pipeline), and a
**text-only vs. structured vs. fused signal comparison** now appears in the developer
view once the pre-fusion pool was surfaced honestly.

## State and data flow

```text
typed text → form captures + clears the widget
  → MusicCompanion.respond_detailed(text, ExecutionPolicy)
  → guard → typed intent → retrieve → fuse → session-taste re-rank → diversify → evaluate → voice
  → CompanionTurn(response + request-local receipt) → immutable MixSnapshot → cards + badges + receipt
```

Two kinds of state are kept apart: **shared cached resources** (the immutable catalog
index and stateless provider clients, built once with `@st.cache_resource`) and
**per-listener session state** (turns, guarded intents, undo snapshots, change
summaries, learned taste). The latter never touches the shared engine or event log,
so one visitor's mix — or feedback — cannot influence another's.

## The Taste Console

The console keeps three often-confused concepts distinct: a **soft preference**
reorders without removing eligibility ("lower energy" lifts calmer tracks); a **hard
constraint** removes tracks before ranking ("instrumental only"); and a **diversity
policy** only changes MMR's near-duplicate penalty. Energy/tone/movement/texture/tempo
become typed `FeatureGoal`s, instrumental/clean become hard booleans, and variety a
`DiversityLevel`; an `IntentPatch` validates the whole proposed change before a new
turn runs. Session-only 👍/👎/⚑ feedback adds a bounded, reversible re-ranking signal.

## Privacy, honest states, and evidence literacy

**Local-only** has a precise meaning: it blocks Gemini calls for a submitted turn.
On a hosted deployment the browser still reaches the Cadence server, so this is
*privacy-first*, not an absolute no-server claim. Sensitive input forces the policy
even if the toggle is off and stays sticky across refinements; raw prompt text is
cleared after submission and never placed in URLs, receipts, evolution summaries,
events, or traces.

The interface renders every bounded action honestly — `recommend`, `degraded` (with a
fallback warning), `clarify`, `no_match`, `safe_response` — and during refinement the
non-recommend outcomes are transient, so experimentation never destroys the last
working set. Each card shows validated facts, hard-constraint confirmations, one
grounded reason, and per-leg ranking signals that explicitly distinguish `N/A` (not
evaluated) from `0.0` (evaluated, no match). **These are ranking signals — not
confidence, probability, or a predicted chance the listener will like the song.** The
developer view adds a request-local receipt (candidate/final ids, latency, guard
category, policy, embedding source, network use, fingerprints, mode, voice source) and
never reads a shared log.

## Verification

```bash
python -m pytest -q tests/test_ui.py   # AppTest: normal, clarify, no-match, crisis-safe,
                                        # PII, outage, refinement, undo, transactional
                                        # console, sticky privacy, feedback, backend/UI parity
python -m pytest -q && python scripts/evaluate.py
```

The evaluation gate must stay PASS: UI work is never allowed to weaken ranking quality
for visual polish, and no fake playback, fabricated popularity, or unsupported "AI
confidence" is added to resemble a commercial service.
