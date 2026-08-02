# Cadence — Product Brief

## One-line positioning

**Cadence is a privacy-first, explainable music-discovery companion**: it turns
everyday language into a small, evidence-backed set of recommendations, explains
*why* each track appears, supports refinement and feedback, and keeps working
locally when cloud AI is unavailable.

## Who it's for

A listener who wants to describe a moment in plain words — *"clean chill beats for
studying, no vocals"*, *"wistful rainy-day music"* — and get a short, trustworthy
set of songs with reasons, without surrendering personal data or depending on a
network connection.

## The problem

Mainstream recommenders are powerful but opaque, data-hungry, and cloud-dependent:
you can't see why a track was chosen, you can't easily keep a request private, and
nothing works if the service is down. For a small catalog you also don't need
billions of behavioral events — you need honest matching and clear evidence.

## The promise (what Cadence does)

- Understands a typed request through a **privacy guard** (redacts secrets/PII,
  strips prompt-injection, routes crisis language to a safe response) and a
  **deterministic intent parser**.
- Retrieves from **multiple grounded sources** (validated catalog + curated context
  guides) with **lexical (TF-IDF)** and, when available, **semantic (Gemini
  embeddings)** signals — blended, with provenance on every hit.
- Runs a **bounded agent** with a small set of actions (recommend / clarify /
  no-match / safe-response / degraded) and a **privacy-safe trace**.
- **Grounds every claim**: the song list is always produced by validated code;
  Cadence's voice only *frames* it and is checked against the evidence.
- **Degrades honestly**: no key, no cache, or a provider outage → local TF-IDF +
  the deterministic voice, labeled `degraded` — never a silent or faked result.
- **Keeps sensitive input entirely local** — it reaches neither the retrieval nor
  the language provider.

## What makes it different (the honest niche)

Transparency, control, privacy, offline resilience, and **self-measured quality**.
It explains its evidence, survives provider failure, and (with the evaluation
harness) measures whether it is actually any good.

## Non-goals (explicitly)

Cadence is **not** a Spotify/Apple/YouTube Music replacement. It does **not**
provide playback, licensed audio, user accounts, persistent cross-session
profiles, a fresh commercial catalog, or large-scale behavioral personalization.
Those require music licenses, commercial infrastructure, and legal operations that
are out of scope. The demo catalog is **fictional** (or, later, a clearly-attributed
public dataset) with **no playback**.

## Current status (see `docs/RUBRIC_EVIDENCE.md`)

Implemented and tested offline (139 tests): the validated scorer + service, a
200-track catalog, TF-IDF + context-guide retrieval, Gemini embeddings + hybrid
ranking (committed cache + fallback), the guard + intent parser, the bounded
`MusicCompanion` with MMR diversity, a grounding evaluator, and Cadence's voice —
all through the CLI; an evaluation report card; and a scoring + observability
foundation (shared feature utilities, a unified `RankedCandidate` breakdown, the
public `build_companion` factory, and a privacy-safe event receipt). Next:
structured-preference ranking, a Streamlit UI, real-dataset ingestion, and
session personalization.
