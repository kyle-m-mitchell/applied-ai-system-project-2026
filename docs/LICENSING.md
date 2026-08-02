# Licensing, Attribution, and Provider Boundaries

This file separates code rights, metadata rights, audio rights, benchmark limits,
and API terms. They are not interchangeable.

## Project code

Cadence source code is licensed under MIT; see [`LICENSE`](../LICENSE). MIT covers
the code authored for this repository. It does not relicense third-party data,
music, provider output, trademarks, or linked web content.

## Project-authored fictional data

The following are synthetic/project-authored teaching artifacts:

- `data/songs.csv`
- `data/legacy_songs.csv`
- `data/context_guides/*.md`
- semantic caches derived from those fictional documents

They do not represent real recordings or artists. The context guides remain
AI-drafted prose pending human content review. See
[`CATALOG_DATA_CARD.md`](CATALOG_DATA_CARD.md).

## Free Music Archive metadata

The [official FMA repository](https://github.com/mdeff/fma) identifies its
metadata as **CC BY 4.0**. A distributed Cadence FMA SQLite artifact must therefore
retain attribution, source/version checksums, and the relevant license notice.
Cadence preserves source scopes for track information, album information, artist
biographies, and tags.

Important distinction: each FMA audio track may have its own license. The metadata
license does not grant one blanket right to stream or redistribute all audio.
Phase 5 distributes metadata and derived catalog indexes only—no audio or preview
files. A source-supplied per-track license is stored when present; missing license
text remains missing and must not be replaced with “CC BY 4.0.”

Echo Nest values and Librosa statistics shipped inside the official metadata
archive are treated as machine-computed metadata with explicit lineage. A derived
Cadence estimate remains labeled `model_estimated`; transformation does not erase
source attribution.

Recommended attribution near any catalog download:

```text
Contains metadata from the Free Music Archive (FMA), licensed CC BY 4.0.
Individual audio-track licenses are separate. Cadence does not distribute audio.
```

Before publishing a release asset, verify the current upstream notice and include
the FMA attribution and this repository's generated manifest beside the asset.

## DEAM benchmark isolation

The [DEAM dataset manual](https://cvml.unige.ch/databases/DEAM/manual.pdf)
describes a **CC BY-NC** boundary. DEAM is therefore not a product training or
calibration source in Cadence.

`scripts/benchmark_deam.py` enforces three visible boundaries:

- the operator must pass `--acknowledge-noncommercial`;
- Cadence does not download DEAM;
- any written output must stay under `eval/noncommercial/`.

DEAM data, weights, thresholds, labels, and generated reports must not enter FMA
SQLite artifacts, production model predictions, release packages, or the public
app. The benchmark reports comparison only and has `production_effect: none`.

## MusicBrainz

Optional per-track research first calls the MusicBrainz web service with an
identifiable Cadence `User-Agent` and a one-request-per-second process limit. Only
the selected title and artist are sent. Cadence stores the result in the current
UI session only and abstains on multiple exact recording identities.

Operators must review the current
[MusicBrainz API documentation](https://musicbrainz.org/doc/MusicBrainz_API) and
[rate-limiting rules](https://musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting)
before deployment. A MusicBrainz identity supports research disambiguation; it
does not grant rights to audio or third-party pages.

## Gemini

Gemini use is optional. Cadence can use it for live embeddings, bounded voice
framing, and user-triggered grounded research. The API key comes from
`GEMINI_API_KEY` or Streamlit secrets and must never be committed.

Privacy boundaries:

- sensitive guarded requests cannot call the provider;
- Local-only is enforced by backend policy, not a cosmetic badge;
- post-ranking research receives only a resolved track identity, never listener
  prompt, history, ratings, or preferences;
- research responses are treated as untrusted and require safe URLs plus complete
  structured citation coverage;
- research is session-only and never becomes catalog truth.

Provider pricing, data-use terms, model availability, and grounding terms change.
Verify the current [Gemini API terms](https://ai.google.dev/gemini-api/terms) and
[Google Search grounding documentation](https://ai.google.dev/gemini-api/docs/google-search)
before enabling it in a public deployment.

## GitHub Releases

The planned full FMA SQLite distribution is a checksummed GitHub Release asset,
not an ordinary repository blob. The committed or co-published manifest records
both uncompressed and compressed SHA-256 digests and byte sizes. Runtime download
accepts an explicitly configured HTTPS URL only and verifies the expected digest
before an atomic replacement.

A release host does not change the underlying FMA or per-track licenses. Review
[GitHub's release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
and current limits before publication.

## Web citations

Research citations link to third-party pages; Cadence does not copy those pages
into the catalog. Briefs contain at most three short, supported claims and should
avoid lyrics or extended quotations. A citation is provenance for a claim, not a
license to republish the source.

## Release checklist

Before public release:

- [ ] verify every bundled source and generated artifact checksum;
- [ ] include FMA CC BY 4.0 attribution beside the catalog asset;
- [ ] confirm no audio is packaged;
- [ ] confirm source-supplied track license fields remain per track/unknown;
- [ ] confirm no DEAM material or derived calibration entered production;
- [ ] verify MusicBrainz User-Agent/contact and rate-limit behavior;
- [ ] review current Gemini, Streamlit, GitHub, and MusicBrainz terms;
- [ ] scan the repository and release package for API keys and local annotations;
- [ ] record the reviewer, date, and release decision.
