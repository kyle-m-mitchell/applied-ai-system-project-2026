# Licensing, Attribution, and Provider Boundaries

Code rights, metadata rights, audio rights, benchmark limits, and API terms are
distinct and not interchangeable.

## Code and project-authored data

- **Code:** MIT ([`LICENSE`](../LICENSE)) — covers this repository's authored code
  only; it does not relicense third-party data, music, provider output, or web content.
- **Fictional data** (`data/songs.csv`, `data/legacy_songs.csv`,
  `data/context_guides/*.md`, and semantic caches derived from them): synthetic,
  project-authored teaching artifacts representing no real recordings or artists. The
  guides are AI-drafted prose pending human content review. See
  [`CATALOG_DATA_CARD.md`](CATALOG_DATA_CARD.md).

## Free Music Archive metadata

FMA [metadata](https://github.com/mdeff/fma) is **CC BY 4.0**, so any distributed
Cadence FMA SQLite artifact must retain attribution, source/version checksums, and
the license notice, preserving the separate scopes of track/album/artist text and
tags. **Crucially, each audio track has its own license — the metadata license grants
no blanket right to stream or redistribute audio.** Cadence distributes metadata and
derived indexes only (no audio or previews); a source-supplied per-track license is
stored when present, and a missing one stays missing (never replaced with "CC BY
4.0"). Echo Nest and Librosa values shipped in the archive are machine-computed
metadata with explicit lineage; a Cadence estimate stays labeled `model_estimated`.
Recommended attribution near any download:

```text
Contains metadata from the Free Music Archive (FMA), licensed CC BY 4.0.
Individual audio-track licenses are separate. Cadence does not distribute audio.
```

## DEAM benchmark isolation

DEAM ([manual](https://cvml.unige.ch/databases/DEAM/manual.pdf)) is **CC BY-NC**, so
it is never a product training or calibration source. `scripts/benchmark_deam.py`
requires `--acknowledge-noncommercial`, never downloads DEAM, and restricts output to
`eval/noncommercial/`. No DEAM data, weights, thresholds, or reports may enter FMA
artifacts, production predictions, releases, or the app (`production_effect: none`).

## Provider boundaries (MusicBrainz, Gemini)

Optional per-track research calls **MusicBrainz** with an identifiable `User-Agent`
and a 1-request/second limit, sending only the selected title and artist, storing the
result in the UI session only, and abstaining on multiple exact identities. Review the
[API](https://musicbrainz.org/doc/MusicBrainz_API) and
[rate-limiting](https://musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting) rules
before deployment.

**Gemini** is optional (live embeddings, bounded voice framing, user-triggered
research); the key comes from `GEMINI_API_KEY` or Streamlit secrets and is never
committed. Boundaries: sensitive guarded requests cannot call the provider; local-only
is enforced by backend policy, not a badge; post-ranking research receives only a
resolved track identity (never the prompt, history, or preferences); research
responses are untrusted (safe URLs + citation coverage required); and research is
session-only and never becomes catalog truth. Verify current
[Gemini terms](https://ai.google.dev/gemini-api/terms) and
[grounding docs](https://ai.google.dev/gemini-api/docs/google-search) before public use.

## Web research and citations

A published brief carries at most three short, cited claims plus an optional short,
model-written **narrative** presenting those cited findings creatively — grounded in
the same sources, injection-screened, withheld if it quotes any title beyond the
resolved identity or a cited source, and shown as "Cadence's note," never catalog
truth. When grounded web search is unavailable, a **non-grounded note** may instead be
written *only* from the track's own catalog attributes and is labeled "not
web-verified." Citations are provenance for a claim, not a license to republish a
source; Cadence copies no third-party page into the catalog and avoids lyrics or
extended quotations.

## Distribution (GitHub Releases)

A full FMA SQLite distribution is a **checksummed GitHub Release asset**, not a normal
repo blob. The manifest records uncompressed and compressed SHA-256 digests and byte
sizes; runtime download accepts an explicitly-configured HTTPS URL only and verifies
the digest before an atomic replacement. A release host changes none of the underlying
licenses. See [GitHub release docs](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## Release checklist

Before public release: verify every source/artifact checksum; include FMA CC BY 4.0
attribution beside the asset; confirm no audio is packaged and per-track license
fields remain per-track/unknown; confirm no DEAM material entered production; verify
MusicBrainz identity/rate-limit behavior; review current Gemini/Streamlit/GitHub/
MusicBrainz terms; scan the repo and package for API keys and local annotations; and
record the reviewer, date, and decision.
