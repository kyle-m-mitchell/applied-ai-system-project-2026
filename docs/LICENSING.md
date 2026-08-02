# Licensing and Data Notices

## Code

The source code is licensed **MIT** — see the top-level [`LICENSE`](../LICENSE)
(permissive, standard for a student/portfolio project). The MIT grant covers code
only; the data notices below govern the bundled and future datasets.

## Data

- **Fictional catalog** (`data/songs.csv`, `data/legacy_songs.csv`,
  `data/context_guides/*.md`): authored/synthesized for this project. Not derived
  from any real service, artist, or audio measurement. See
  [CATALOG_DATA_CARD.md](CATALOG_DATA_CARD.md).
- **Embedding cache** (`data/embeddings/*.json`): Gemini vectors derived from the
  fictional catalog documents; a derived artifact, committed for reproducibility.

## Provider / API

- Gemini is **optional**; the app runs fully locally without it. The API key is
  read only from `GEMINI_API_KEY` in a git-ignored `.env` — never committed. Under
  the free-tier terms, content sent to the provider may be reviewed/used to improve
  products, so no secrets or personal data are sent (see the cloud-AI disclosure in
  the README).

## Future real datasets (planned)

If/when a real dataset is ingested (Free Music Archive is the intended first
profile), its license and attribution **must be recorded per-source**, and metadata
rights must not be conflated with audio rights:

- **FMA** metadata: CC BY 4.0 (individual audio licenses still apply and must be
  recorded).
- **MTG-Jamendo**: noncommercial research use only — not an automatic choice for a
  public/commercial launch.
- **AcousticBrainz**: CC0, but a static 2022 dump (collection ended), not a live
  service.

Each imported bundle will carry a manifest with source URL/version, file checksum,
license, required attribution, and per-field provenance (observed / mapped /
derived / default / missing). No descriptions or values are fabricated for real
tracks.
