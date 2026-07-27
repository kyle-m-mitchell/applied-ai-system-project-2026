# Data Card: Fictional Music Catalog

Version: **2.0**  
Generated: **2026-07-26**  
Intended system: **Applied AI Music Companion / Cadence**  
Status: **Automated validation complete; human content review pending**

## Summary

This dataset is the authoritative grounding catalog for the music recommender.
It contains exactly **200 fictional tracks across 20 genres, with 10 tracks per
genre**. It expands the original 20-track classroom catalog while preserving
every original value for IDs 1–20.

The dataset is designed to support two different jobs:

1. the original deterministic scorer uses genre, mood, and numeric audio-style
   features; and
2. future RAG retrieval will use descriptions, tags, contexts, instruments, and
   provenance to find semantically relevant tracks.

The tracks are fictional. The values do not come from audio analysis, listener
behavior, licensed music services, or real artist profiles.

## Files and provenance

- [`data/songs.csv`](../data/songs.csv) is the application catalog.
- [`data/legacy_songs.csv`](../data/legacy_songs.csv) is the immutable 20-track
  baseline used to prove that Feature 2 did not rewrite the original project.
- [`scripts/generate_catalog.py`](../scripts/generate_catalog.py) contains the
  deterministic authoring rules and generation-time validations.

Baseline SHA-256:

```text
422930d26024fc1a0a52ee35de9e373aa0110b6ec5d91b23044f9833990f2b8e
```

Generated catalog SHA-256:

```text
122f10f41ad04be3ebb08ef463e17a1d7a0861bcd4a59910cb88d8be9d0091ab
```

Regeneration is byte-idempotent: unchanged source profiles and baseline data
produce the same catalog hash.

## Composition

The 20 genres are:

```text
ambient, blues, classical, country, edm, folk, funk, hip hop, house,
indie pop, jazz, lofi, metal, pop, punk, r&b, reggae, rock, soul, synthwave
```

The original catalog had 17 genres. Feature 2 added `house`, `soul`, and `punk`
and filled every genre to exactly 10 records. This artificial balance is useful
for testing coverage; it is not meant to represent music-market popularity.

Additional distribution facts:

- 48 tracks are marked instrumental; 152 are marked vocal.
- 10 tracks carry a fictional explicit-content flag; 190 are marked clean.
- Era labels range from the `1950s` through the `2020s`.
- All 16 moods belong to the scorer’s documented mood families.

## Schema

| Field | Loaded type | Constraint | Purpose |
|---|---|---|---|
| `id` | integer | Unique, 1–200 | Stable catalog identity and tie-breaker |
| `title` | string | Nonempty | Display and retrieval text |
| `artist` | string | Nonempty, fictional | Display and retrieval text |
| `genre` | string | One of 20 catalog genres | Scoring, filter, retrieval |
| `mood` | string | Mapped mood vocabulary | Scoring and retrieval |
| `energy` | float | 0–1 | Original content score |
| `tempo_bpm` | float | 50–200 | Original content score |
| `valence` | float | 0–1 | Original content score |
| `danceability` | float | 0–1 | Original content score |
| `acousticness` | float | 0–1 | Original content score |
| `description` | string | 20–500 characters in contract; generated records use richer sentences | Primary semantic retrieval text |
| `tags` | tuple after load | Pipe-delimited, unique, 2–12 values | Keywords and evidence |
| `contexts` | tuple after load | Pipe-delimited, unique, 2–12 values | Activity/situation retrieval |
| `instruments` | tuple after load | Pipe-delimited, unique, 1–12 values | Sound-specific retrieval |
| `instrumental` | boolean after load | CSV must be `true` or `false` | Future hard constraint |
| `explicit` | boolean after load | CSV must be `true` or `false` | Future clean-only hard constraint |
| `era` | string | Canonical decade such as `1990s` | Style/filter clue |

`era` is a **stylistic decade label**, not a verified release date. All tracks
are fictional and were generated for this project.

## Authoring method

No network call or language model runs during catalog generation.

The generator defines one reviewed profile per genre containing:

- fictional titles and artist-name vocabularies;
- allowed moods;
- genre-specific tags, contexts, instruments, and a description template;
- a center point for each numeric feature;
- deterministic offsets that create controlled within-genre variation;
- instrumental, explicit, and era assignments.

The process is:

```text
verify immutable 20-row baseline hash
  → copy the original ten fields for IDs 1–20
  → add rich metadata to those rows
  → generate enough fictional rows to bring every genre to ten
  → assign IDs 21–200
  → run generation-time integrity checks
  → write the canonical CSV
```

This approach was chosen over runtime AI generation because the application
needs stable, reviewable, testable evidence. An API-generated catalog could
change between runs, introduce real names or unsafe content, and make evaluation
irreproducible.

## Automated validation

Generation and application tests jointly verify:

- exactly 200 rows and 20 genres × 10;
- IDs exactly 1–200;
- no duplicate normalized title/artist identity;
- exact preservation of all original fields for IDs 1–20;
- exact column schema and canonical encoding;
- valid numeric/BPM ranges;
- nonempty, unique pipe-delimited metadata;
- canonical booleans;
- mapped mood vocabulary;
- only the expected three new genres;
- successful Pydantic validation for every loaded track;
- exact-genre recommendations for house, soul, and punk through the real service;
- rejection of malformed booleans, list cells, and schema drift.

Verification command:

```bash
python -m pytest -q
```

Verified result on 2026-07-26:

```text
30 passed
```

## Human review protocol

Automation can prove shape and consistency, but it cannot decide whether a
description is culturally tasteful, a context feels appropriate, or a genre
profile encodes a stereotype. A person should review one representative record
per genre and every automatically flagged extreme before final submission.

For each record, ask:

1. Do the title and artist appear fictional, respectful, and non-confusing?
2. Do genre, mood, numeric features, description, and tags tell a coherent story?
3. Are the suggested contexts appropriate and non-manipulative?
4. Are instrumental and explicit flags plausible within this fictional record?
5. Does the language avoid cultural caricatures or claims of objective truth?

### Representative sample for sign-off

| Reviewed | ID | Genre | Track | Mood | Example context |
|---|---:|---|---|---|---|
| [ ] | 6 | ambient | Spacewalk Thoughts — Orbit Bloom | chill | meditation |
| [ ] | 17 | blues | Rainwater Blues — Delta Marrow | somber | late-night reflection |
| [ ] | 12 | classical | Winter Adagio — The Hollow Strings | melancholy | focused reading |
| [ ] | 15 | country | Dust Road Home — Marigold County | nostalgic | scenic drives |
| [ ] | 16 | edm | Neon Cathedral — Pulsewidth | euphoric | dance workouts |
| [ ] | 18 | folk | Paper Compass — Wander & Wren | hopeful | quiet road trips |
| [ ] | 20 | funk | Sidewalk Strut — The Groove Committee | playful | dance breaks |
| [ ] | 11 | hip hop | Concrete Kings — Block Cipher | energetic | confidence boosts |
| [ ] | 171 | house | Open Door Rhythm — Civic Groove | uplifting | dance floors |
| [ ] | 10 | indie pop | Rooftop Lights — Indigo Parade | happy | sunny walks |
| [ ] | 7 | jazz | Coffee Shop Stories — Slow Stereo | relaxed | dinner ambience |
| [ ] | 2 | lofi | Midnight Coding — LoRoom | chill | deep study |
| [ ] | 14 | metal | Iron Verdict — Ashfall Method | aggressive | maximum-effort training |
| [ ] | 1 | pop | Sunrise City — Neon Echo | happy | morning motivation |
| [ ] | 191 | punk | Borrowed Megaphone — The Loose Bolts | aggressive | skate sessions |
| [ ] | 19 | r&b | Velvet Hours — Sable Rose | romantic | date-night ambience |
| [ ] | 13 | reggae | Island Mailbox — Palm & Tide | uplifting | beach afternoons |
| [ ] | 3 | rock | Storm Runner — Voltline | intense | hard workouts |
| [ ] | 181 | soul | Hold the Light — Amara Wells | romantic | slow Sunday mornings |
| [ ] | 8 | synthwave | Night Drive Loop — Neon Echo | moody | night driving |

Reviewer: ____________________  
Date: ____________________  
Decision/issues: ____________________

### Flagged extremes for review

Boundary values are valid but may produce overly strong retrieval/ranking
signals. Review these IDs intentionally:

| Feature | Minimum | Maximum |
|---|---|---|
| Energy | ID 45, Tidal Glass — 0.14 | ID 112, Obsidian March — 1.00 |
| Tempo | ID 45, Tidal Glass — 54 BPM | ID 197, Loud Enough Now — 178 BPM |
| Valence | ID 114, Last Iron Dawn — 0.20 | ID 164, Groove Receipt — 0.96 |
| Danceability | ID 93, Letters in Adagio — 0.08 | ID 171, Open Door Rhythm — 1.00 |
| Acousticness | ID 37, Static Horizon — 0.00 | ID 96, Quiet Triumph — 1.00 |

Also review all fictional explicit-flag IDs:

```text
42, 83, 87, 109, 113, 116, 158, 193, 196, 199
```

## Risks and limitations

- **Synthetic labels:** Numeric values and metadata are authored, not measured
  from audio. They are useful for software behavior, not musicological claims.
- **Template artifacts:** Similar sentence structure and repeated genre terms may
  make retrieval look better than it would on messy real-world documents.
- **Author bias:** Genre families, contexts, instruments, moods, and names reflect
  the designers’ assumptions and cultural exposure.
- **Artificial balance:** Ten tracks per genre improves testing equality but says
  nothing about listener demand or real catalog distribution.
- **Genre simplification:** Music crosses genres; assigning one label per track
  hides hybridity and regional variation.
- **Era ambiguity:** Decades describe a fictional aesthetic, not provenance.
- **Explicit flag limitation:** There are no lyrics, so the flag is a simulated
  product constraint rather than a content-analysis result.
- **No popularity or collaborative data:** The system cannot infer “people like
  you also enjoyed” behavior.
- **No demographic attributes:** This avoids demographic profiling but also means
  fairness cannot be evaluated across listener groups from this dataset alone.

## Appropriate and inappropriate use

Appropriate:

- classroom demonstrations of recommendation, RAG, validation, and evaluation;
- deterministic testing of filters and ranking;
- portfolio demonstrations that clearly identify the fictional data.

Inappropriate:

- claiming these are real songs or measured audio properties;
- using the data to make factual claims about artists, cultures, or genres;
- evaluating production recommendation quality or demographic fairness;
- presenting `explicit`, mood, or era fields as objective labels.

## Update policy

Any catalog change must:

1. intentionally update the deterministic source profiles;
2. regenerate `data/songs.csv`;
3. run all tests;
4. update the generated SHA-256 here;
5. preserve or explicitly migrate the legacy snapshot;
6. repeat representative and outlier human review;
7. rebuild any retrieval index whose content hash no longer matches.
