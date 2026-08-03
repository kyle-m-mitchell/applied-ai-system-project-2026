# Cadence — Catalog Data Card

**Status:** fictional control verified; FMA build code fixture-tested; real Full/Lite
artifact measurements pending a pinned-source build.

## Two catalogs

| Catalog | Purpose | Data claim |
|---|---|---|
| `fictional` | preserve the original classroom behavior and its evaluation baseline | 200 project-authored tracks; complete synthetic fields |
| `fma` | discover real independent-music records without inventing missing metadata | official FMA metadata; values stay source-scoped, computed, estimated, derived, or unknown |

Catalogs may reuse local integer ids, so every persistent or cross-component
identity uses `TrackRef(catalog_id, track_id, external_id)`. FMA Full and Lite are
editions of one catalog: the resolver prefers a verified Full artifact and discloses
any fallback to Lite.

## Sources

Following the [official FMA repository](https://github.com/mdeff/fma) and
[FMA paper](https://arxiv.org/abs/1612.01840), the metadata archive contributes:
`tracks.csv` (identity, track/album/artist text, dates, genres, scoped tags,
license, some URLs — coverage varies), `genres.csv` (genre id→name), `echonest.csv`
(Echo Nest-*computed* audio features for a smaller overlap — machine estimates, not
observed truth), and `features.csv` (518 Librosa statistics — offline model inputs,
not descriptions). The archive checksum is pinned in `scripts/build_fma_catalog.py`;
mismatched or unsafe archives are rejected before extraction, and a manifest records
every consumed source's SHA-256.

## Acceptance, quarantine, and Lite selection

FMA Full accepts any row with a valid positive id and nonempty normalized
title+artist; missing genre/license/date/prose/tags/URL/features do **not** reject an
identifiable track (those fields stay unknown). Invalid/duplicate ids and missing
identities go to a deterministic quarantine report. Actual accepted/quarantined
counts come from the generated manifest (the ~106k FMA total is context, not a build
assertion). **Lite** is a deterministic 300-track fallback drawn only from rows with
complete Echo Nest features, grouped by `genre_top`, ordered by a salted stable hash
of the id, and selected round-robin — a portable fallback with useful numeric
evidence, not a statistically representative sample.

## Schema and evidence scope

Only identity is universal; all other fields are optional and scope-preserving:

| Destination | Source / scope | Boundary |
|---|---|---|
| `catalog_id`,`id`,`external_id` | ETL identity | namespaced; `external_id` is never a fabricated URL |
| `title`,`artist` | FMA metadata | required display identity |
| `genre`,`genres` | `genre_top`, `genres_all`, `genres.csv` | source classifications; missing stays unknown |
| `tags` / `album_tags` / `artist_tags` | track / album / artist tags | kept at their own scope; never relabeled |
| `track_/album_/artist_information` | source prose | sanitized to visible plain text; indexed at its scope |
| `license` | track license | source text; missing is not replaced by the metadata license |
| URL fields | supplied URLs | safe absolute HTTP(S) only; no page URL synthesized |
| six audio features | Echo Nest overlap or released estimate | value carries origin, method, confidence, interval when estimated |
| `era` | release/creation date | deterministic decade, not an aesthetic claim |
| `mood_profile` | trustworthy energy + valence | experimental four-quadrant scores; never written into authored `mood` |
| `explicit`,`instrumental` | unavailable for FMA | always unknown; cannot pass clean-/instrumental-only filters |

The fictional catalog's `description`/`mood`/`explicit`/`instrumental` remain valid
for it, and their completeness must never leak into FMA.

## Field lineage — and why unknown ≠ false or zero

Each evidence-sensitive field may carry `FieldLineage` (one of `authored`,
`artist_supplied`, `fma_metadata`, `librosa_computed`, `echonest_computed`,
`model_estimated`, `deterministic_derived`, `unknown`) with source fields, method/
model version, confidence, and interval where applicable. `model_estimated` requires
a version and confidence: having a number is not enough — the product must say where
it came from. Enforced in contracts, retrieval, scoring, evaluation, and the UI:
`explicit=None` does not prove clean lyrics; `instrumental=None` proves nothing; a
missing numeric contributes to neither score numerator nor denominator; two missing
moods are not an MMR match; retrieval documents omit absent values instead of
indexing the word `None`; and evidence lists only fields actually used.

## Specialized estimates and experimental mood

Echo Nest values are kept as `echonest_computed`. A missing value may receive a
`model_estimated` value only if a target-specific model **and** that row pass the
global gates (≥5% held-out MAE improvement over median-dummy *and* Ridge, 75–90%
interval coverage, ≥30% retained coverage) and the local gates (missing-input, range,
OOD, and calibrated width, with retained MAE ≤0.15 for unit features / ≤15 BPM for
tempo); otherwise it stays `None`. The serving app reads baked predictions and never
loads scikit-learn. Real metrics are claimed only from a committed report generated
from the pinned sources. Mood maps energy and valence (sigmoids centered at 0.5,
scale 0.15) into four summing-to-one quadrant scores; the label is `None` when the
lead is <0.10, and every profile is marked `experimental`. The local annotation
harness (`build_mood_annotation_sample.py` / `annotate_mood.py`, gated by
`CADENCE_LOCAL_ANNOTATION=1`) hides predictions and refuses duplicate judgments;
labels are local and uncommitted, and reaching 300 primary + 60 audit labels does not
by itself change production.

## Artifacts and verification

Every SQLite edition ships a canonical JSON manifest plus a checksum sidecar
(artifact/edition identity; schema/ETL versions; source, database, and distribution
SHA-256; accepted/quarantined counts; byte sizes; per-field coverage; licenses and
attribution; supported filters/features/retrieval; research/calibration status).
SQLite opens `mode=ro&immutable=1` with `PRAGMA query_only=ON`; validation checks the
application id, schema version, `quick_check`, FTS5, count, and artifact checksum.
Resolver order: verified local Full → verified release cache / checksummed HTTPS
download → committed Lite → fail closed. The manifest is checksummed, not signed.

## Fictional regression control

`data/songs.csv` holds 200 synthetic tracks across 20 balanced genres (with
`data/legacy_songs.csv` keeping the original 20-row baseline), generated for
deterministic education — not from audio or listener behavior.

```text
legacy baseline  SHA-256: 422930d26024fc1a0a52ee35de9e373aa0110b6ec5d91b23044f9833990f2b8e
fictional catalog SHA-256: 122f10f41ad04be3ebb08ef463e17a1d7a0861bcd4a59910cb88d8be9d0091ab
```

Its evaluation control — **0.863 average genre satisfaction, 100% hard-constraint
adherence** — must be preserved; a new FMA metric never replaces it.

## Uses, risks, and rebuild policy

**Intended:** transparent recommendation/retrieval education; metadata discovery with
no playback claim; testing unknown-safe ranking and lineage; offline evaluation of
estimates and abstention. **Prohibited:** claiming Cadence analyzed audio at runtime;
treating Echo Nest/model values as truth; using missing explicit/instrumental
booleans as evidence; redistributing FMA audio under the metadata license; treating
the catalog as evidence of popularity or quality; using DEAM data/thresholds in
production; persisting web research as catalog truth. **Risks:** FMA reflects one
independent-music ecosystem/era; coverage and vocabulary vary; richer biographies can
help text retrieval for reasons unrelated to fit; the Echo Nest overlap is not random
(selection bias); Librosa/mood axes reduce music to limited dimensions; and Lite's
balancing shifts the source distribution. Any source/schema/parser/salt/model/
threshold change requires a version bump, a clean rebuild from pinned inputs,
checksum comparison, full tests + the fictional gate, regenerated reports, and human
approval — never a hand-edited SQLite file or manifest.
