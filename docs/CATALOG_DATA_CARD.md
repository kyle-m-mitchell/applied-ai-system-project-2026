# Cadence Catalog Data Card

Version: **3.0 / Phase 5**  
Updated: **2026-08-02**  
System: **Cadence Applied AI Music Companion**  
Status: **fictional control verified; FMA build code fixture-tested; real Full/Lite artifact measurements pending**

## Why there are two catalogs

Cadence separates a regression control from a real-data product catalog:

| Catalog | Purpose | Data claim |
|---|---|---|
| `fictional` | Preserve the original classroom behavior and its evaluation baseline | 200 project-authored fictional tracks; complete synthetic fields |
| `fma` | Discover real independent-music records without inventing missing metadata | Official FMA metadata; values remain source-scoped, computed, estimated, derived, or unknown |

The catalogs may reuse local integer IDs. Every persistent or cross-component
identity therefore uses `TrackRef(catalog_id, track_id, external_id)`.

FMA Full and FMA Lite are editions of one catalog, not separate listener choices.
The resolver prefers a verified Full artifact and discloses when it falls back to
Lite.

## Sources and primary references

The FMA design follows the [official FMA repository](https://github.com/mdeff/fma)
and [FMA paper](https://arxiv.org/abs/1612.01840).

The official metadata archive contains multiple files with different roles:

- `tracks.csv`: track identity, track information, album information, artist
  biography, dates, genres, scoped tags, license, and some supplied URLs. Coverage
  varies by field.
- `genres.csv`: genre IDs and names used to normalize the track genre lists.
- `echonest.csv`: Echo Nest-computed audio features for a smaller overlap. These
  are machine estimates, not directly observed truth.
- `features.csv`: 518 Librosa-derived summary statistics for nearly all tracks.
  These are offline model inputs, not track descriptions.

The archive checksum is pinned by `scripts/build_fma_catalog.py`; unsafe or
mismatched archives are rejected before extraction. A generated manifest records
the SHA-256 of every consumed source. Do not substitute a mirror or later archive
without an intentional checksum/version update and rebuild.

## Acceptance and quarantine policy

FMA Full attempts to accept every source row with:

1. a valid positive track ID;
2. a nonempty normalized title; and
3. a nonempty normalized artist name.

Missing genre, license, date, prose, tags, URL, or audio features do not reject an
otherwise identifiable track. Those fields remain empty/unknown. Invalid or
duplicate IDs and missing title/artist identities are written to a deterministic
quarantine report with reason codes.

The actual accepted and quarantined counts must come from the generated manifest.
The commonly cited FMA total of about 106,000 is context, not a build assertion.
No production count is recorded here until a real pinned build is committed.

## FMA Lite selection

Lite is a deterministic 300-track fallback selected only from rows with complete
Echo Nest target features. Eligible rows are grouped by available `genre_top`,
ordered by a salted stable hash of track ID, and selected round-robin across genre
groups. Source row order is never a tie-breaker. The final database is stored in
track-ID order.

This produces a portable fallback with useful numeric evidence. It does not claim
to be statistically representative of FMA or listening demand.

## Catalog schema and evidence scope

Only identity is universal. All other fields are optional.

| Destination | FMA source/scope | Meaning and boundary |
|---|---|---|
| `catalog_id`, `id`, `external_id` | ETL identity | Namespaced stable identity; `external_id` is not a fabricated URL |
| `title`, `artist` | FMA metadata | Required display identity |
| `genre`, `genres` | `track.genre_top`, `genres(_all)`, `genres.csv` | Source classifications; missing remains unknown |
| `tags` | track tags | Track-scoped keywords only |
| `album_tags` | album tags | Album-scoped keywords; never relabeled as track tags |
| `artist_tags` | artist tags | Artist-scoped keywords; never relabeled as track tags |
| `track_information` | track information | Source text about the track; sanitized to visible plain text |
| `album_information` | album information | Source text about the album; displayed and indexed as album context |
| `artist_biography` | artist biography | Source text about the artist; displayed and indexed as artist context |
| `license` | track license | Source-supplied text; missing is not replaced by the metadata license |
| URL fields | source-supplied URL candidates | Absolute safe HTTP(S) values only; no FMA page URL is synthesized |
| `energy`, `valence`, `acousticness`, `danceability`, `tempo_bpm`, `instrumentalness` | Echo Nest overlap or released specialized estimate | Value always carries origin, method, confidence, and interval when estimated |
| `era` | release/creation date | Deterministic decade derivation, not an authored aesthetic claim |
| `mood_profile` | trustworthy energy + valence | Experimental four-quadrant distribution; never written into authored `mood` |
| `explicit`, `instrumental` | unavailable as verified booleans | Always unknown for FMA; cannot pass clean-only/instrumental-only filters |

The fictional `description`, `mood`, `explicit`, and `instrumental` fields remain
valid for the fictional catalog. Their completeness must never leak into FMA.

## Field lineage

Each populated evidence-sensitive field may carry `FieldLineage`:

- `authored`
- `artist_supplied`
- `fma_metadata`
- `librosa_computed`
- `echonest_computed`
- `model_estimated`
- `deterministic_derived`
- `unknown`

Lineage records the destination field, source fields, method/model version,
confidence, and prediction interval where applicable. `model_estimated` requires
a model version and confidence. Having a number is not enough; the product must
also be able to say where that number came from.

### Unknown is not false or zero

- `explicit=None` does not prove clean lyrics.
- `instrumental=None` does not prove vocals or instrumentals.
- a missing numeric value contributes neither the score numerator nor denominator;
- two missing moods are not an MMR similarity match;
- retrieval documents omit absent values rather than indexing the word `None`;
- evidence lists only fields actually used.

This rule is enforced in contracts, retrieval, scoring, evaluation, and
presentation—not merely described here.

## Text normalization and retrieval documents

The ETL uses explicit FMA multi-row column names. It does not guess positions.
It applies:

- pandas 3.0.x with 3.0.3 pinned in the ML/ETL environment;
- explicit missing-value handling;
- `ast.literal_eval` only for expected list-like cells;
- Unicode NFC normalization and whitespace normalization;
- visible plain-text extraction from HTML, ignoring script/style content;
- finite/range checks for numeric values;
- safe absolute HTTP(S) URL checks;
- deterministic row ordering and canonical JSON serialization.

FTS5 stores separate weighted columns for title, artist, genres, track/album/artist
tags, track/album/artist prose, and deterministic feature terms. Scope is never
erased to make retrieval look denser.

## Numeric features and specialized estimates

Echo Nest values are retained as `echonest_computed`. Missing values may receive a
specialized estimate only if a target-specific model and that row pass both global
and local gates. Model inputs are Librosa-derived statistics. The serving app
reads baked predictions; it does not install scikit-learn or deserialize a model.

For each target, the report records:

- artist-group train/calibration/locked-test counts;
- median-dummy, Ridge, and gradient-boosting MAE;
- R²;
- 10th–90th interval coverage;
- retained coverage after uncertainty/OOD gates;
- model/version and release status.

Global release requires at least 5% held-out MAE improvement over both baselines,
75–90% interval coverage, and at least 30% retained coverage on otherwise-missing
tracks. Calibration sets a row gate consistent with retained MAE no worse than
`0.15` for unit features or `15 BPM` for tempo. Out-of-range, OOD, excessively
wide, or unreleased predictions remain `None`.

No real metrics are claimed until a report generated from the pinned FMA sources
is committed and reviewed.

## Experimental mood profile

Cadence maps energy and valence into four probability-like quadrant scores:

```text
upbeat  = high_arousal       × positive
calm    = (1 - high_arousal) × positive
intense = high_arousal       × (1 - positive)
somber  = (1 - high_arousal) × (1 - positive)
```

where both axes use a sigmoid centered at `0.5` with scale `0.15`. If the best
quadrant is less than `0.10` ahead of the runner-up, `label=None`. Confidence from
estimated input axes propagates into the profile. The schema marks every profile
`experimental`.

FMA tags can be displayed as source tags but do not change mood scores until a
human calibration gate explicitly permits a bounded tag weight.

## Human annotation data

`scripts/build_mood_annotation_sample.py` creates a deterministic, genre-stratified
target set and deliberately excludes prediction fields. `scripts/annotate_mood.py`
is gated by `CADENCE_LOCAL_ANNOTATION=1`, accepts pseudonymous primary/audit labels,
and refuses duplicate judgments from the same rater for the same track.

The readiness report targets 300 primary-labeled tracks and 60 independent audit
pairs. Reaching raw counts does not automatically change production. A future
promotion script must also define and pass agreement thresholds, document the
human decision, and limit any adjustment to reviewed calibration parameters.

Labels are local research data and are not committed by default. Do not collect
names or listening histories.

## Artifact layout and verification

Every SQLite edition has a canonical JSON manifest plus a manifest checksum
sidecar. The manifest includes:

- artifact/catalog/edition identity;
- schema and ETL versions;
- source, database, and compressed-distribution SHA-256 values;
- accepted/quarantined counts;
- artifact/distribution byte sizes;
- per-field coverage;
- licenses and attribution;
- supported filters/features/retrieval methods;
- research and calibration status.

SQLite is opened with URI `mode=ro&immutable=1` plus `PRAGMA query_only=ON`.
Validation checks the application ID, schema version, `quick_check`, FTS5
availability, track count, and artifact checksum.

The resolver order is:

```text
verified local FMA Full
→ verified release cache or checksummed HTTPS download
→ verified committed FMA Lite
→ fail closed if no artifact is valid
```

The manifest is checksummed, not cryptographically signed. Do not describe it as
a digital signature.

## Fictional regression-control catalog

The fictional catalog remains at `data/songs.csv`; `data/legacy_songs.csv` keeps
the original 20-row baseline. It contains 200 synthetic tracks across 20 evenly
balanced genres. It was generated for deterministic education and testing, not
from audio or listener behavior.

Historical hashes:

```text
legacy baseline SHA-256:
422930d26024fc1a0a52ee35de9e373aa0110b6ec5d91b23044f9833990f2b8e

fictional catalog SHA-256:
122f10f41ad04be3ebb08ef463e17a1d7a0861bcd4a59910cb88d8be9d0091ab
```

The established evaluation control is `0.863` average genre satisfaction with
100% required hard-constraint adherence. Phase 5 is required to preserve this
control; a new FMA metric does not replace it.

## Intended uses

- transparent recommendation and retrieval education;
- independent-music metadata discovery with no playback claim;
- testing unknown-safe ranking and evidence lineage;
- offline evaluation of specialized estimates and abstention;
- portfolio demonstration of an integrated RAG/agent/reliability product.

## Prohibited or unsupported uses

- asserting that Cadence listened to or analyzed audio at runtime;
- treating Echo Nest values or model estimates as observed truths;
- using missing FMA explicit/instrumental booleans as hard evidence;
- redistributing FMA audio under the metadata license;
- using the catalog as evidence of artist popularity, quality, identity, or
  listener preference;
- using DEAM data or derived thresholds in production;
- persisting web research as catalog truth.

## Biases and risks

- FMA represents a particular independent-music ecosystem and time period.
- metadata coverage and vocabulary vary by artist, genre, and contributor;
- richer biographies may improve text retrieval for reasons unrelated to musical
  fit;
- Echo Nest overlap is not a random sample, so supervised targets may create
  selection bias;
- Librosa statistics and the mood formula reduce music to limited acoustic axes;
- genre labels and mood language are culturally contingent;
- FMA Lite's balancing improves demo coverage but changes the source distribution;
- URLs and license text can become stale even when source-supplied.

Evaluation must report results by genre and provenance where sample sizes permit,
not only one aggregate score.

## Rebuild and update policy

Any source, schema, parser, selection salt, model, threshold, or mood-method change
requires:

1. a version bump where behavior changed;
2. a clean rebuild from pinned inputs;
3. deterministic checksum comparison;
4. complete tests and the fictional regression gate;
5. regenerated model and coverage reports;
6. licensing/attribution review;
7. artifact/performance measurements on the documented machine;
8. documentation updates from generated evidence;
9. human approval before replacing a published release asset.

Never edit a generated SQLite file or manifest by hand.
