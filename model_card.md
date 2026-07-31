# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**TasteTether 1.0** — the deterministic content-based baseline for the planned
**Cadence** applied-AI music companion.

---

## 2. Intended Use  

My recommender is designed to gather the music "taste" of a user and suggest songs based off of their preferences. The system assumes the user would prefer listening to their favorite genre, at that moment, by default and returns a list of songs, fine-tuned to their preferences. This is purely for classroom exploration.

---

## 3. How the Model Works  

Real world recommenders tend to be hybrids of content-based filtering and
collaborative filtering. Content based filtering compares a single user's
preference with the attributes of an item to suggest content to the user. My
simulation is content based. It scores each song by how closely its attributes
match a user's taste profile, using no data about other users. It prioritizes
genre most heavily, then mood, then how close a song's numeric qualities are to
what the user wants. Genre is *provably* the decisive signal: its weight (4.0)
is set to exceed everything else combined (mood 1.5 + numeric budget 2.0 = 3.5),
so an exact-genre song always outranks any unrelated-genre song while the mood
and numeric features fine-tune the ranking within a genre.

In plain terms, each song earns points for what it gets right: a matching genre
or mood adds a fixed number of points, and each numeric quality (energy,
valence, danceability, acousticness) adds more points the closer it is to the
user's target. Genre is worth the most points, mood a little less, and the
numeric qualities the least. Every song's points are added into one total score,
and the songs are then ranked from highest to lowest so the best matches appear
first.

**Song features used:** genre, mood, energy, acousticness, valence,
danceability, tempo.

**UserProfile preferences considered:** favorite_genre, favorite_mood,
target_energy, target_acousticness, target_valence, target_danceability,
target_tempo (in BPM). Genre and mood are matched case-insensitively.

Every preference has exactly one song attribute it's compared against.

---

## 4. Data  

The authoritative catalog now contains **200 fictional songs across 20 genres,
with exactly 10 songs per genre**. The original 20 records and their original
values are preserved in the first 20 IDs and in a separate immutable legacy
snapshot. House, soul, and punk were added to the original 17 genres.

Each record contains the original title, artist, genre, mood, and five numeric
audio-style fields (energy, tempo, valence, danceability, acousticness). Feature
2 adds a natural-language description, tags, listening contexts, instruments,
instrumental and explicit flags, and a decade-style era label. These new fields
are validated and intended to ground the future RAG retriever; the current
deterministic score still uses only the original scoring fields.

The catalog is synthetic and reproducibly generated from curated genre
profiles. Its values are authored rather than measured from audio, so they are
appropriate for classroom software tests but not factual music analysis. There
are still no lyrics, language, real popularity, or listener histories, which is
why a collaborative “people like you” approach is not possible. Full provenance,
validation, review status, and limitations are in the
[Catalog Data Card](docs/CATALOG_DATA_CARD.md).

---

## 5. Strengths  

TasteTether does three things well:

**Coherent picks when a taste is clear.** When a profile's dials agree — a genre,
a mood, and sound targets that all point the same way — the results form a tight,
sensible cluster. The *Workout* profile (edm + energetic + loud/fast) returns edm
on top followed by its closest pop and synthwave cousins in energy order, and the
*Study* profile (lofi + chill + calm/quiet) returns a clean run of lofi and
mellow tracks. In these everyday cases the recommendations match intuition
(see Section 7).

**Graceful fallback to "cousin" genres and moods.** Instead of collapsing to
nothing when exact matches run out, the similarity families award partial credit
to related genres and moods, so a lofi listener sees ambient and jazz before
anything jarring. The list degrades smoothly from exact matches, to close
cousins, to everything else.

**Simple, fast, and needs no training data.** The whole model is a weighted sum a
person can read top to bottom and adjust by hand. It runs instantly on the
catalog with no user history, no training step, and no outside services, which
makes it easy to reason about and easy to test.

---

## 6. Limitations and Bias

**Discovered weakness — genre lock-in and authored categories.** Genre is
weighted 4.0 specifically so an exact-genre match always outranks any unrelated
genre. This protects an explicit preference but locks the top of the list into a
hand-drawn neighborhood and makes cross-genre discovery difficult. The balanced
catalog removes the baseline problem where most genres had only one record, but
it does not make genre families fair or objective. Family sizes still differ,
and assigning house to `pop_elec`, punk to `rock_heavy`, or soul to `groove`
reflects designer judgment.

The richer synthetic metadata adds a second risk: descriptions and contexts can
encode stereotypes or overstate what a fictional track is “for.” Embeddings will
make those labels easier to retrieve, not more factual. Automated validation can
prove ranges, uniqueness, and schema integrity, but a person must still review
language, cultural assumptions, and outliers. Match strength helps expose weak
fit, but it is request-relative and must not be presented as calibrated
confidence.

---

## 7. Historical Baseline Evaluation

The experiments below were recorded against the original 20-track baseline and
are preserved because they motivated later changes. They are not current
200-track acceptance results. The current automated catalog/service/retrieval
result is `95 passed` (Features 3/3b/4 added retrieval, context-guide, and
embedding/hybrid tests; Phase 4 added input-guard, intent-parser, and companion
tests, all offline); a dedicated AI/RAG evaluation harness with quality metrics
remains future work.

**Input-privacy and safety (Phase 4).** Natural-language queries pass through a
deterministic guard before anything else: it redacts emails, phone numbers, and
key-like secrets (which are never sent to the provider or logged), strips
prompt-injection directives, and routes clear crisis language to a brief,
non-clinical safe response that points to real help. Sensitive queries are
answered by the local retriever only. The guard is a coarse regex net, not a
perfect classifier, so it deliberately errs toward redacting or staying local.

### Everyday profiles I tested

Think of a taste profile as a set of dials: one for **genre**, one for **mood**,
and five sliders for the sound itself (energy, acousticness, valence,
danceability, and tempo). To see what each dial actually *does*, I tested the
profiles in **pairs**, changing only one dial at a time and leaving everything
else identical. That way, any change in the recommendations can only be caused
by the one dial I moved.

| Nickname | Genre | Mood | Sliders | #1 recommendation |
|---|---|---|---|---|
| **Study** | lofi | chill | calm & quiet (low energy, 75 bpm, acoustic) | Library Rain *(lofi)* |
| **Study→Metal** | metal | chill | calm & quiet *(same as Study)* | Iron Verdict *(metal)* |
| **Study→Energetic** | lofi | energetic | calm & quiet *(same as Study)* | Library Rain *(lofi)* |
| **Study→Hype** | lofi | chill | loud & fast (high energy, 135 bpm, danceable) | Midnight Coding *(lofi)* |
| **Workout** | edm | energetic | loud & fast | Neon Cathedral *(edm)* |

**Pair 1 — change only the genre (Study vs Study→Metal): what does the "genre"
dial control?** I kept the mood (chill) and every slider (quiet, slow, acoustic)
exactly the same and changed a single word, lofi → metal. The whole list flipped:
a calm lofi track on top became a loud metal track on top. The surprising part is
that the metal song that comes back, *Iron Verdict*, is loud, fast, and
"aggressive" — the exact opposite of the quiet, chill settings I asked for, yet
it still wins.
That makes sense: the genre dial outranks everything else combined, so it decides
the *neighborhood* you get recommended from even when your other dials disagree
with it. This is valid behavior (genre is meant to be decisive), but it also
shows the profile was asking for two contradictory things and genre quietly won.
Notice too that the metal list is much less confident (top score 5.11 vs 7.40 for
the lofi list) and its runners-up are actually the same quiet lofi songs — because
metal has only one song and one close cousin (rock), the list runs out of real
matches almost immediately.

**Pair 2 — change only the mood (Study vs Study→Energetic): what does the "mood"
dial control?** I kept the genre (lofi) and the sliders the same and switched the
mood from chill to energetic. Surprisingly, I got back the *exact same three lofi
songs*, just with lower scores and a slightly shuffled order. The reason is that
none of the lofi songs are "energetic" (they are chill or focused), so asking for
an energetic mood simply removed the mood bonus from all of them — and nobody new
could take their place because genre still dominates. One telling detail: in the
chill version *Midnight Coding* was #2, but in the energetic version it slipped to
#3 while *Focus Flow* rose to #2. That happened because Midnight Coding lost a full
chill-match bonus (1.50 points) while Focus Flow only lost a smaller cousin bonus
(0.75), so the near-ties reshuffled. Takeaway: the mood dial fine-tunes the
emotional flavor *within* your genre and breaks ties — but if your genre has no
songs in that mood, asking for it barely changes *who* you get. It is valid, but
it degrades quietly: it never tells you "there are no energetic lofi songs," it
just hands back the chill ones with a smaller number.

**Pair 3 — change only the sliders (Study vs Study→Hype): what do the sound
sliders control?** I kept the genre (lofi) and mood (chill) and flipped every
slider from calm/quiet/slow to loud/fast/danceable, a workout setting. I *still*
got quiet lofi study music. The only thing that changed was that #1 and #2 traded
places: *Midnight Coding* (a hair louder at energy 0.42 and 78 bpm) edged ahead of
the very quiet *Library Rain* (0.35, 72 bpm), because when you ask for a loud song
the slightly-louder track is the closer match. This makes sense: the sliders are
*fine-tuning knobs, not a steering wheel*. They decide which lofi song rises to the
top; they cannot get you out of lofi. It is valid by design, but a listener who
cranks energy to the maximum expecting gym music — while leaving the genre on
lofi — will be surprised to still receive study beats.

**Pair 4 — change everything (Study vs Workout): does it hold together for a
consistent persona?** Here all three dials agree: a loud electronic genre (edm),
an energetic mood, and loud/fast sliders. The result is a tidy, sensible cluster,
edm on top, then its close pop and synthwave cousins, ordered from most to least
energetic. Compared with the quiet, coherent lofi cluster from Study, this shows
the recommender works cleanly when your preferences all point the same direction.
It is the mirror image of Pair 1's metal profile, where the dials fought each
other and produced a jumbled, low-confidence list. So the system is most
trustworthy when genre, mood, and sliders agree, and least trustworthy when they
contradict — in which case it silently sides with genre.

**What was surprising, in one line.** Two of the three preference types, mood and
the sound sliders, mostly *cannot change who you get recommended*; only the genre
dial can. Moving them just changes the scores and the fine ordering. In plain
terms, a profile really means "pick a genre, then gently sort the songs inside
it," which is worth knowing before trusting the mood and slider settings to steer
the results.

### Adversarial edge-case profiles

I also ran a set of **adversarial edge-case profiles** against the original
20-song catalog to find where the scoring logic breaks. Each block below is
historical output from `recommend_songs`.

**Finding 1 — "genre is decisive" holds only conditionally.** Genre is weighted
3.0, but every other signal combined sums to 5.0 (mood 1.5 + numeric 3.5). A
coherent-looking user ("lofi genre, aggressive mood") gets a **metal** track as
their #1 pick, ahead of every lofi song, because the intruder collects mood +
numeric points that exceed the exact-genre song's total. Genre only "wins" when
no off-genre song can rescue itself that way (profile A).

```
A) genre=lofi, mood=romantic, energy=0.97, acousticness=0.03, valence=0.30, danceability=0.42
  1. Midnight Coding      [lofi/chill]        score=4.91
  2. Focus Flow           [lofi/focused]      score=4.80
  3. Library Rain         [lofi/chill]        score=4.67

B) genre=lofi, mood=aggressive, energy=0.97, acousticness=0.03, valence=0.30, danceability=0.42
  1. Iron Verdict         [metal/aggressive]  score=5.00   <-- wrong genre wins
  2. Midnight Coding      [lofi/chill]        score=4.91
  3. Focus Flow           [lofi/focused]      score=4.80
```

**Finding 2 — no input validation.** Out-of-range numeric targets (typos) clamp
to 0 silently; the model drops all numeric signal and falls back to
categorical-only with no error or warning.

```
C) genre=lofi, mood=chill, energy=9000, acousticness=-5, valence=50, danceability=1e9
  1. Midnight Coding      [lofi/chill]     score=4.50   (numeric contribution: 0)
  2. Library Rain         [lofi/chill]     score=4.50
  3. Focus Flow           [lofi/focused]   score=3.75
```

**Finding 3 — categorical matching is case-sensitive.** `"Lofi"` != `"lofi"`, so
capitalizing the favorite silently discards genre and mood; ranking collapses to
numerics and a **folk** song ties the top lofi track.

```
D) genre=Lofi, mood=Chill, energy=0.40, acousticness=0.80, valence=0.55, danceability=0.40
  1. Focus Flow           [lofi/focused]   score=3.34
  2. Midnight Coding      [lofi/chill]     score=3.27
  3. Paper Compass        [folk/hopeful]   score=3.27   <-- folk ties lofi
```

**Finding 4 — no "nothing matches" signal.** When nothing matches, every song
scores 0.00 and the top picks are just the lowest-`id` songs in catalog order.
An empty profile is indistinguishable from a completely wrong one.

```
E) genre=polka, mood=spicy, energy=500, acousticness=500, valence=500, danceability=500
  1. Sunrise City         [pop/happy]      score=0.00
  2. Midnight Coding      [lofi/chill]     score=0.00
  3. Storm Runner         [rock/intense]   score=0.00

F) {}  (empty profile)  -> identical output to E, i.e. plain id order
```

**What surprised me.** The failure mode is always *silent* — no exception, no
warning, just quietly wrong output. The genre-dominance guarantee I reasoned
about in the README was really a property of this catalog's value ranges, not of
the weights themselves. I also confirmed `tempo_bpm` was loaded but never scored,
so a tempo-driven listener was entirely unserved.

**Changes made in response.** I retuned the weights and re-ran the same
profiles:

- Genre 3.0 → 4.0 and the numeric budget 3.5 → 2.0, enforcing
  `W_genre > W_mood + numerics`. Findings 1 is now closed — profile B returns
  lofi songs on top; a wrong-genre track can no longer win.
- Genre/mood matching is now case-insensitive, closing Finding 3.
- Tempo is now scored (target in BPM, normalized to 0–1), closing the
  never-scored gap.
- Strict Pydantic request contracts now reject out-of-range/NaN/infinite values
  and empty requests, closing Finding 2 and the empty-profile half of Finding 4
  at the public boundary.
- A syntactically valid but unknown all-miss genre can still return zero-score
  ID order. That needs an explicit `no_match` situation policy rather than a
  weight change and remains under Future Work.

---

## 8. Future Work  

My testing (Sections 6 and 7) pointed to four concrete next steps:

- **Add a "no confident matches" signal.** Empty requests are now rejected, but
  an unknown all-miss preference can still return zero-score ID order. The
  recommender should notice when nothing clears a minimum bar and say so, instead
  of presenting filler as picks.
- **Inject genre diversity into the top-k.** To soften the filter-bubble /
  family-size lottery from Section 6, the ranker could reserve a slot or two for
  strong matches outside the user's family, trading a little genre purity for
  real discovery.
- **Use and evaluate the new retrieval-ready catalog.** The catalog-growth step
  is complete at 200 tracks/20 genres. Feature 3 added a provenance-aware local
  TF-IDF retriever (`src/retrieval.py`) over the catalog, and Feature 3b added a
  second source — curated context guides (`data/context_guides/`) that expand a
  query toward catalog vocabulary and ride along as cited evidence — making this
  multi-source RAG, with a before/after demo. Feature 4 then added Gemini
  embeddings and semantic+lexical hybrid ranking behind the same `Retriever`
  interface — reproducible via a committed vector cache, a deterministic fake for
  tests, and an honest TF-IDF fallback, with the API key kept in a git-ignored
  `.env`. Still ahead: session feedback as a third source and quantitative
  retrieval measurements in an evaluation harness.

  **Cloud-AI use disclosure.** The embedding path sends catalog and query text to
  Google's Gemini API; under the current free-tier terms that content may be used
  to improve products and reviewed by humans, so no secrets or personal data are
  embedded, and the app runs fully locally without a key.
- **Support smarter, more expressive preferences.** Allow directional targets
  ("at least this energetic" rather than "near this"), and learn the genre/mood
  families from the data instead of hand-drawing them, so the groupings aren't
  just the designer's judgment calls.

---

## 9. Personal Reflection  

The biggest thing I took away is that bias in a recommender rarely arrives as one
dramatic decision — it hides inside small, reasonable-looking choices. Drawing the
genre "families" by hand felt harmless, and so did working with a small catalog,
but together they quietly produced unfair results: a lofi fan gets a rich,
coherent list while a metal fan falls off a cliff into near-random songs, and
nothing in the output warns anyone that it happened. Watching a single number —
the genre weight — decide who gets a good experience and who doesn't made the
trade-offs feel real in a way I didn't expect going in. It has changed how I look
at the apps I use every day: behind a tidy list of "recommended for you" sit
dozens of quiet judgment calls, and the ones that create bias are usually not the
ones that looked risky up front.
