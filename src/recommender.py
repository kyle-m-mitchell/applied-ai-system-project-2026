import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

from src.features import categorical_score, normalize_unit

# --- Scoring configuration -------------------------------------------------
# Weights control how much each feature can contribute to a song's score.
#
# Genre is the decisive signal, and we make that provable rather than hopeful:
#
#     W_genre (4.0)  >  W_mood (1.5) + sum(numeric weights) (2.0)  =  3.5
#
# An unrelated-genre song earns 0 genre points, so the most it can ever reach
# is W_mood + all numerics = 3.5. An exact-genre song banks W_genre = 4.0 up
# front. Therefore *any* exact-genre match outranks *any* unrelated-genre song,
# no matter how well the numerics line up (this is the fix for the "a metal
# track topped a lofi profile" edge case — see README > Experiments). Cousins
# (0.5 * W_genre = 2.0) can still overtake a *weak* exact match when overall
# fit is strong, which is the graceful degradation we want.
#
# The numeric features share a small 2.0 budget: they *fine-tune* the order
# within a genre instead of competing with it.
WEIGHTS = {
    "genre": 4.0,
    "mood": 1.5,
    "energy": 0.50,
    "valence": 0.45,
    "danceability": 0.40,
    "acousticness": 0.35,
    "tempo": 0.30,
}

# Tempo is stored in raw BPM, so it can't use the 0-1 closeness formula
# directly. We map BPM onto 0-1 over a fixed musical range before comparing,
# which keeps tempo on the same footing as the other numeric features and makes
# the default target (125 BPM) land exactly at the neutral midpoint (0.5).
TEMPO_MIN_BPM = 50.0
TEMPO_MAX_BPM = 200.0

# Similarity families give partial credit for "cousin" genres/moods so the
# recommender degrades gracefully (a lofi fan sees ambient/jazz before metal).
# Each genre/mood belongs to exactly one family.
GENRE_FAMILIES = {
    "mellow":     {"lofi", "ambient", "jazz", "classical"},
    "pop_elec":   {"pop", "indie pop", "synthwave", "edm", "house"},
    "rock_heavy": {"rock", "metal", "punk"},
    "roots":      {"country", "folk", "blues"},
    "groove":     {"hip hop", "r&b", "funk", "reggae", "soul"},
}

MOOD_FAMILIES = {
    "calm":     {"chill", "relaxed", "focused"},
    "upbeat":   {"happy", "uplifting", "euphoric", "playful", "hopeful"},
    "intense":  {"intense", "energetic", "aggressive"},
    "somber":   {"moody", "melancholy", "somber", "nostalgic"},
    "romantic": {"romantic"},
}

# Reverse lookups (member -> family) so the category helpers are O(1).
GENRE_TO_FAMILY = {g: fam for fam, members in GENRE_FAMILIES.items() for g in members}
MOOD_TO_FAMILY = {m: fam for fam, members in MOOD_FAMILIES.items() for m in members}

# 0-1 numeric features scored by closeness = 1 - |target - value|.
# Tempo is handled separately because it lives on a BPM scale (see below).
NUMERIC_FEATURES = ("energy", "acousticness", "valence", "danceability")


def _normalize_tempo(bpm: float) -> float:
    """Map a BPM value onto 0-1 over [TEMPO_MIN_BPM, TEMPO_MAX_BPM], clamped."""
    return normalize_unit(bpm, TEMPO_MIN_BPM, TEMPO_MAX_BPM)


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    target_acousticness: float = 0.5
    target_valence: float = 0.5
    target_danceability: float = 0.5
    target_tempo: float = 125.0  # BPM; 125 is the neutral midpoint of [50, 200]

def _profile_to_prefs(user: UserProfile) -> Dict:
    """Adapt a UserProfile object to the dict shape score_song expects."""
    return {
        "genre": user.favorite_genre,
        "mood": user.favorite_mood,
        "energy": user.target_energy,
        "acousticness": user.target_acousticness,
        "valence": user.target_valence,
        "danceability": user.target_danceability,
        "tempo": user.target_tempo,
    }


class Recommender:
    """
    OOP wrapper around the same scoring core used by the functional path.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Store the catalog of Song objects to recommend from."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k Song objects for the user, highest score first."""
        # Reuse the functional core: score + rank as dicts, then map back to Songs.
        prefs = _profile_to_prefs(user)
        ranked = recommend_songs(prefs, [asdict(s) for s in self.songs], k)
        by_id = {s.id: s for s in self.songs}
        return [by_id[song["id"]] for song, _score, _why in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a one-line explanation of how the song scored for the user."""
        prefs = _profile_to_prefs(user)
        score, reasons = score_song(prefs, asdict(song))
        detail = "; ".join(reasons) if reasons else "no strong matches with your profile"
        return f"{song.title} scored {score:.2f}: {detail}"

CATALOG_FIELDS = (
    "id",
    "title",
    "artist",
    "genre",
    "mood",
    "energy",
    "tempo_bpm",
    "valence",
    "danceability",
    "acousticness",
    "description",
    "tags",
    "contexts",
    "instruments",
    "instrumental",
    "explicit",
    "era",
)

LIST_FIELDS = ("tags", "contexts", "instruments")
BOOLEAN_FIELDS = ("instrumental", "explicit")


def _parse_pipe_values(value: Optional[str], field: str) -> Tuple[str, ...]:
    """Parse one canonical pipe-delimited metadata cell."""
    if value is None or not value.strip():
        raise ValueError(f"{field} cannot be empty")

    values = tuple(part.strip().lower() for part in value.split("|"))
    if any(not part for part in values):
        raise ValueError(f"{field} contains an empty pipe-delimited value")
    if len(values) != len(set(values)):
        raise ValueError(f"{field} contains duplicate values")
    return values


def _parse_boolean(value: Optional[str], field: str) -> bool:
    """Parse only the two canonical CSV boolean spellings."""
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{field} must be exactly 'true' or 'false'")


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file into a list of dictionaries.

    The loader is the serialization boundary for the authoritative catalog:
    integers and floats become numbers, pipe-delimited metadata becomes tuples,
    and only canonical lowercase ``true``/``false`` values become booleans.
    Header order is validated so a misspelled or silently added field cannot
    flow into retrieval or ranking unnoticed.

    Required by src/main.py
    """
    int_fields = {"id"}
    float_fields = {"energy", "tempo_bpm", "valence", "danceability", "acousticness"}

    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        actual_fields = tuple(reader.fieldnames or ())
        if actual_fields != CATALOG_FIELDS:
            raise ValueError(
                "catalog columns must be exactly: " + ", ".join(CATALOG_FIELDS)
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                for field in int_fields:
                    row[field] = int(row[field])
                for field in float_fields:
                    row[field] = float(row[field])
                for field in LIST_FIELDS:
                    row[field] = _parse_pipe_values(row[field], field)
                for field in BOOLEAN_FIELDS:
                    row[field] = _parse_boolean(row[field], field)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid catalog data on CSV line {line_number}: {exc}"
                ) from exc
            songs.append(row)
    return songs

def _category_score(pref_value: Optional[str], song_value: Optional[str],
                    mapping: Dict[str, str]) -> float:
    """Categorical match: 1.0 exact, 0.5 same family, 0.0 otherwise.

    Thin wrapper over the shared :func:`src.features.categorical_score` so the
    scorer, MMR diversity, and the structured leg share one definition of "same
    family" (including the ``None == None`` family guard).
    """
    return categorical_score(pref_value, song_value, mapping)


def _genre_score(pref_genre: Optional[str], song_genre: Optional[str]) -> float:
    """Genre similarity: 1.0 exact, 0.5 same family, 0.0 otherwise."""
    return _category_score(pref_genre, song_genre, GENRE_TO_FAMILY)


def _mood_score(pref_mood: Optional[str], song_mood: Optional[str]) -> float:
    """Mood similarity: 1.0 exact, 0.5 same family, 0.0 otherwise."""
    return _category_score(pref_mood, song_mood, MOOD_TO_FAMILY)


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against the user's taste profile.

    Applies the algorithm recipe:
        score = 4.0*genre + 1.5*mood + 0.50*energy + 0.45*valence
              + 0.40*danceability + 0.35*acousticness + 0.30*tempo
    where genre/mood use similarity families (1.0 exact / 0.5 cousin / 0 else)
    and each numeric feature scores its closeness = 1 - |target - value|.
    Tempo's target is given in BPM and normalized to 0-1 before comparison.

    Every profile key is optional; a missing key simply contributes 0.

    Returns (score, reasons), where reasons lists each feature that added
    points so the user can see *why* a song was recommended.
    """
    score = 0.0
    reasons: List[str] = []

    # --- Categorical: genre, mood (exact 1.0 / cousin 0.5 / else 0) ---
    genre_sub = _genre_score(user_prefs.get("genre"), song.get("genre"))
    if genre_sub > 0:
        points = WEIGHTS["genre"] * genre_sub
        score += points
        if genre_sub == 1.0:
            reasons.append(f"genre match ({song['genre']}) +{points:.2f}")
        else:
            reasons.append(f"genre cousin of {user_prefs['genre']} ({song['genre']}) +{points:.2f}")

    mood_sub = _mood_score(user_prefs.get("mood"), song.get("mood"))
    if mood_sub > 0:
        points = WEIGHTS["mood"] * mood_sub
        score += points
        if mood_sub == 1.0:
            reasons.append(f"mood match ({song['mood']}) +{points:.2f}")
        else:
            reasons.append(f"mood cousin of {user_prefs['mood']} ({song['mood']}) +{points:.2f}")

    # --- Numeric: closeness = 1 - |target - value|, scaled by weight ---
    for feature in NUMERIC_FEATURES:
        target = user_prefs.get(feature)
        value = song.get(feature)
        if target is None or value is None:
            continue
        closeness = max(0.0, 1.0 - abs(target - value))
        points = WEIGHTS[feature] * closeness
        if points > 0:
            score += points
            reasons.append(f"{feature} fit (target {target}, song {value}) +{points:.2f}")

    # --- Tempo: same closeness idea, but on a BPM scale so normalize first ---
    target_bpm = user_prefs.get("tempo")
    song_bpm = song.get("tempo_bpm")
    if target_bpm is not None and song_bpm is not None:
        closeness = max(0.0, 1.0 - abs(_normalize_tempo(target_bpm) - _normalize_tempo(song_bpm)))
        points = WEIGHTS["tempo"] * closeness
        if points > 0:
            score += points
            reasons.append(f"tempo fit (target {target_bpm:g} bpm, song {song_bpm:g} bpm) +{points:.2f}")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Ranks the whole catalog against the taste profile and returns the top k.

    score_song is the per-song "judge": every song in the catalog is scored
    first (local step). recommend_songs is the global step — once every song
    has a number, it sorts them all from highest to lowest score (ties broken
    by id so the order is stable) and keeps the best k.

    Returns a list of (song, score, explanation), where explanation is the
    song's reasons joined into one readable string.
    """
    # Judge every song. score_song returns (score, reasons); the * unpacks that
    # tuple so each item becomes (song, score, reasons).
    scored = [(song, *score_song(user_prefs, song)) for song in songs]

    # sorted() returns a NEW list without touching the caller's `songs`.
    # key = (-score, id): negative score => highest first; id => ascending tie-break.
    ranked = sorted(scored, key=lambda item: (-item[1], item[0]["id"]))

    return [
        (song, score, "; ".join(reasons) if reasons else "no strong matches")
        for song, score, reasons in ranked[:k]
    ]
