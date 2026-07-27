"""Build the deterministic 200-track fictional recommendation catalog.

The first 20 records come from ``data/legacy_songs.csv`` and retain every
original field verbatim. New records and retrieval metadata are derived from
curated genre profiles below; no network service or language model is used.

Run from any directory with::

    python scripts/generate_catalog.py
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATH = REPO_ROOT / "data" / "legacy_songs.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "songs.csv"

BASE_FIELDS = (
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
)
METADATA_FIELDS = (
    "description",
    "tags",
    "contexts",
    "instruments",
    "instrumental",
    "explicit",
    "era",
)
OUTPUT_FIELDS = BASE_FIELDS + METADATA_FIELDS

# This pins the archived input to the exact 20-row catalog that Feature 2
# extends. If the archive changes accidentally, generation stops loudly.
EXPECTED_LEGACY_SHA256 = (
    "422930d26024fc1a0a52ee35de9e373aa0110b6ec5d91b23044f9833990f2b8e"
)


@dataclass(frozen=True)
class GenreProfile:
    """Curated vocabulary and numeric center points for one genre."""

    titles: tuple[str, ...]
    artists: tuple[str, ...]
    moods: tuple[str, ...]
    tags: tuple[str, ...]
    contexts: tuple[str, ...]
    instruments: tuple[str, ...]
    vibe: str
    energy: float
    tempo_bpm: int
    valence: float
    danceability: float
    acousticness: float
    eras: tuple[str, ...]
    instrumental_slots: tuple[int, ...] = ()
    explicit_slots: tuple[int, ...] = ()


PROFILES: dict[str, GenreProfile] = {
    "pop": GenreProfile(
        titles=("Confetti Weather", "Golden Signal", "Open Window", "Polaroid Summer", "Better in Color", "Satellite Smile", "Afterglow Avenue", "Good News Parade", "Weekend Gravity", "Brighter Than Before"),
        artists=("June Arcade", "Mira Bloom", "The Daylights", "Violet Transit", "Cass Nova"),
        moods=("happy", "hopeful", "happy", "playful", "energetic", "uplifting", "romantic", "playful", "hopeful", "euphoric"),
        tags=("catchy hooks", "polished vocals", "bright chorus"),
        contexts=("morning motivation", "singalong drives", "friendly gatherings", "weekend errands"),
        instruments=("synthesizer", "electric guitar", "live drums", "layered vocals"),
        vibe="a polished hook-forward arrangement and a bright singalong chorus",
        energy=0.76, tempo_bpm=118, valence=0.78, danceability=0.76, acousticness=0.20,
        eras=("2010s", "2020s"),
    ),
    "lofi": GenreProfile(
        titles=("Window Seat Notes", "Soft Pencil", "Tea at Two", "Margin Doodles", "Blue Desk Lamp", "Quiet Deadline", "Cloudy Bookmark", "Late Bus Home", "Dust on the Keys", "Half-Finished Letter"),
        artists=("Cassette Garden", "Mosslight", "Study Hall FM", "Small Hours Club", "Juniper Tape"),
        moods=("chill", "focused", "relaxed", "moody", "relaxed", "focused", "chill", "relaxed", "nostalgic", "focused"),
        tags=("dusty beats", "vinyl texture", "soft focus"),
        contexts=("deep study", "quiet reading", "late-night coding", "gentle unwinding"),
        instruments=("electric piano", "muted drums", "vinyl crackle", "warm bass"),
        vibe="dust-softened beats and an intimate loop that stays out of the listener's way",
        energy=0.36, tempo_bpm=76, valence=0.54, danceability=0.57, acousticness=0.76,
        eras=("2010s", "2020s"), instrumental_slots=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    ),
    "rock": GenreProfile(
        titles=("Fault Line Radio", "Redline Morning", "Static Horizon", "Run Through Thunder", "Last Match Burning", "Signal Fire Motel", "Unbroken Engine", "Blacktop Anthem", "Fever Pitch", "Northbound Noise"),
        artists=("Granite Youth", "Signal Quarry", "The Breakers Union", "Copper Riot", "Exit Voltage"),
        moods=("intense", "aggressive", "intense", "moody", "euphoric", "aggressive", "intense", "energetic", "aggressive", "energetic"),
        tags=("guitar riffs", "live drums", "anthemic chorus"),
        contexts=("hard workouts", "road trips", "game-day energy", "confidence boosts"),
        instruments=("distorted guitar", "bass guitar", "acoustic drums", "lead vocals"),
        vibe="cranked guitar riffs and a live-room rhythm section built for release",
        energy=0.84, tempo_bpm=138, valence=0.54, danceability=0.58, acousticness=0.11,
        eras=("1990s", "2000s", "2020s"), explicit_slots=(7,),
    ),
    "ambient": GenreProfile(
        titles=("Cloud Atlas Room", "Tidal Glass", "Moonlit Current", "No Horizon", "Slow Orbit", "Frosted Air", "Weightless Field", "Distant Weather", "Luminous Quiet", "Drift Map"),
        artists=("Aerial Archive", "Still Meridian", "Quiet Geometry", "Pale Current", "Longform Light"),
        moods=("chill", "relaxed", "focused", "chill", "relaxed", "relaxed", "focused", "chill", "chill", "relaxed"),
        tags=("slow evolving", "spacious texture", "meditative"),
        contexts=("meditation", "sleep preparation", "quiet reflection", "stress relief"),
        instruments=("synth pads", "field recordings", "processed piano", "soft drones"),
        vibe="slow-evolving harmonies and open space that encourage patient listening",
        energy=0.22, tempo_bpm=62, valence=0.57, danceability=0.24, acousticness=0.83,
        eras=("2000s", "2010s", "2020s"), instrumental_slots=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    ),
    "jazz": GenreProfile(
        titles=("Blue Hour Table", "Sunday Brass", "Velvet Platform", "Lantern District", "Second Cup Swing", "Rain on Seventh", "Pocket Watch Waltz", "After Midnight Set", "Corner Booth", "Marmalade Moon"),
        artists=("The Rowan Quartet", "Marlowe Keys", "East Ferry Trio", "Cedar Street Five", "Nia Bell Ensemble"),
        moods=("relaxed", "relaxed", "playful", "moody", "romantic", "focused", "energetic", "romantic", "playful", "relaxed"),
        tags=("improvisation", "swing feel", "acoustic ensemble"),
        contexts=("dinner ambience", "coffeehouse work", "evening conversation", "focused listening"),
        instruments=("upright bass", "piano", "brushed drums", "tenor saxophone"),
        vibe="conversational improvisation and an acoustic pocket with room to breathe",
        energy=0.48, tempo_bpm=106, valence=0.63, danceability=0.55, acousticness=0.72,
        eras=("1950s", "1960s", "2020s"), instrumental_slots=(0, 1, 2, 4, 5, 6, 8),
    ),
    "synthwave": GenreProfile(
        titles=("Chrome Sunset", "Laser Exit", "Memory Highway", "Arcade Ghost", "Midnight Overpass", "Electric Mirage", "Signal 1986", "Violet Pursuit", "Digital Raincoat", "Retrograde City"),
        artists=("Night Vector", "Glass Circuit", "Magenta Driver", "Future Polaroid", "Static Vista"),
        moods=("nostalgic", "moody", "moody", "intense", "moody", "moody", "intense", "intense", "chill", "energetic"),
        tags=("retro synths", "night drive", "cinematic pulse"),
        contexts=("night driving", "retro gaming", "creative sprints", "cinematic workouts"),
        instruments=("analog synthesizer", "drum machine", "sequenced bass", "electric guitar"),
        vibe="retro-futurist synthesizers and a cinematic pulse beneath glowing melodies",
        energy=0.72, tempo_bpm=112, valence=0.51, danceability=0.70, acousticness=0.13,
        eras=("1980s", "2010s", "2020s"), instrumental_slots=(1, 3, 6, 8),
    ),
    "indie pop": GenreProfile(
        titles=("Apricot Sky", "Borrowed Bicycle", "Cinema Flowers", "Tiny Revolutions", "Porchlight Theory", "Daydream Receipt", "Suburban Comet", "Lucky Sweater", "Postcard Weather", "Almost Famous Friday"),
        artists=("Honey Static", "Maple Cinema", "The Soft Detours", "June Kite", "Bedroom Atlas"),
        moods=("playful", "hopeful", "nostalgic", "happy", "romantic", "playful", "romantic", "playful", "nostalgic", "happy"),
        tags=("jangly guitars", "diary-like lyrics", "handmade pop"),
        contexts=("sunny walks", "creative breaks", "casual hangs", "coming-of-age playlists"),
        instruments=("clean guitar", "synthesizer", "hand percussion", "soft vocals"),
        vibe="handmade pop detail and a diary-like melody with an off-center charm",
        energy=0.66, tempo_bpm=116, valence=0.70, danceability=0.69, acousticness=0.36,
        eras=("2000s", "2010s", "2020s"),
    ),
    "hip hop": GenreProfile(
        titles=("Blueprint Steps", "City Limit Lessons", "No Shortcuts", "Corner Store Crown", "Quiet Flex", "Meter Running", "Rooftop Cypher", "Built From Scratch", "Late Fee Wisdom", "Forward Motion"),
        artists=("Northside Quill", "Mosaic Major", "Kilo Verse", "Avenue Sage", "Plainspoken J"),
        moods=("energetic", "focused", "energetic", "moody", "intense", "focused", "euphoric", "relaxed", "intense", "focused"),
        tags=("boom bap drums", "lyrical flow", "bass-heavy"),
        contexts=("confidence boosts", "urban commutes", "focused training", "head-nod sessions"),
        instruments=("drum sampler", "sub bass", "chopped keys", "turntable textures"),
        vibe="punchy drums and a clear rhythmic pocket supporting tightly framed verses",
        energy=0.74, tempo_bpm=94, valence=0.58, danceability=0.82, acousticness=0.16,
        eras=("1990s", "2010s", "2020s"), explicit_slots=(3, 7),
    ),
    "classical": GenreProfile(
        titles=("Nocturne for Empty Streets", "Spring Room Sonata", "Ember Quartet", "Riverstone Prelude", "Letters in Adagio", "Morning Conservatory", "Glasswood Etude", "Quiet Triumph", "Snowfall Variations", "Homeward Cadenza"),
        artists=("Alder Chamber Players", "Elena Voss", "The Northbridge Quartet", "Milo Serrin", "Orchard Hall Ensemble"),
        moods=("focused", "relaxed", "melancholy", "romantic", "intense", "relaxed", "romantic", "hopeful", "melancholy", "hopeful"),
        tags=("orchestral", "dynamic movement", "acoustic detail"),
        contexts=("focused reading", "quiet mornings", "formal dinners", "reflective study"),
        instruments=("grand piano", "violin", "cello", "viola"),
        vibe="carefully shaped acoustic dynamics and a patient melodic arc",
        energy=0.38, tempo_bpm=78, valence=0.49, danceability=0.20, acousticness=0.94,
        eras=("1990s", "2010s", "2020s"), instrumental_slots=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    ),
    "reggae": GenreProfile(
        titles=("Sun on the Veranda", "Easy Tide", "Good Neighbor", "Green Horizon", "Harbor Morning", "One More Mango", "Open Gate", "Warm Rain Rhythm", "Slow Road Smiling", "Palm Shade Promise"),
        artists=("Kingston Lantern", "The Easy Currents", "Sol Harbor", "Bright Roots", "Coconut Radio"),
        moods=("uplifting", "relaxed", "happy", "hopeful", "relaxed", "happy", "uplifting", "relaxed", "relaxed", "happy"),
        tags=("offbeat guitar", "dub warmth", "sunlit groove"),
        contexts=("beach afternoons", "backyard gatherings", "slow weekends", "sunny commutes"),
        instruments=("skank guitar", "electric bass", "organ", "hand percussion"),
        vibe="a buoyant offbeat groove and warm low end that leave plenty of sunlight",
        energy=0.60, tempo_bpm=88, valence=0.80, danceability=0.76, acousticness=0.38,
        eras=("1970s", "2000s", "2020s"),
    ),
    "metal": GenreProfile(
        titles=("Cinder Throne", "Beneath the Anvil", "Unyielding", "Ashen Signal", "Teeth of Winter", "Obsidian March", "Gravelung", "Last Iron Dawn", "Ritual Engine", "No Gods in Static"),
        artists=("Warden Ash", "Grim Meridian", "Hollow Foundry", "North Pyre", "The Severed Oath"),
        moods=("aggressive", "aggressive", "moody", "intense", "moody", "intense", "intense", "aggressive", "intense", "aggressive"),
        tags=("heavy riffs", "double kick", "dramatic intensity"),
        contexts=("maximum-effort training", "cathartic release", "high-intensity gaming", "focused adrenaline"),
        instruments=("down-tuned guitar", "double-kick drums", "distorted bass", "harsh vocals"),
        vibe="dense down-tuned riffs and precision percussion aimed at controlled intensity",
        energy=0.93, tempo_bpm=156, valence=0.32, danceability=0.40, acousticness=0.05,
        eras=("1990s", "2000s", "2020s"), explicit_slots=(2, 6, 9),
    ),
    "country": GenreProfile(
        titles=("County Line Coffee", "Porchlight Still On", "Two-Lane Memory", "Wildflower Mile", "Saturday at the Feed Store", "Map in the Glovebox", "Old Barn Radio", "River Bend Letter", "Home Before the Rain", "Copper Sky Goodbye"),
        artists=("Larkspur County", "Eli Mason Road", "The Prairie Letters", "Mae Holloway", "Red Cedar Union"),
        moods=("nostalgic", "happy", "hopeful", "nostalgic", "relaxed", "romantic", "focused", "nostalgic", "relaxed", "nostalgic"),
        tags=("storytelling", "acoustic twang", "open-road warmth"),
        contexts=("scenic drives", "porch evenings", "family cookouts", "reflective travel"),
        instruments=("acoustic guitar", "pedal steel", "fiddle", "brush drums"),
        vibe="plainspoken storytelling and an open-road acoustic arrangement",
        energy=0.53, tempo_bpm=96, valence=0.62, danceability=0.55, acousticness=0.61,
        eras=("1970s", "2000s", "2020s"),
    ),
    "edm": GenreProfile(
        titles=("Voltage Bloom", "Drop the Daylight", "Infinite Floor", "Prism Rush", "Skyline Frequency", "Afterhours Lift", "Pulse Together", "Gravity Strobe", "Festival Signal", "Bright Noise Forever"),
        artists=("Astra Phase", "Lumen Drop", "Kinetic North", "Echo Array", "Nova Circuit"),
        moods=("euphoric", "energetic", "euphoric", "energetic", "uplifting", "intense", "euphoric", "intense", "happy", "euphoric"),
        tags=("festival build", "four-on-the-floor", "synth drop"),
        contexts=("dance workouts", "party peaks", "festival energy", "fast-paced gaming"),
        instruments=("software synthesizer", "drum machine", "sub bass", "vocal chops"),
        vibe="a tension-building electronic arrangement that opens into a bright festival-sized drop",
        energy=0.90, tempo_bpm=128, valence=0.76, danceability=0.88, acousticness=0.04,
        eras=("2010s", "2020s"), instrumental_slots=(1, 4, 7),
    ),
    "blues": GenreProfile(
        titles=("Last Train Lantern", "Kitchen Table Trouble", "Blue Coat Weather", "Rent Due Monday", "River Took My Name", "Slow Fuse Heart", "Back Door Morning", "Tin Roof Mercy", "Empty Pocket Shuffle", "Tuesday Night Remedy"),
        artists=("Etta Clay", "The Low River Band", "Jonah Flint", "Mercy Rail", "Briar James"),
        moods=("somber", "somber", "somber", "aggressive", "focused", "melancholy", "hopeful", "moody", "playful", "hopeful"),
        tags=("twelve-bar feel", "expressive guitar", "weathered vocals"),
        contexts=("late-night reflection", "slow drives", "focused listening", "rainy evenings"),
        instruments=("electric guitar", "harmonica", "upright piano", "shuffle drums"),
        vibe="weathered vocals and expressive guitar phrases grounded in an unhurried shuffle",
        energy=0.46, tempo_bpm=82, valence=0.39, danceability=0.48, acousticness=0.64,
        eras=("1950s", "1970s", "2020s"),
    ),
    "folk": GenreProfile(
        titles=("Threadbare Map", "Lantern in the Pines", "Small Town Astronomy", "Carry the Morning", "Cedar Smoke", "Borrowed Field", "North Wind Postcard", "Kitchen Window Hymn", "Wild Finch Road", "Where the Creek Turns"),
        artists=("Moss & Marrow", "Clara Fielding", "The Juniper Lines", "Owen Vale", "Hearthside Atlas"),
        moods=("relaxed", "hopeful", "romantic", "focused", "nostalgic", "happy", "nostalgic", "relaxed", "hopeful", "nostalgic"),
        tags=("acoustic storytelling", "close harmony", "woodland warmth"),
        contexts=("quiet road trips", "campfire evenings", "journaling", "slow mornings"),
        instruments=("fingerpicked guitar", "mandolin", "upright bass", "close harmonies"),
        vibe="close-miked acoustic storytelling with natural textures and patient harmonies",
        energy=0.43, tempo_bpm=98, valence=0.60, danceability=0.43, acousticness=0.87,
        eras=("1960s", "2010s", "2020s"),
    ),
    "r&b": GenreProfile(
        titles=("Satin Conversation", "Call Me at Blue", "Slow Motion Honest", "After the Last Train", "Honeyed Silence", "Room for Two", "Soft Focus Truth", "Sunday Night Signal", "Warm Side of Midnight", "Stay Until the Rain"),
        artists=("Imani Vale", "Cobalt Rose", "Theo August", "Luna Saye", "The Velvet Standard"),
        moods=("romantic", "romantic", "relaxed", "romantic", "romantic", "energetic", "romantic", "focused", "happy", "melancholy"),
        tags=("silky vocals", "slow groove", "warm harmony"),
        contexts=("date-night ambience", "late-night unwinding", "quiet connection", "slow dancing"),
        instruments=("electric piano", "sub bass", "finger snaps", "layered vocals"),
        vibe="silky vocal layers and spacious harmony riding a patient pocket",
        energy=0.55, tempo_bpm=78, valence=0.64, danceability=0.68, acousticness=0.34,
        eras=("1990s", "2010s", "2020s"), explicit_slots=(6,),
    ),
    "funk": GenreProfile(
        titles=("Pocket Full of Sun", "The Get-Down Clause", "Mustard Jacket", "Groove Receipt", "Elevator Boogie", "Fresh Socks Friday", "Turn Signal Funk", "Basement Roller", "Good Foot Forecast", "One More Strut"),
        artists=("Professor Pocket", "The Brass Habit", "Velvet Sidewalk", "Solar Mustache", "Rhythm Department"),
        moods=("playful", "energetic", "happy", "playful", "energetic", "happy", "relaxed", "euphoric", "energetic", "playful"),
        tags=("syncopated bass", "brass punches", "dance-floor pocket"),
        contexts=("dance breaks", "party arrivals", "cooking with friends", "mood boosts"),
        instruments=("slap bass", "rhythm guitar", "horn section", "tight drums"),
        vibe="a rubbery bass pocket and clipped rhythm guitar accented by bright brass",
        energy=0.79, tempo_bpm=110, valence=0.84, danceability=0.90, acousticness=0.15,
        eras=("1970s", "1990s", "2020s"), instrumental_slots=(5,),
    ),
    "house": GenreProfile(
        titles=("Open Door Rhythm", "Warehouse Sunrise", "Deep End Dancing", "Sunday Floor", "Motion in Blue", "Keys After Midnight", "Common Pulse", "Terrace Lights", "Stay in the Groove", "Morning Comes in Four"),
        artists=("Civic Groove", "Mara Loop", "The Four Count", "Velvet Transit", "Common Room DJ"),
        moods=("uplifting", "focused", "energetic", "happy", "euphoric", "focused", "uplifting", "energetic", "happy", "intense"),
        tags=("four-on-the-floor", "piano stabs", "club groove"),
        contexts=("dance floors", "cardio sessions", "sunset gatherings", "late-night focus"),
        instruments=("drum machine", "piano stabs", "synth bass", "vocal samples"),
        vibe="a steady four-on-the-floor pulse with warm keys and a communal club lift",
        energy=0.82, tempo_bpm=124, valence=0.73, danceability=0.91, acousticness=0.06,
        eras=("1990s", "2010s", "2020s"), instrumental_slots=(1, 4, 7),
    ),
    "soul": GenreProfile(
        titles=("Hold the Light", "Sunday Best Heart", "A Little More Mercy", "Golden Hour Promise", "Truth in Your Hands", "Home in the Chorus", "Good Love Returning", "Carry Me Kindly", "Window Full of Morning", "Still We Rise"),
        artists=("Amara Wells", "The Kindred Sound", "Leon Harbor", "Ruby Maren", "South Street Assembly"),
        moods=("romantic", "uplifting", "romantic", "happy", "romantic", "happy", "romantic", "hopeful", "happy", "euphoric"),
        tags=("gospel harmony", "expressive vocals", "vintage warmth"),
        contexts=("slow Sunday mornings", "meaningful gatherings", "romantic dinners", "restorative listening"),
        instruments=("Hammond organ", "horn section", "electric bass", "gospel choir"),
        vibe="expressive lead vocals framed by gospel harmony and a warm live-band pocket",
        energy=0.62, tempo_bpm=86, valence=0.72, danceability=0.63, acousticness=0.46,
        eras=("1960s", "1970s", "2020s"),
    ),
    "punk": GenreProfile(
        titles=("Borrowed Megaphone", "Three Chords Late", "No Permission Slip", "Basement Deadline", "Cheap Coffee Revolt", "Exit Through the Fence", "Loud Enough Now", "Rules in the Rain", "Fast Forward Failure", "Start Again Screaming"),
        artists=("The Loose Bolts", "Curbside Static", "Minor Emergency", "Paper Barricade", "Lunch Break Riot"),
        moods=("aggressive", "intense", "aggressive", "aggressive", "moody", "intense", "aggressive", "energetic", "playful", "intense"),
        tags=("fast guitars", "shouted hooks", "raw momentum"),
        contexts=("skate sessions", "cathartic commutes", "fast workouts", "garage energy"),
        instruments=("overdriven guitar", "pick bass", "fast drums", "shouted vocals"),
        vibe="fast overdriven chords and shouted hooks delivered with unpolished momentum",
        energy=0.89, tempo_bpm=166, valence=0.50, danceability=0.47, acousticness=0.06,
        eras=("1970s", "1990s", "2020s"), explicit_slots=(2, 5, 8),
    ),
}

GENRE_ORDER = tuple(PROFILES)
RATIO_OFFSETS = (-0.12, -0.08, -0.04, 0.00, 0.04, 0.08, 0.12, -0.06, 0.06, 0.02)
BPM_OFFSETS = (-12, -8, -4, 0, 4, 8, 12, -6, 6, 2)
ALLOWED_MOODS = {
    "chill", "relaxed", "focused",
    "happy", "uplifting", "euphoric", "playful", "hopeful",
    "intense", "energetic", "aggressive",
    "moody", "melancholy", "somber", "nostalgic",
    "romantic",
}


def _read_legacy() -> list[dict[str, str]]:
    source_bytes = LEGACY_PATH.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != EXPECTED_LEGACY_SHA256:
        raise ValueError(
            "data/legacy_songs.csv does not match the archived Feature 1 catalog; "
            f"expected SHA-256 {EXPECTED_LEGACY_SHA256}, got {digest}"
        )

    with LEGACY_PATH.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != BASE_FIELDS:
            raise ValueError(f"Legacy columns must be exactly {BASE_FIELDS}")
        rows = list(reader)

    if len(rows) != 20:
        raise ValueError(f"Legacy catalog must contain 20 tracks, found {len(rows)}")
    if [int(row["id"]) for row in rows] != list(range(1, 21)):
        raise ValueError("Legacy IDs must be the consecutive integers 1 through 20")
    if any(row["genre"] not in PROFILES for row in rows):
        raise ValueError("Every legacy genre must have a curated genre profile")
    return rows


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))


def _metrics(profile: GenreProfile, ordinal: int) -> dict[str, str]:
    """Return repeatable genre-centered features with controlled variation."""

    return {
        "energy": f"{_clamp_ratio(profile.energy + RATIO_OFFSETS[ordinal]):.2f}",
        "tempo_bpm": str(profile.tempo_bpm + BPM_OFFSETS[ordinal]),
        "valence": f"{_clamp_ratio(profile.valence + RATIO_OFFSETS[(ordinal + 3) % 10]):.2f}",
        "danceability": f"{_clamp_ratio(profile.danceability + RATIO_OFFSETS[(ordinal + 6) % 10]):.2f}",
        "acousticness": f"{_clamp_ratio(profile.acousticness + RATIO_OFFSETS[(ordinal + 8) % 10]):.2f}",
    }


def _join_unique(values: tuple[str, ...]) -> str:
    return "|".join(dict.fromkeys(value.strip().lower() for value in values if value.strip()))


def _metadata(
    row: dict[str, str], profile: GenreProfile, ordinal: int
) -> dict[str, str]:
    context = profile.contexts[ordinal % len(profile.contexts)]
    mood_article = "an" if row["mood"][0].lower() in "aeiou" else "a"
    description = (
        f"{row['title']} by {row['artist']} is {mood_article} "
        f"{row['mood']} {row['genre']} track "
        f"with {profile.vibe}. {profile.instruments[0].capitalize()} and "
        f"{profile.instruments[1]} shape a sound designed for {context}."
    )
    tags = _join_unique((row["genre"], row["mood"], *profile.tags))
    rotated_contexts = tuple(
        profile.contexts[(ordinal + offset) % len(profile.contexts)] for offset in range(3)
    )
    return {
        "description": description,
        "tags": tags,
        "contexts": _join_unique(rotated_contexts),
        "instruments": _join_unique(profile.instruments),
        "instrumental": "true" if ordinal in profile.instrumental_slots else "false",
        "explicit": "true" if ordinal in profile.explicit_slots else "false",
        "era": profile.eras[ordinal % len(profile.eras)],
    }


def build_catalog(legacy_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Enrich the legacy rows and fill each genre to exactly ten tracks."""

    output: list[dict[str, str]] = []
    seen_per_genre: defaultdict[str, int] = defaultdict(int)

    for legacy in legacy_rows:
        row = dict(legacy)
        genre = row["genre"]
        ordinal = seen_per_genre[genre]
        seen_per_genre[genre] += 1
        row.update(_metadata(row, PROFILES[genre], ordinal))
        output.append(row)

    next_id = 21
    for genre in GENRE_ORDER:
        profile = PROFILES[genre]
        for ordinal in range(seen_per_genre[genre], 10):
            row = {
                "id": str(next_id),
                "title": profile.titles[ordinal],
                "artist": profile.artists[ordinal % len(profile.artists)],
                "genre": genre,
                "mood": profile.moods[ordinal],
                **_metrics(profile, ordinal),
            }
            row.update(_metadata(row, profile, ordinal))
            output.append(row)
            next_id += 1

    return output


def validate_catalog(
    rows: list[dict[str, str]], legacy_rows: list[dict[str, str]]
) -> None:
    """Fail generation if balance, integrity, schema, or metadata drift."""

    if len(PROFILES) != 20:
        raise ValueError(f"Exactly 20 genre profiles are required, found {len(PROFILES)}")
    for genre, profile in PROFILES.items():
        if len(profile.titles) != 10 or len(profile.moods) != 10:
            raise ValueError(f"{genre} must define exactly 10 titles and 10 moods")
        if set(profile.moods) - ALLOWED_MOODS:
            raise ValueError(
                f"{genre} uses moods outside the recommender's MOOD_FAMILIES: "
                f"{sorted(set(profile.moods) - ALLOWED_MOODS)}"
            )
    if len(rows) != 200:
        raise ValueError(f"Catalog must contain exactly 200 tracks, found {len(rows)}")
    if [int(row["id"]) for row in rows] != list(range(1, 201)):
        raise ValueError("Catalog IDs must be the consecutive integers 1 through 200")

    genre_counts = Counter(row["genre"] for row in rows)
    expected_counts = {genre: 10 for genre in GENRE_ORDER}
    if dict(genre_counts) != expected_counts:
        raise ValueError(f"Every genre must have 10 tracks; got {dict(genre_counts)}")

    identities = [
        (
            " ".join(row["title"].split()).casefold(),
            " ".join(row["artist"].split()).casefold(),
        )
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        duplicates = [identity for identity, count in Counter(identities).items() if count > 1]
        raise ValueError(f"Duplicate title/artist pairs found: {duplicates}")

    for index, legacy in enumerate(legacy_rows):
        generated = rows[index]
        for field in BASE_FIELDS:
            if generated[field] != legacy[field]:
                raise ValueError(
                    f"Legacy value changed for ID {legacy['id']} field {field}: "
                    f"{legacy[field]!r} became {generated[field]!r}"
                )

    for row in rows:
        if set(row) != set(OUTPUT_FIELDS):
            raise ValueError(f"ID {row.get('id')} does not match the output schema")
        if any(not row[field].strip() for field in OUTPUT_FIELDS):
            raise ValueError(f"ID {row['id']} contains a blank required field")
        if row["mood"] not in ALLOWED_MOODS:
            raise ValueError(
                f"ID {row['id']} uses mood {row['mood']!r} outside MOOD_FAMILIES"
            )
        for field in ("energy", "valence", "danceability", "acousticness"):
            value = float(row[field])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"ID {row['id']} has out-of-range {field}: {value}")
        tempo = int(row["tempo_bpm"])
        if not 50 <= tempo <= 200:
            raise ValueError(f"ID {row['id']} has implausible tempo_bpm: {tempo}")
        for field, minimum in (("tags", 4), ("contexts", 3), ("instruments", 4)):
            items = row[field].split("|")
            if (
                len(items) < minimum
                or any(not item.strip() for item in items)
                or len(items) != len(set(items))
            ):
                raise ValueError(f"ID {row['id']} has invalid pipe-delimited {field}")
        for field in ("instrumental", "explicit"):
            if row[field] not in {"true", "false"}:
                raise ValueError(f"ID {row['id']} has a non-canonical {field} value")
        era = row["era"]
        if not (
            len(era) == 5
            and era[:4].isdigit()
            and era[:2] in {"19", "20"}
            and era[3] == "0"
            and era[4] == "s"
        ):
            raise ValueError(f"ID {row['id']} has non-canonical era {era!r}")
        if len(row["description"].split()) < 20:
            raise ValueError(f"ID {row['id']} needs a richer description")

    added_genres = set(genre_counts) - {row["genre"] for row in legacy_rows}
    if added_genres != {"house", "soul", "punk"}:
        raise ValueError(f"Expected added genres house/soul/punk, got {added_genres}")


def write_catalog(rows: list[dict[str, str]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=OUTPUT_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    legacy_rows = _read_legacy()
    rows = build_catalog(legacy_rows)
    validate_catalog(rows, legacy_rows)
    write_catalog(rows)
    digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    counts = Counter(row["genre"] for row in rows)
    print(f"Wrote {len(rows)} tracks to {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Genres: {len(counts)} (10 tracks each)")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
