"""Integrity tests for the retrieval-ready fictional music catalog."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from src.contracts import CatalogTrack, RecommendationRequest
from src.recommender import (
    CATALOG_FIELDS,
    GENRE_TO_FAMILY,
    MOOD_TO_FAMILY,
    load_songs,
)
from src.service import RecommendationService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "songs.csv"
LEGACY_CATALOG_PATH = PROJECT_ROOT / "data" / "legacy_songs.csv"
LEGACY_FIELDS = CATALOG_FIELDS[:10]
EXPECTED_GENRES = {
    "ambient",
    "blues",
    "classical",
    "country",
    "edm",
    "folk",
    "funk",
    "hip hop",
    "house",
    "indie pop",
    "jazz",
    "lofi",
    "metal",
    "pop",
    "punk",
    "r&b",
    "reggae",
    "rock",
    "soul",
    "synthwave",
}


@pytest.fixture(scope="module")
def loaded_catalog() -> list[dict]:
    """Load the real catalog once for the module's integrity checks."""
    return load_songs(str(CATALOG_PATH))


def _read_raw_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as catalog_file:
        return list(csv.DictReader(catalog_file))


def test_catalog_has_exact_balanced_shape(loaded_catalog):
    assert len(loaded_catalog) == 200
    assert [track["id"] for track in loaded_catalog] == list(range(1, 201))

    genre_counts = Counter(track["genre"] for track in loaded_catalog)
    assert set(genre_counts) == EXPECTED_GENRES
    assert set(genre_counts.values()) == {10}


def test_every_catalog_row_is_retrieval_ready(loaded_catalog):
    validated_tracks = [
        CatalogTrack.model_validate(track) for track in loaded_catalog
    ]

    identities = {
        (track.title.casefold(), track.artist.casefold())
        for track in validated_tracks
    }
    assert len(identities) == len(validated_tracks)

    for track in validated_tracks:
        assert len(track.description.split()) >= 20
        assert track.genre in GENRE_TO_FAMILY
        assert track.mood in MOOD_TO_FAMILY
        assert len(track.tags) == len(set(track.tags))
        assert len(track.contexts) == len(set(track.contexts))
        assert len(track.instruments) == len(set(track.instruments))
        assert isinstance(track.instrumental, bool)
        assert isinstance(track.explicit, bool)


def test_original_twenty_legacy_values_are_unchanged():
    expanded_rows = _read_raw_csv(CATALOG_PATH)[:20]
    legacy_rows = _read_raw_csv(LEGACY_CATALOG_PATH)

    assert len(legacy_rows) == 20
    assert [
        {field: row[field] for field in LEGACY_FIELDS}
        for row in expanded_rows
    ] == legacy_rows


@pytest.mark.parametrize("genre", ["house", "soul", "punk"])
def test_new_genres_work_through_the_real_service(loaded_catalog, genre):
    result = RecommendationService(loaded_catalog).recommend(
        RecommendationRequest(genre=genre, limit=5)
    )

    assert len(result.recommendations) == 5
    assert all(item.track.genre == genre for item in result.recommendations)


def test_loader_rejects_noncanonical_boolean(tmp_path):
    raw_rows = _read_raw_csv(CATALOG_PATH)
    raw_rows[0]["instrumental"] = "yes"
    malformed_path = tmp_path / "bad_boolean.csv"

    with malformed_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        writer.writerow(raw_rows[0])

    with pytest.raises(ValueError, match="exactly 'true' or 'false'"):
        load_songs(str(malformed_path))


def test_loader_rejects_malformed_pipe_metadata(tmp_path):
    raw_rows = _read_raw_csv(CATALOG_PATH)
    raw_rows[0]["contexts"] = "studying||reading"
    malformed_path = tmp_path / "bad_metadata.csv"

    with malformed_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        writer.writerow(raw_rows[0])

    with pytest.raises(ValueError, match="empty pipe-delimited value"):
        load_songs(str(malformed_path))


def test_loader_rejects_schema_drift(tmp_path):
    malformed_path = tmp_path / "bad_header.csv"
    malformed_path.write_text("id,title,unexpected\n1,Song,value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="catalog columns must be exactly"):
        load_songs(str(malformed_path))
