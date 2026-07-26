from copy import deepcopy
from math import inf, nan

import pytest
from pydantic import ValidationError

from src.contracts import OperatingMode, RecommendationRequest
from src.recommender import recommend_songs
from src.service import RecommendationService


def make_catalog() -> list[dict]:
    return [
        {
            "id": 2,
            "title": "Chill Lofi Loop",
            "artist": "Test Artist",
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.4,
            "tempo_bpm": 80,
            "valence": 0.6,
            "danceability": 0.5,
            "acousticness": 0.9,
        },
        {
            "id": 1,
            "title": "Test Pop Track",
            "artist": "Test Artist",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "tempo_bpm": 120,
            "valence": 0.9,
            "danceability": 0.8,
            "acousticness": 0.2,
        },
    ]


def test_service_preserves_legacy_ranking_and_scores():
    catalog = make_catalog()
    request = RecommendationRequest(
        genre="pop",
        mood="happy",
        energy=0.8,
        acousticness=0.2,
        valence=0.9,
        danceability=0.8,
        tempo_bpm=120,
        limit=2,
    )
    legacy_preferences = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "acousticness": 0.2,
        "valence": 0.9,
        "danceability": 0.8,
        "tempo": 120,
    }

    legacy = recommend_songs(legacy_preferences, catalog, k=2)
    result = RecommendationService(catalog).recommend(request)

    assert [item.track.id for item in result.recommendations] == [
        song["id"] for song, _score, _reason in legacy
    ]
    assert [item.raw_score for item in result.recommendations] == pytest.approx(
        [score for _song, score, _reason in legacy]
    )
    assert result.max_possible_score == pytest.approx(7.5)
    assert result.recommendations[0].match_strength == pytest.approx(1.0)
    assert result.operating_mode is OperatingMode.LOCAL


def test_genre_only_exact_match_has_full_match_strength():
    result = RecommendationService(make_catalog()).recommend(
        RecommendationRequest(genre=" POP ", limit=2)
    )

    assert result.max_possible_score == 4.0
    assert result.recommendations[0].track.genre == "pop"
    assert result.recommendations[0].match_strength == pytest.approx(1.0)


def test_zero_numeric_target_is_an_active_preference():
    result = RecommendationService(make_catalog()).recommend(
        RecommendationRequest(energy=0.0, limit=1)
    )

    assert result.max_possible_score == 0.5
    assert 0.0 <= result.recommendations[0].match_strength <= 1.0


def test_unknown_genre_preserves_stable_id_tie_breaking():
    result = RecommendationService(make_catalog()).recommend(
        RecommendationRequest(genre="genre-not-in-catalog", limit=2)
    )

    assert [item.track.id for item in result.recommendations] == [1, 2]
    assert all(item.match_strength == 0.0 for item in result.recommendations)


def test_requesting_more_tracks_than_exist_returns_warning():
    result = RecommendationService(make_catalog()).recommend(
        RecommendationRequest(genre="pop", limit=5)
    )

    assert len(result.recommendations) == 2
    assert result.warnings == (
        "requested 5 tracks, but the catalog contains only 2",
    )


def test_service_does_not_mutate_callers_catalog():
    catalog = make_catalog()
    original = deepcopy(catalog)
    service = RecommendationService(catalog)

    catalog[0]["title"] = "Changed after construction"
    service.recommend(RecommendationRequest(mood="chill", limit=1))

    assert original[0]["title"] == service.catalog[0].title
    assert service.catalog[0].title == "Chill Lofi Loop"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("energy", -0.01),
        ("energy", 1.01),
        ("energy", True),
        ("energy", nan),
        ("tempo_bpm", 49),
        ("tempo_bpm", 201),
        ("tempo_bpm", inf),
        ("limit", True),
        ("limit", 0),
        ("limit", 21),
    ],
)
def test_request_rejects_invalid_numeric_values(field, value):
    with pytest.raises(ValidationError):
        RecommendationRequest.model_validate({field: value, "genre": "pop"})


def test_request_rejects_empty_and_unknown_fields():
    with pytest.raises(ValidationError):
        RecommendationRequest()

    with pytest.raises(ValidationError):
        RecommendationRequest(genre="pop", confidence=0.8)


def test_catalog_rejects_empty_duplicate_and_invalid_records():
    with pytest.raises(ValueError, match="at least one"):
        RecommendationService([])

    duplicate_catalog = make_catalog()
    duplicate_catalog[1]["id"] = duplicate_catalog[0]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        RecommendationService(duplicate_catalog)

    invalid_catalog = make_catalog()
    invalid_catalog[0]["title"] = " "
    with pytest.raises(ValidationError):
        RecommendationService(invalid_catalog)


def test_reasons_are_structured_values():
    result = RecommendationService(make_catalog()).recommend(
        RecommendationRequest(genre="pop", mood="happy", limit=1)
    )

    assert result.recommendations[0].reasons
    assert all(
        isinstance(reason, str) and reason.strip()
        for reason in result.recommendations[0].reasons
    )
