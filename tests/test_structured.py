"""Tests for the structured-preference scoring leg (direction-aware, offline)."""

from __future__ import annotations

import pytest

from src.contracts import (
    CatalogTrack,
    FeatureGoal,
    FeatureRelation,
    FieldLineage,
    FieldOrigin,
    MusicIntent,
)
from src.structured import goal_score, structured_relevance


def _track(track_id=1, *, genre="jazz", mood="chill", energy=0.5, valence=0.5,
           danceability=0.5, acousticness=0.5, tempo_bpm=125) -> CatalogTrack:
    return CatalogTrack.model_validate(
        {
            "id": track_id, "title": "T", "artist": "A", "genre": genre, "mood": mood,
            "energy": energy, "tempo_bpm": tempo_bpm, "valence": valence,
            "danceability": danceability, "acousticness": acousticness,
            "description": "A placeholder track used only for structured-scoring tests.",
            "tags": ("one", "two"), "contexts": ("alpha", "beta"), "instruments": ("piano",),
            "instrumental": True, "explicit": False, "era": "2020s",
        }
    )


def _goal(feature, relation, **kw):
    return FeatureGoal(feature=feature, relation=relation, cue_id=f"{feature}_test", **kw)


def test_prefer_high_and_low_are_directions_not_targets():
    assert goal_score(_goal("energy", FeatureRelation.PREFER_HIGH), _track(energy=0.9)) == 0.9
    assert goal_score(_goal("energy", FeatureRelation.PREFER_HIGH), _track(energy=0.1)) == 0.1
    assert goal_score(_goal("energy", FeatureRelation.PREFER_LOW), _track(energy=0.1)) == 0.9
    assert goal_score(_goal("energy", FeatureRelation.PREFER_LOW), _track(energy=0.9)) == pytest.approx(0.1)


def test_near_at_least_at_most_range():
    near = _goal("valence", FeatureRelation.NEAR, target=0.5)
    assert goal_score(near, _track(valence=0.5)) == 1.0
    assert goal_score(near, _track(valence=0.2)) == pytest.approx(0.7)

    atleast = _goal("danceability", FeatureRelation.AT_LEAST, target=0.6)
    assert goal_score(atleast, _track(danceability=0.8)) == 1.0        # satisfied -> full
    assert goal_score(atleast, _track(danceability=0.4)) == pytest.approx(0.8)  # ramps below

    atmost = _goal("acousticness", FeatureRelation.AT_MOST, target=0.4)
    assert goal_score(atmost, _track(acousticness=0.2)) == 1.0
    assert goal_score(atmost, _track(acousticness=0.7)) == pytest.approx(0.7)

    rng = _goal("energy", FeatureRelation.RANGE, low=0.3, high=0.7)
    assert goal_score(rng, _track(energy=0.5)) == 1.0
    assert goal_score(rng, _track(energy=0.1)) == pytest.approx(0.8)


def test_tempo_is_normalized_onto_the_unit_scale():
    # 125 BPM is the midpoint of [50, 200]; a NEAR-125 goal peaks there.
    near = _goal("tempo_bpm", FeatureRelation.NEAR, target=125)
    assert goal_score(near, _track(tempo_bpm=125)) == 1.0
    # 50 BPM normalizes to 0.0; distance from 0.5 -> closeness 0.5.
    assert goal_score(near, _track(tempo_bpm=50)) == pytest.approx(0.5)


def test_structured_relevance_none_when_no_signal():
    relevance, reasons = structured_relevance(MusicIntent(), _track())
    assert relevance is None and reasons == ()


def test_structured_relevance_genre_exact_family_and_miss():
    jazz = MusicIntent(genre="jazz")
    assert structured_relevance(jazz, _track(genre="jazz"))[0] == 1.0     # exact
    assert structured_relevance(jazz, _track(genre="lofi"))[0] == 0.5     # same family (mellow)
    assert structured_relevance(jazz, _track(genre="metal"))[0] == 0.0    # unrelated


def test_structured_relevance_combines_components_and_reasons():
    intent = MusicIntent(genre="jazz", feature_goals=(_goal("energy", FeatureRelation.PREFER_HIGH),))
    relevance, reasons = structured_relevance(intent, _track(genre="jazz", energy=0.8))
    # weighted average of genre(=1.0, w4) and energy(=0.8, w0.75): 4.6/4.75
    assert relevance == pytest.approx(4.6 / 4.75)
    assert "genre jazz" in reasons and "energy prefer_high" in reasons
    assert 0.0 <= relevance <= 1.0


def test_unknown_features_abstain_instead_of_weakening_the_score():
    energy = _goal("energy", FeatureRelation.PREFER_HIGH)
    unknown = _track(energy=None)

    assert goal_score(energy, unknown) is None
    assert structured_relevance(MusicIntent(feature_goals=(energy,)), unknown) == (None, ())

    relevance, reasons = structured_relevance(
        MusicIntent(genre="jazz", feature_goals=(energy,)), unknown
    )
    assert relevance == 1.0
    assert reasons == ("genre jazz",)


def test_unknown_category_abstains_instead_of_becoming_a_mismatch():
    assert structured_relevance(MusicIntent(genre="jazz"), _track(genre=None)) == (None, ())
    assert structured_relevance(MusicIntent(mood="chill"), _track(mood=None)) == (None, ())


def test_instrumentalness_is_a_soft_numeric_goal():
    goal = _goal("instrumentalness", FeatureRelation.PREFER_HIGH)
    track = _track().model_copy(update={"instrumentalness": 0.85})
    assert goal_score(goal, track) == 0.85


def test_model_estimate_confidence_discounts_its_structured_contribution():
    goal = _goal("energy", FeatureRelation.PREFER_HIGH)
    track = CatalogTrack.model_validate(
        {
            **_track(energy=0.8).model_dump(),
            "lineage": (
                FieldLineage(
                    field_name="energy",
                    origin=FieldOrigin.MODEL_ESTIMATED,
                    method_version="energy-model-v1",
                    confidence=0.5,
                ),
            ),
        }
    )
    relevance, reasons = structured_relevance(MusicIntent(feature_goals=(goal,)), track)
    assert relevance == pytest.approx(0.4)
    assert reasons == ()  # low-confidence evidence cannot produce a strong-match reason
