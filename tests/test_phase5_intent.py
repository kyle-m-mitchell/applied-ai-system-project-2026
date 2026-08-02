"""Catalog-aware intent behavior for FMA's evidence boundaries."""

from __future__ import annotations

import pytest

from src.contracts import FeatureRelation
from src.intent import IntentParser


def _goal_map(intent):
    return {goal.feature: goal.relation for goal in intent.feature_goals}


def test_fictional_parser_keeps_legacy_single_axis_calm_behavior():
    intent = IntentParser().parse("calm music for reading")
    assert _goal_map(intent) == {"energy": FeatureRelation.PREFER_LOW}


@pytest.mark.parametrize(
    "word, expected",
    [
        (
            "upbeat",
            {"energy": FeatureRelation.PREFER_HIGH, "valence": FeatureRelation.PREFER_HIGH},
        ),
        (
            "calm",
            {"energy": FeatureRelation.PREFER_LOW, "valence": FeatureRelation.PREFER_HIGH},
        ),
        (
            "intense",
            {"energy": FeatureRelation.PREFER_HIGH, "valence": FeatureRelation.PREFER_LOW},
        ),
        (
            "somber",
            {"energy": FeatureRelation.PREFER_LOW, "valence": FeatureRelation.PREFER_LOW},
        ),
    ],
)
def test_fma_quadrants_are_transparent_two_axis_goals_not_authored_mood(word, expected):
    intent = IntentParser(experimental_mood_axes=True).parse(f"{word} independent music")
    assert _goal_map(intent) == expected
    assert intent.mood is None


def test_conflicting_quadrants_request_clarification():
    intent = IntentParser(experimental_mood_axes=True).parse("calm but somber music")
    assert intent.needs_clarification
    assert "valence" in (intent.clarification or "")


def test_more_instrumental_is_soft_but_bare_instrumental_stays_hard():
    parser = IntentParser(soft_instrumentalness=True)
    soft = parser.parse("make this more instrumental")
    hard = parser.parse("instrumental music with no vocals")

    assert not soft.instrumental_only
    assert _goal_map(soft) == {"instrumentalness": FeatureRelation.PREFER_HIGH}
    assert hard.instrumental_only
    assert not hard.feature_goals
