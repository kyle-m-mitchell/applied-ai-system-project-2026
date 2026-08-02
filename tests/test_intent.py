"""Tests for the deterministic intent parser."""

from __future__ import annotations

import pytest

from src.intent import IntentParser


@pytest.fixture(scope="module")
def parser() -> IntentParser:
    return IntentParser()


def test_hard_filter_cues(parser):
    intent = parser.parse("clean instrumental study music")
    assert intent.instrumental_only is True
    assert intent.exclude_explicit is True
    assert intent.needs_clarification is False


@pytest.mark.parametrize(
    "text, expected_genre",
    [
        ("some jazz please", "jazz"),
        ("indie pop tracks", "indie pop"),  # multi-word wins over "pop"
        ("play some r&b", "r&b"),
    ],
)
def test_genre_detection(parser, text, expected_genre):
    assert parser.parse(text).genre == expected_genre


def test_mood_detection(parser):
    assert parser.parse("something chill for the evening").mood == "chill"


def test_vague_query_asks_for_clarification(parser):
    intent = parser.parse("music")
    assert intent.needs_clarification is True
    assert intent.clarification


def test_free_text_is_preserved_for_retrieval(parser):
    intent = parser.parse("wistful rainy afternoon")
    assert intent.needs_clarification is False
    assert intent.query == "wistful rainy afternoon"
    assert intent.genre is None and intent.mood is None


def test_redacted_placeholder_does_not_count_as_signal(parser):
    # Only the marker plus one word -> still too little to search on.
    intent = parser.parse("[redacted]")
    assert intent.needs_clarification is True


def _goals(parser, text):
    return {(g.feature, g.relation.value, g.target) for g in parser.parse(text).feature_goals}


def test_directional_numeric_cues(parser):
    assert ("energy", "prefer_high", None) in _goals(parser, "high energy workout")
    assert ("energy", "prefer_low", None) in _goals(parser, "low-energy evening")
    assert ("acousticness", "prefer_high", None) in _goals(parser, "acoustic set")
    assert ("valence", "prefer_low", None) in _goals(parser, "something melancholy")
    assert ("danceability", "prefer_high", None) in _goals(parser, "danceable party music")


def test_tempo_cues_carry_a_target(parser):
    assert ("tempo_bpm", "near", 120.0) in _goals(parser, "songs around 120 bpm")
    assert ("tempo_bpm", "at_least", 130.0) in _goals(parser, "at least 130 bpm")
    assert ("tempo_bpm", "at_most", 90.0) in _goals(parser, "under 90 bpm")


def test_pure_free_text_has_no_goals_reproduction_path(parser):
    # A query with no structured signal must extract no goals, so fusion is skipped
    # downstream and the result reproduces the pre-structured behavior exactly.
    assert parser.parse("tunes for cramming before an exam").feature_goals == ()


def test_cues_carry_controlled_ids(parser):
    ids = {g.cue_id for g in parser.parse("low energy acoustic around 100 bpm").feature_goals}
    assert ids == {"energy_low_v1", "acoustic_high_v1", "tempo_near_v1"}
