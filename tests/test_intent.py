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
