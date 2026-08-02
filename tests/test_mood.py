"""Tests for the transparent experimental valence/arousal mapping."""

from __future__ import annotations

import pytest

from src.contracts import MoodProfile
from src.mood import compute_mood_profile, mood_axis_directions


@pytest.mark.parametrize(
    ("energy", "valence", "expected"),
    (
        (0.9, 0.9, "upbeat"),
        (0.1, 0.9, "calm"),
        (0.9, 0.1, "intense"),
        (0.1, 0.1, "somber"),
    ),
)
def test_four_quadrants_follow_energy_and_valence_axes(energy, valence, expected):
    profile = compute_mood_profile(energy, valence)
    assert profile is not None
    assert profile.label == expected
    assert sum(profile.scores.values()) == pytest.approx(1.0)
    assert profile.experimental is True


def test_balanced_axes_keep_scores_but_abstain_from_label():
    profile = compute_mood_profile(0.5, 0.5)
    assert profile is not None
    assert profile.label is None
    assert profile.margin == pytest.approx(0.0)
    assert profile.as_profile_kwargs() == {
        "upbeat": pytest.approx(0.25),
        "calm": pytest.approx(0.25),
        "intense": pytest.approx(0.25),
        "somber": pytest.approx(0.25),
        "label": None,
        "confidence": None,
        "method_version": "cadence-va-quadrant-v1",
        "experimental": True,
    }


def test_input_evidence_confidence_is_not_invented_from_quadrant_score():
    without_lineage = compute_mood_profile(0.95, 0.95)
    with_lineage = compute_mood_profile(
        0.95, 0.95, energy_confidence=0.82, valence_confidence=0.61
    )
    assert without_lineage is not None and without_lineage.confidence is None
    assert with_lineage is not None and with_lineage.confidence == 0.61


def test_computation_payload_matches_shared_immutable_contract():
    computation = compute_mood_profile(0.8, 0.8, energy_confidence=0.9)
    assert computation is not None
    contract = MoodProfile(**computation.as_profile_kwargs())
    assert contract.label is not None and contract.label.value == "upbeat"
    assert contract.confidence == 0.9


def test_missing_axes_abstain_and_corrupt_axes_fail_loudly():
    assert compute_mood_profile(None, 0.7) is None
    assert compute_mood_profile(0.7, None) is None
    with pytest.raises(ValueError, match="energy"):
        compute_mood_profile(float("nan"), 0.7)
    with pytest.raises(ValueError, match="valence"):
        compute_mood_profile(0.7, 1.2)
    with pytest.raises(ValueError, match="real number"):
        compute_mood_profile(True, 0.7)


def test_quadrant_requests_map_to_underlying_numeric_directions():
    assert dict(mood_axis_directions(" Calm ")) == {
        "energy": "prefer_low",
        "valence": "prefer_high",
    }
    with pytest.raises(ValueError, match="unknown experimental mood"):
        mood_axis_directions("mysterious")
