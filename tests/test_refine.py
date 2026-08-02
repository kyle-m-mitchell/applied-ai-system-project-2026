"""Controlled intent-patch and refinement tests (deterministic, offline)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.contracts import FeatureGoal, FeatureRelation, GuardCategory, MusicIntent
from src.refine import (
    IntentPatch,
    apply_intent_patch,
    apply_refinement,
    combine_guard_categories,
    merge_intents,
    remove_intent_facet,
)


def _goal(feature: str, relation: FeatureRelation, cue: str) -> FeatureGoal:
    return FeatureGoal(feature=feature, relation=relation, cue_id=cue)


def test_opposing_goal_replaces_by_feature_and_base_stays_immutable():
    low = _goal("energy", FeatureRelation.PREFER_LOW, "energy_low_v1")
    base = MusicIntent(query="calm jazz", feature_goals=(low,))

    changed = apply_refinement(base, "energetic")

    assert base.feature_goals == (low,)  # frozen source was not mutated
    assert len(changed.feature_goals) == 1
    assert changed.feature_goals[0].cue_id == "energy_high_v1"


def test_patch_can_clear_neutral_and_set_hard_filters_transactionally():
    base = MusicIntent(
        query="calm acoustic music",
        feature_goals=(
            _goal("energy", FeatureRelation.PREFER_LOW, "energy_low_v1"),
            _goal("acousticness", FeatureRelation.PREFER_HIGH, "acoustic_high_v1"),
        ),
    )
    changed = apply_intent_patch(
        base,
        IntentPatch(
            clear_features=("energy",),
            instrumental_only=True,
            exclude_explicit=True,
        ),
    )
    assert [goal.feature for goal in changed.feature_goals] == ["acousticness"]
    assert changed.instrumental_only and changed.exclude_explicit


def test_remove_visible_facet_uses_allowlist():
    base = MusicIntent(query="jazz", genre="jazz")
    assert remove_intent_facet(base, "genre").genre is None
    with pytest.raises(ValueError, match="unknown intent facet"):
        remove_intent_facet(base, "not-a-facet")


def test_patch_rejects_ambiguous_or_duplicate_operations():
    with pytest.raises(ValidationError, match="set and clear genre"):
        IntentPatch(set_genre="jazz", clear_genre=True)
    with pytest.raises(ValidationError, match="at most one goal"):
        IntentPatch(
            goals=(
                _goal("energy", FeatureRelation.PREFER_LOW, "low"),
                _goal("energy", FeatureRelation.PREFER_HIGH, "high"),
            )
        )


def test_feature_goal_rejects_out_of_domain_or_extraneous_parameters():
    with pytest.raises(ValidationError, match="between 0 and 1"):
        FeatureGoal(
            feature="energy",
            relation=FeatureRelation.NEAR,
            target=999,
            cue_id="bad",
        )
    with pytest.raises(ValidationError, match="between 50 and 200"):
        FeatureGoal(
            feature="tempo_bpm",
            relation=FeatureRelation.NEAR,
            target=0,
            cue_id="bad",
        )
    with pytest.raises(ValidationError, match="accepts no numeric parameters"):
        FeatureGoal(
            feature="energy",
            relation=FeatureRelation.PREFER_HIGH,
            target=0.8,
            cue_id="bad",
        )

    with pytest.raises(ValidationError, match="at most one goal per feature"):
        MusicIntent(
            query="conflicting typed intent",
            feature_goals=(
                _goal("energy", FeatureRelation.PREFER_LOW, "low"),
                _goal("energy", FeatureRelation.PREFER_HIGH, "high"),
            ),
        )


def test_guarded_follow_up_rebuilds_query_without_stale_structured_cues():
    base = MusicIntent(
        query="high energy workout",
        feature_goals=(
            _goal("energy", FeatureRelation.PREFER_HIGH, "energy_high_v1"),
        ),
    )
    follow = MusicIntent(
        query="calmer and more acoustic",
        feature_goals=(
            _goal("energy", FeatureRelation.PREFER_LOW, "energy_low_v1"),
            _goal("acousticness", FeatureRelation.PREFER_HIGH, "acoustic_high_v1"),
        ),
    )
    merged = merge_intents(base, follow)
    by_feature = {goal.feature: goal for goal in merged.feature_goals}
    assert by_feature["energy"].relation is FeatureRelation.PREFER_LOW
    assert by_feature["acousticness"].relation is FeatureRelation.PREFER_HIGH
    assert merged.query == "workout acoustic low energy"
    assert "high energy" not in merged.query


def test_replacing_genre_removes_the_old_genre_from_retrieval_query():
    base = MusicIntent(query="some jazz please", genre="jazz")
    follow = MusicIntent(query="rock", genre="rock")
    merged = merge_intents(base, follow)
    assert merged.genre == "rock"
    assert "jazz" not in merged.query
    assert merged.query.endswith("rock")


def test_controlled_refinement_removes_opposing_cue_but_keeps_activity_context():
    base = MusicIntent(
        query="happy upbeat pop for a party",
        genre="pop",
        feature_goals=(
            _goal("energy", FeatureRelation.PREFER_HIGH, "energy_high_v1"),
            _goal("valence", FeatureRelation.PREFER_HIGH, "valence_high_v1"),
            _goal("danceability", FeatureRelation.PREFER_HIGH, "dance_high_v1"),
        ),
    )
    changed = apply_refinement(base, "moodier")
    assert "happy" not in changed.query and "upbeat" not in changed.query
    assert "party" in changed.query and "moody" in changed.query


def test_guard_category_is_monotonic_across_a_thread():
    assert (
        combine_guard_categories(GuardCategory.SENSITIVE, GuardCategory.OK)
        is GuardCategory.SENSITIVE
    )
    assert (
        combine_guard_categories(GuardCategory.INJECTION, GuardCategory.SENSITIVE)
        is GuardCategory.SENSITIVE
    )
    assert (
        combine_guard_categories(GuardCategory.SENSITIVE, GuardCategory.HIGH_RISK)
        is GuardCategory.HIGH_RISK
    )
    assert (
        combine_guard_categories(GuardCategory.EMPTY, GuardCategory.OK)
        is GuardCategory.OK
    )
