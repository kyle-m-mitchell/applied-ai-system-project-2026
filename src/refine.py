"""Controlled, immutable refinement of an already-guarded music intent.

The UI never edits rankings directly.  It converts a chip or mixing-desk control
into the same typed :class:`FeatureGoal` values the rule parser produces, merges
those values into a new :class:`MusicIntent`, and asks ``MusicCompanion`` to run
the normal retrieve → fuse → diversify → evaluate → voice pipeline again.

No free text enters through this module.  A conversational follow-up goes through
``MusicCompanion.refine`` so the input guard runs before ``merge_intents``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import Field, model_validator

from src.contracts import (
    ContractModel,
    FeatureGoal,
    FeatureRelation,
    GuardCategory,
    MusicIntent,
)
from src.intent import neutral_query_text, rebuild_retrieval_query


@dataclass(frozen=True)
class Refinement:
    """One allowlisted UI refinement with a typed goal and human label."""

    key: str
    label: str
    description: str
    goal: FeatureGoal


def _goal(feature: str, relation: FeatureRelation, cue_id: str) -> FeatureGoal:
    return FeatureGoal(feature=feature, relation=relation, cue_id=cue_id)


REFINEMENTS: Final[dict[str, Refinement]] = {
    "calmer": Refinement(
        "calmer", "Calmer", "Lower the energy without changing the hard rules.",
        _goal("energy", FeatureRelation.PREFER_LOW, "energy_low_v1"),
    ),
    "energetic": Refinement(
        "energetic", "More energy", "Lift higher-energy tracks in the ordering.",
        _goal("energy", FeatureRelation.PREFER_HIGH, "energy_high_v1"),
    ),
    "acoustic": Refinement(
        "acoustic", "More acoustic", "Favor acoustic textures.",
        _goal("acousticness", FeatureRelation.PREFER_HIGH, "acoustic_high_v1"),
    ),
    "brighter": Refinement(
        "brighter", "Brighter", "Favor more positive emotional color.",
        _goal("valence", FeatureRelation.PREFER_HIGH, "valence_high_v1"),
    ),
    "moodier": Refinement(
        "moodier", "Moodier", "Favor lower-valence, reflective tracks.",
        _goal("valence", FeatureRelation.PREFER_LOW, "valence_low_v1"),
    ),
    "danceable": Refinement(
        "danceable", "More movement", "Favor tracks with stronger danceability.",
        _goal("danceability", FeatureRelation.PREFER_HIGH, "dance_high_v1"),
    ),
}

CONSOLE_FEATURES: Final[tuple[str, ...]] = (
    "energy",
    "valence",
    "danceability",
    "acousticness",
)


def directional_goal(feature: str, direction: int) -> FeatureGoal | None:
    """Translate a console ``-1 / 0 / +1`` into one controlled goal.

    Zero is truly neutral and returns ``None``; it never fabricates a midpoint.
    """
    if feature not in CONSOLE_FEATURES:
        raise ValueError(f"unsupported console feature: {feature}")
    if direction not in (-1, 0, 1):
        raise ValueError("direction must be -1, 0, or 1")
    if direction == 0:
        return None
    relation = (
        FeatureRelation.PREFER_LOW if direction < 0 else FeatureRelation.PREFER_HIGH
    )
    suffix = "low" if direction < 0 else "high"
    established = {
        ("energy", "low"): "energy_low_v1",
        ("energy", "high"): "energy_high_v1",
        ("valence", "low"): "valence_low_v1",
        ("valence", "high"): "valence_high_v1",
        ("danceability", "high"): "dance_high_v1",
        ("acousticness", "high"): "acoustic_high_v1",
    }
    cue_id = established.get((feature, suffix), f"ui_{feature}_{suffix}_v1")
    return FeatureGoal(feature=feature, relation=relation, cue_id=cue_id)


def tempo_goal(target_bpm: float) -> FeatureGoal:
    """Create an explicit near-tempo goal; callers must opt in before using it."""
    if not 50.0 <= target_bpm <= 200.0:
        raise ValueError("tempo target must be between 50 and 200 BPM")
    return FeatureGoal(
        feature="tempo_bpm",
        relation=FeatureRelation.NEAR,
        target=target_bpm,
        cue_id="tempo_near_ui_v1",
    )


class IntentPatch(ContractModel):
    """One transactional, controlled edit to an immutable ``MusicIntent``.

    Explicit set/clear fields avoid the classic ambiguity where ``None`` might
    mean either "leave unchanged" or "remove this value."  A UI transaction may
    perform several compatible edits, but it cannot set and clear the same facet.
    """

    set_genre: str | None = Field(default=None, min_length=1, max_length=80)
    clear_genre: bool = False
    set_mood: str | None = Field(default=None, min_length=1, max_length=80)
    clear_mood: bool = False
    goals: tuple[FeatureGoal, ...] = ()
    clear_features: tuple[str, ...] = ()
    instrumental_only: bool | None = None
    exclude_explicit: bool | None = None
    source: str = "controlled_ui"

    @model_validator(mode="after")
    def coherent_operations(self) -> "IntentPatch":
        if self.set_genre is not None and self.clear_genre:
            raise ValueError("cannot set and clear genre in one patch")
        if self.set_mood is not None and self.clear_mood:
            raise ValueError("cannot set and clear mood in one patch")
        unknown = sorted(set(self.clear_features) - set(FeatureGoal.NUMERIC_FEATURES))
        if unknown:
            raise ValueError(f"unknown features to clear: {unknown}")
        if len({goal.feature for goal in self.goals}) != len(self.goals):
            raise ValueError("a patch may set at most one goal per feature")
        return self


def merge_goals(
    existing: tuple[FeatureGoal, ...],
    additions: tuple[FeatureGoal, ...],
    clear_features: tuple[str, ...] = (),
) -> tuple[FeatureGoal, ...]:
    """Merge goals deterministically; a newer goal replaces that feature's old one."""
    cleared = set(clear_features)
    by_feature = {goal.feature: goal for goal in existing if goal.feature not in cleared}
    for goal in additions:
        by_feature[goal.feature] = goal
    return tuple(by_feature[feature] for feature in sorted(by_feature))


def apply_intent_patch(base: MusicIntent, patch: IntentPatch) -> MusicIntent:
    """Apply a validated patch atomically; never mutate the base intent."""
    update: dict[str, object] = {
        "feature_goals": merge_goals(
            base.feature_goals, patch.goals, patch.clear_features
        ),
        "source": patch.source,
        "needs_clarification": False,
        "clarification": None,
    }
    if patch.set_genre is not None:
        update["genre"] = patch.set_genre
    elif patch.clear_genre:
        update["genre"] = None
    if patch.set_mood is not None:
        update["mood"] = patch.set_mood
    elif patch.clear_mood:
        update["mood"] = None
    if patch.instrumental_only is not None:
        update["instrumental_only"] = patch.instrumental_only
    if patch.exclude_explicit is not None:
        update["exclude_explicit"] = patch.exclude_explicit
    changed = MusicIntent.model_validate(base.model_dump() | update)
    musical_fields = (
        "genre",
        "mood",
        "instrumental_only",
        "exclude_explicit",
        "feature_goals",
    )
    if all(getattr(changed, field) == getattr(base, field) for field in musical_fields):
        return base
    return changed.model_copy(
        update={
            "query": rebuild_retrieval_query(
                changed, neutral_query_text(base.query, base)
            )
        }
    )


def apply_refinement(base: MusicIntent, key: str) -> MusicIntent:
    """Apply one registry entry by key, rejecting unrecognized UI actions."""
    try:
        refinement = REFINEMENTS[key]
    except KeyError as exc:
        raise ValueError(f"unknown refinement: {key}") from exc
    return apply_intent_patch(base, IntentPatch(goals=(refinement.goal,)))


def remove_intent_facet(base: MusicIntent, facet: str) -> MusicIntent:
    """Remove one visible intent chip by an allowlisted facet/cue identifier."""
    if facet == "genre":
        return apply_intent_patch(base, IntentPatch(clear_genre=True))
    if facet == "mood":
        return apply_intent_patch(base, IntentPatch(clear_mood=True))
    if facet == "instrumental_only":
        return apply_intent_patch(base, IntentPatch(instrumental_only=False))
    if facet == "exclude_explicit":
        return apply_intent_patch(base, IntentPatch(exclude_explicit=False))
    goal = next((goal for goal in base.feature_goals if goal.cue_id == facet), None)
    if goal is None:
        raise ValueError(f"unknown intent facet: {facet}")
    return apply_intent_patch(
        base, IntentPatch(clear_features=(goal.feature,))
    )


def merge_intents(base: MusicIntent, follow_up: MusicIntent) -> MusicIntent:
    """Merge a *guarded and parsed* follow-up into the current intent.

    Recognized follow-up facets replace their counterparts.  Hard filters only
    become stricter from natural language; the controlled console is the explicit
    place to turn them off.  The searchable query is the concatenation of two
    sanitized strings and is never logged or rendered by the UI.
    """
    patch = IntentPatch(
        set_genre=follow_up.genre,
        set_mood=follow_up.mood,
        goals=follow_up.feature_goals,
        instrumental_only=True if follow_up.instrumental_only else None,
        exclude_explicit=True if follow_up.exclude_explicit else None,
        source="refined_rules",
    )
    merged = apply_intent_patch(base, patch)
    return merged.model_copy(
        update={
            "query": rebuild_retrieval_query(
                merged,
                neutral_query_text(base.query, base),
                neutral_query_text(follow_up.query, follow_up),
            )
        }
    )


def combine_guard_categories(
    base: GuardCategory, follow_up: GuardCategory
) -> GuardCategory:
    """Carry the strongest privacy-relevant category across a refinement thread."""
    if follow_up is GuardCategory.HIGH_RISK:
        return GuardCategory.HIGH_RISK
    if GuardCategory.SENSITIVE in (base, follow_up):
        return GuardCategory.SENSITIVE
    if GuardCategory.INJECTION in (base, follow_up):
        return GuardCategory.INJECTION
    # Empty/too-long/high-risk outcomes belong to one attempt; they must not
    # contaminate the next valid refinement. Only privacy/injection categories
    # intentionally carry across an otherwise ordinary turn.
    return follow_up
