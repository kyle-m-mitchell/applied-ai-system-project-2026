"""Typed, request-local Streamlit state and reversible refinement history."""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.contracts import (
    CompanionAction,
    CompanionTurn,
    DiversityLevel,
    ExecutionPolicy,
    FeatureGoal,
    FeatureRelation,
    GuardCategory,
    MusicIntent,
)
from src.refine import combine_guard_categories


MAX_SNAPSHOTS = 25


@dataclass(frozen=True)
class EvolutionEntry:
    """A controlled change summary; it never stores a user's words."""

    changes: tuple[str, ...]
    entered_ids: tuple[int, ...] = ()
    dropped_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class MixSnapshot:
    turn: CompanionTurn
    policy: ExecutionPolicy
    evolution: EvolutionEntry | None = None


@dataclass(frozen=True)
class UiSession:
    """All listener state is confined to one Streamlit WebSocket session."""

    catalog_id: str = "fma"
    snapshots: tuple[MixSnapshot, ...] = ()
    transient: CompanionTurn | None = None
    guard_category: GuardCategory = GuardCategory.OK

    @property
    def current(self) -> MixSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None


def start_session(
    turn: CompanionTurn,
    policy: ExecutionPolicy,
    *,
    catalog_id: str = "fictional",
) -> UiSession:
    return UiSession(
        catalog_id=catalog_id,
        snapshots=(MixSnapshot(turn=turn, policy=policy),),
        guard_category=turn.receipt.guard_category,
    )


def commit_turn(
    state: UiSession,
    turn: CompanionTurn,
    policy: ExecutionPolicy,
    changes: tuple[str, ...],
) -> UiSession:
    """Commit a usable set, or surface a non-result without replacing the last set."""
    # Sensitivity is a privacy lock for the mix. A high-risk transient is handled
    # safely but does not permanently poison an otherwise valid music session.
    guard_category = (
        state.guard_category
        if turn.receipt.guard_category is GuardCategory.HIGH_RISK
        else combine_guard_categories(
            state.guard_category, turn.receipt.guard_category
        )
    )
    if turn.response.action in (
        CompanionAction.SAFE_RESPONSE,
        CompanionAction.CLARIFY,
        CompanionAction.NO_MATCH,
    ):
        return replace(state, transient=turn, guard_category=guard_category)
    before_ids = state.current.turn.receipt.final_ids if state.current else ()
    after_ids = turn.receipt.final_ids
    entry = EvolutionEntry(
        changes=changes or ("Rebuilt the set from the updated intent.",),
        entered_ids=tuple(track_id for track_id in after_ids if track_id not in before_ids),
        dropped_ids=tuple(track_id for track_id in before_ids if track_id not in after_ids),
    )
    snapshots = state.snapshots + (
        MixSnapshot(turn=turn, policy=policy, evolution=entry),
    )
    if len(snapshots) > MAX_SNAPSHOTS:
        snapshots = (snapshots[0],) + snapshots[-(MAX_SNAPSHOTS - 1) :]
    return UiSession(
        catalog_id=state.catalog_id,
        snapshots=snapshots,
        guard_category=guard_category,
    )


def dismiss_transient(state: UiSession) -> UiSession:
    return replace(state, transient=None)


def undo(state: UiSession) -> UiSession:
    if len(state.snapshots) <= 1:
        return state
    return UiSession(
        catalog_id=state.catalog_id,
        snapshots=state.snapshots[:-1],
        guard_category=state.guard_category,
    )


def feature_direction(intent: MusicIntent, feature: str) -> int:
    goal = next((goal for goal in intent.feature_goals if goal.feature == feature), None)
    if goal is None:
        return 0
    if goal.relation is FeatureRelation.PREFER_LOW:
        return -1
    if goal.relation is FeatureRelation.PREFER_HIGH:
        return 1
    return 0


def tempo_target(intent: MusicIntent) -> float | None:
    goal = next(
        (goal for goal in intent.feature_goals if goal.feature == "tempo_bpm"),
        None,
    )
    if goal is None:
        return None
    if goal.target is not None:
        return goal.target
    if goal.low is not None and goal.high is not None:
        return (goal.low + goal.high) / 2
    return None


def current_tempo_goal(intent: MusicIntent) -> FeatureGoal | None:
    """Return the complete tempo rule so the UI never collapses its relation."""
    return next(
        (goal for goal in intent.feature_goals if goal.feature == "tempo_bpm"),
        None,
    )


def describe_intent_delta(
    before: MusicIntent,
    after: MusicIntent,
    *,
    before_diversity: DiversityLevel,
    after_diversity: DiversityLevel,
) -> tuple[str, ...]:
    """Describe only controlled facets and policies—never query text."""
    changes: list[str] = []
    if before.genre != after.genre:
        changes.append(
            f"Genre {'cleared' if after.genre is None else f'set to {after.genre}'}."
        )
    if before.mood != after.mood:
        changes.append(
            f"Mood {'cleared' if after.mood is None else f'set to {after.mood}'}."
        )
    if before.instrumental_only != after.instrumental_only:
        changes.append(
            "Instrumental-only filter enabled."
            if after.instrumental_only
            else "Instrumental-only filter removed."
        )
    if before.exclude_explicit != after.exclude_explicit:
        changes.append(
            "Clean-only filter enabled."
            if after.exclude_explicit
            else "Clean-only filter removed."
        )

    old = {goal.feature: goal for goal in before.feature_goals}
    new = {goal.feature: goal for goal in after.feature_goals}
    feature_names = {
        "energy": "Energy",
        "valence": "Mood tone",
        "danceability": "Movement",
        "acousticness": "Acoustic texture",
        "tempo_bpm": "Tempo",
        "instrumentalness": "Instrumental character",
    }
    relation_names = {
        FeatureRelation.PREFER_LOW: "lower",
        FeatureRelation.PREFER_HIGH: "higher",
        FeatureRelation.NEAR: "near target",
        FeatureRelation.AT_LEAST: "minimum",
        FeatureRelation.AT_MOST: "maximum",
        FeatureRelation.RANGE: "range",
    }
    for feature in sorted(set(old) | set(new)):
        if old.get(feature) == new.get(feature):
            continue
        if feature not in new:
            changes.append(f"{feature_names[feature]} preference cleared.")
            continue
        goal: FeatureGoal = new[feature]
        detail = relation_names[goal.relation]
        if goal.target is not None:
            detail += f" {goal.target:g}"
            if feature == "tempo_bpm":
                detail += " BPM"
        changes.append(f"{feature_names[feature]} moved {detail}.")
    if before_diversity is not after_diversity:
        changes.append(f"Set variety changed to {after_diversity.value}.")
    if before.query != after.query and not changes:
        changes.append("Expanded the searchable music description.")
    return tuple(changes)
