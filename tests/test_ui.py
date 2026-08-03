"""Headless functional checks for Cadence's Streamlit flagship UI."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.contracts import (
    CompanionAction,
    EmbeddingSource,
    ExecutionPolicy,
    FeatureRelation,
    GuardCategory,
    OperatingMode,
    ResearchBrief,
    ResearchCitation,
    ResearchClaim,
    ResearchStatus,
)
from src.research import ResearchOutcome
from ui.runtime import get_runtime


@pytest.fixture(autouse=True)
def offline_runtime(monkeypatch):
    """The UI suite must never discover or call a developer's real provider key."""
    monkeypatch.setenv("CADENCE_DISABLE_PROVIDER", "1")
    get_runtime.clear()
    yield
    get_runtime.clear()


def _new_app(*, catalog: str | None = "fictional") -> AppTest:
    app = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
    assert not app.exception
    if catalog is not None and app.selectbox[0].value != catalog:
        app.selectbox[0].set_value(catalog)
        app.run()
        assert not app.exception
    return app


def _start(
    query: str,
    *,
    local_only: bool = True,
    catalog: str = "fictional",
) -> AppTest:
    app = _new_app(catalog=catalog)
    if not local_only:
        app.toggle[0].set_value(False)
        app.run()
    app.text_input[0].set_value(query)
    next(button for button in app.button if button.label == "Build my set").click()
    app.run()
    assert not app.exception
    return app


def _session(app: AppTest):
    return app.session_state["cadence_ui_session"]


def _input(app: AppTest, label: str):
    return next(field for field in app.text_input if field.label == label)


def _current_button(app: AppTest, label: str):
    """AppTest can retain pre-rerun elements; the last match is the live widget."""
    return [button for button in app.button if button.label == label][-1]


def _rendered_text(app: AppTest) -> str:
    parts: list[str] = []
    for collection in (
        app.markdown,
        app.caption,
        app.info,
        app.warning,
        app.error,
        app.code,
        app.text,
    ):
        parts.extend(str(getattr(element, "value", "")) for element in collection)
    parts.extend(element.value for element in app.text_input)
    return "\n".join(parts)


def test_fma_is_the_initial_catalog_and_discloses_the_concrete_edition():
    app = _new_app(catalog=None)
    text = _rendered_text(app)

    assert app.selectbox[0].label == "Music catalog"
    assert app.selectbox[0].value == "fma"
    assert _session(app).catalog_id == "fma"
    assert "FMA Lite" in text
    assert "verified 300-track artifact" in text
    assert "unknown values stay unknown" in text


def test_fma_cards_and_console_render_unknowns_and_capabilities_honestly():
    app = _start("calm folk music", catalog="fma")
    state = _session(app)
    text = _rendered_text(app)
    tracks = tuple(hit.track for hit in state.current.turn.response.retrieval.hits)

    assert state.catalog_id == "fma"
    assert all(track.catalog_id == "fma" for track in tracks)
    assert "FMA Lite" in text
    assert "experimental" in text.lower()
    assert "Instrumental character" in [item.label for item in app.segmented_control]
    assert next(item for item in app.toggle if item.label == "Instrumental only").disabled
    assert next(item for item in app.toggle if item.label == "Clean only").disabled
    assert "Clean confirmed" not in text
    assert "Instrumental confirmed" not in text
    assert ">None<" not in text and "License\nNone" not in text


def test_catalog_switch_clears_mix_undo_rating_research_and_dynamic_state():
    app = _start("calm folk music", catalog="fma")
    request_id = _session(app).current.turn.receipt.request_id
    app.session_state["cadence_research_cache"] = {"catalog:fma:1": object()}
    app.session_state[f"fit_{request_id}"] = 4
    app.session_state["console_test_draft"] = "stale"

    app.selectbox[0].set_value("fictional")
    app.run()

    state = _session(app)
    stored = app.session_state.filtered_state
    assert not app.exception
    assert state.catalog_id == "fictional"
    assert state.snapshots == () and state.transient is None
    assert stored["cadence_research_cache"] == {}
    assert f"fit_{request_id}" not in stored
    assert "console_test_draft" not in stored


def test_cached_research_claims_render_with_citations_and_a_sanitized_trace():
    app = _start("calm folk music", catalog="fma")
    track = _session(app).current.turn.response.retrieval.hits[0].track
    citation = ResearchCitation(
        citation_id="source-1",
        title="Verified source",
        url="https://musicbrainz.org/recording/example",
        source_domain="musicbrainz.org",
    )
    brief = ResearchBrief(
        track_ref=track.ref,
        status=ResearchStatus.PUBLISHED,
        identity_confidence=1.0,
        claims=(
            ResearchClaim(
                text="A citation-backed session-only fact.",
                citation_ids=(citation.citation_id,),
            ),
        ),
        citations=(citation,),
        source_domains=(citation.source_domain,),
        provider="fixture",
        model_id="fixture-v1",
        timestamp="2026-08-02T00:00:00+00:00",
    )
    app.session_state["cadence_research_cache"] = {
        track.ref.source_id: ResearchOutcome(
            brief=brief,
            trace=(
                "local recommendation complete",
                "research requested",
                "identity resolved",
                "citations validated",
                "brief published",
            ),
        )
    }

    app.run()
    text = _rendered_text(app)
    assert not app.exception
    assert "A citation-backed session-only fact." in text
    assert "musicbrainz.org" in text
    assert "Research did not alter eligibility, ranking, or catalog fields" in text


def test_first_run_and_normal_ids_match_the_public_service():
    app = _start("some jazz please")
    snapshot = _session(app).current
    direct = get_runtime("fictional").companion.respond_detailed(
        "some jazz please", policy=ExecutionPolicy(force_local=True)
    )

    assert snapshot.turn.receipt.final_ids == direct.receipt.final_ids
    assert snapshot.turn.response.action is CompanionAction.RECOMMEND
    assert all(field.value == "" for field in app.text_input)  # submitted text cleared
    assert app.query_params == {}
    assert sum("cadence-title" in block.value for block in app.markdown) >= 5


@pytest.mark.parametrize(
    "query, action",
    [
        ("   ", CompanionAction.CLARIFY),  # only genuinely-empty input clarifies now
        ("i want to end my life", CompanionAction.SAFE_RESPONSE),
    ],
)
def test_graceful_non_recommendation_states(query, action):
    app = _start(query)
    turn = _session(app).current.turn
    assert turn.response.action is action
    assert not app.exception
    if action is CompanionAction.SAFE_RESPONSE:
        assert turn.response.retrieval is None
        assert not any(
            '<div class="cadence-track-head">' in block.value for block in app.markdown
        )


@pytest.mark.parametrize("query", ["music", "surprise me", "xyzzy zzz qqq"])
def test_vague_and_open_requests_get_an_honest_best_effort_set(query):
    app = _start(query)
    turn = _session(app).current.turn
    assert not app.exception
    assert turn.response.action is CompanionAction.RECOMMEND
    assert len(turn.response.retrieval.hits) >= 1


def test_pii_is_not_echoed_and_sensitive_turn_is_provider_free():
    raw = "alice@example.com"
    app = _start(f"my email is {raw}, find me melancholy piano", local_only=False)
    turn = _session(app).current.turn
    blob = _rendered_text(app) + turn.receipt.model_dump_json()

    assert turn.receipt.guard_category is GuardCategory.SENSITIVE
    assert turn.receipt.force_local and not turn.receipt.network_used
    assert raw not in blob
    assert raw not in str(app.query_params)
    assert all(raw not in field.value for field in app.text_input)


def test_uncached_standard_query_degrades_honestly_without_a_key():
    app = _start(
        "gentle acoustic songs for writing beside the window",
        local_only=False,
    )
    turn = _session(app).current.turn
    assert turn.response.action is CompanionAction.DEGRADED
    assert turn.receipt.operating_mode is OperatingMode.DEGRADED
    assert turn.receipt.network_used is False
    assert "Semantic search was unavailable" in _rendered_text(app)


def test_cached_semantic_query_remains_available_in_provider_free_mode():
    app = _start("music to concentrate")
    turn = _session(app).current.turn

    assert turn.response.action is CompanionAction.RECOMMEND
    assert turn.receipt.embedding_source is EmbeddingSource.CACHE
    assert turn.receipt.network_used is False
    assert turn.receipt.force_local is True


def test_failed_refinement_is_transient_and_preserves_the_working_set():
    app = _start("some jazz please")
    original = _session(app).current

    _input(app, "One musical change").set_value("more energy and less energy")
    next(button for button in app.button if button.label == "Apply refinement").click()
    app.run()
    state = _session(app)

    assert state.transient is not None
    assert state.transient.response.action is CompanionAction.CLARIFY
    assert state.current.turn.receipt.request_id == original.turn.receipt.request_id
    assert state.current.turn.receipt.final_ids == original.turn.receipt.final_ids


def test_failed_new_mix_is_transient_and_preserves_the_working_set():
    app = _start("some jazz please")
    original = _session(app).current

    # Gibberish now yields an exploratory set, so a *genuinely* failed replacement
    # is an empty one — it must stay transient and preserve the working mix.
    _input(app, "Start a new direction").set_value("   ")
    next(button for button in app.button if button.label == "Build a new set").click()
    app.run()
    state = _session(app)

    assert state.transient is not None
    assert state.transient.response.action is CompanionAction.CLARIFY
    assert len(state.snapshots) == 1
    assert state.current.turn.receipt.request_id == original.turn.receipt.request_id
    assert state.current.turn.receipt.final_ids == original.turn.receipt.final_ids


def test_unsupported_multiword_follow_up_does_not_create_fake_evolution():
    app = _start("some jazz please")
    original = _session(app).current

    _input(app, "One musical change").set_value("make it warmer please")
    next(button for button in app.button if button.label == "Apply refinement").click()
    app.run()
    state = _session(app)

    assert state.transient is not None
    assert state.transient.response.action is CompanionAction.CLARIFY
    assert len(state.snapshots) == 1
    assert state.current.turn.receipt.request_id == original.turn.receipt.request_id


def test_console_preserves_non_near_tempo_rule_when_another_control_changes():
    app = _start("jazz at least 130 bpm")
    app.segmented_control[0].set_value("More")
    app.run()
    next(button for button in app.button if button.label == "Remix this set").click()
    app.run()

    tempo = next(
        goal
        for goal in _session(app).current.turn.response.intent.feature_goals
        if goal.feature == "tempo_bpm"
    )
    assert tempo.relation is FeatureRelation.AT_LEAST
    assert tempo.target == 130.0


def test_removing_an_intent_chip_rebuilds_without_it_and_undo_restores():
    app = _start("calm acoustic jazz for the evening")
    original = _session(app).current
    assert original.turn.response.intent.genre == "jazz"

    # The removable genre chip is a button whose label is the badge text.
    _current_button(app, "Genre · jazz").click()
    app.run()
    changed = _session(app)
    assert not app.exception
    assert len(changed.snapshots) == 2
    assert changed.current.turn.response.intent.genre is None
    assert "Genre cleared" in " ".join(changed.current.evolution.changes)

    next(button for button in app.button if button.label == "Undo last change").click()
    app.run()
    restored = _session(app)
    assert len(restored.snapshots) == 1
    assert restored.current.turn.response.intent.genre == "jazz"
    assert restored.current.turn.receipt.final_ids == original.turn.receipt.final_ids


def test_removing_a_chip_keeps_a_sticky_privacy_lock():
    app = _start("my email is chip@example.com, high energy jazz", local_only=False)
    assert _session(app).current.turn.receipt.guard_category is GuardCategory.SENSITIVE

    _current_button(app, "Genre · jazz").click()
    app.run()
    later = _session(app).current.turn
    assert later.receipt.guard_category is GuardCategory.SENSITIVE
    assert later.receipt.force_local and not later.receipt.network_used
    assert "chip@example.com" not in _rendered_text(app)


def test_repeating_an_active_quick_move_is_a_true_no_op():
    app = _start("high energy workout")
    original = _session(app).current.turn.receipt.request_id

    next(button for button in app.button if button.label == "More energy").click()
    app.run()

    state = _session(app)
    assert len(state.snapshots) == 1
    assert state.current.turn.receipt.request_id == original


def test_console_deselection_cannot_crash_or_submit_an_ambiguous_value():
    app = _start("some jazz please")
    app.segmented_control[0].set_value(None)
    app.run()
    next(button for button in app.button if button.label == "Remix this set").click()
    app.run()
    assert not app.exception
    assert len(_session(app).snapshots) == 1


def test_quick_refine_and_undo_restore_the_exact_snapshot():
    app = _start("high energy workout")
    initial = _session(app).current

    next(button for button in app.button if button.label == "Calmer").click()
    app.run()
    changed = _session(app)
    energy = next(
        goal
        for goal in changed.current.turn.response.intent.feature_goals
        if goal.feature == "energy"
    )
    assert len(changed.snapshots) == 2
    assert energy.relation is FeatureRelation.PREFER_LOW

    next(button for button in app.button if button.label == "Undo last change").click()
    app.run()
    restored = _session(app)
    assert len(restored.snapshots) == 1
    assert restored.current.turn.receipt.request_id == initial.turn.receipt.request_id
    assert restored.current.turn.receipt.final_ids == initial.turn.receipt.final_ids


def test_console_is_transactional_and_records_one_controlled_change():
    app = _start("upbeat party music")
    original_request = _session(app).current.turn.receipt.request_id

    # A widget rerun is presentation/draft state only; it must not call the engine.
    app.segmented_control[0].set_value("Less")
    app.segmented_control[4].set_value("Exploratory")
    app.run()
    assert len(_session(app).snapshots) == 1
    assert _session(app).current.turn.receipt.request_id == original_request

    next(button for button in app.button if button.label == "Remix this set").click()
    app.run()
    state = _session(app)
    assert len(state.snapshots) == 2
    assert state.current.policy.diversity.value == "exploratory"
    assert state.current.turn.receipt.request_id != original_request
    changes = " ".join(state.current.evolution.changes)
    assert "Energy moved lower" in changes
    assert "variety changed to exploratory" in changes


def test_guarded_follow_up_clears_text_and_sensitive_state_stays_sticky():
    app = _start("upbeat party music", local_only=False)
    _input(app, "One musical change").set_value(
        "my email is refine@example.com, make it calmer and more acoustic"
    )
    next(button for button in app.button if button.label == "Apply refinement").click()
    app.run()
    state = _session(app)
    turn = state.current.turn
    assert turn.receipt.guard_category is GuardCategory.SENSITIVE
    assert turn.receipt.force_local and not turn.receipt.network_used
    assert all(field.value == "" for field in app.text_input)
    assert "refine@example.com" not in _rendered_text(app)

    # A later controlled chip cannot downgrade the sticky sensitive category.
    _current_button(app, "Moodier").click()
    app.run()
    later = _session(app).current.turn
    assert later.receipt.guard_category is GuardCategory.SENSITIVE
    assert later.receipt.force_local and not later.receipt.network_used


def test_sensitive_transient_and_undo_cannot_downgrade_the_session_privacy_lock():
    app = _start("upbeat party music", local_only=False)

    # Sensitive but unsupported: the musical change is transient, while the
    # privacy classification must still become sticky.
    _input(app, "One musical change").set_value(
        "my email is lock@example.com, make it warmer please"
    )
    next(button for button in app.button if button.label == "Apply refinement").click()
    app.run()
    assert _session(app).transient.response.action is CompanionAction.CLARIFY
    assert _session(app).guard_category is GuardCategory.SENSITIVE

    _current_button(app, "Moodier").click()
    app.run()
    assert _session(app).current.turn.receipt.force_local

    # Undo restores the musical snapshot, not permission to use a provider.
    next(button for button in app.button if button.label == "Undo last change").click()
    app.run()
    assert _session(app).guard_category is GuardCategory.SENSITIVE
    _current_button(app, "Brighter").click()
    app.run()
    later = _session(app).current.turn
    assert later.receipt.guard_category is GuardCategory.SENSITIVE
    assert later.receipt.force_local and not later.receipt.network_used


def test_sensitive_non_result_still_exposes_request_local_developer_evidence():
    # A bare email redacts to nothing searchable, so this stays a clarify — a
    # sensitive turn with no music result, which must still expose dev evidence.
    app = _start("hidden@example.com", local_only=False)
    assert _session(app).current.turn.response.action is CompanionAction.CLARIFY
    app.toggle[1].set_value(True)
    app.run()
    assert not app.exception
    assert any("request-local receipt" in item.value for item in app.caption)
    assert "hidden@example.com" not in _rendered_text(app)


def test_safe_response_developer_pipeline_shows_the_early_exit_truthfully():
    app = _start("i want to end my life")
    app.toggle[1].set_value(True)
    app.run()
    pipeline = "\n".join(
        block.value for block in app.markdown if "cadence-pipeline" in block.value
    )
    assert "Safe branch" in pipeline
    assert "retrieval skipped" in pipeline
    assert "Retrieve" not in pipeline and "Fusion" not in pipeline


def test_developer_view_shows_the_signal_comparison_only_when_open():
    app = _start("some jazz please")
    # Hidden while the developer toggle is off.
    assert not any("How the pool ranked under each leg" in b.value for b in app.markdown)

    app.toggle[1].set_value(True)
    app.run()
    assert not app.exception
    text = _rendered_text(app)
    assert "How the pool ranked under each leg" in text
    assert "Fused · what Cadence used" in text
    # Structured lifted at least one jazz track that text ranking alone missed.
    assert any("cadence-lift" in b.value for b in app.markdown)


def test_signal_comparison_carries_no_query_text():
    app = _start("my email is see@example.com, some jazz please", local_only=False)
    app.toggle[1].set_value(True)
    app.run()
    comparison = _session(app).current.turn.comparison
    assert comparison is not None and comparison.structured_active
    assert "see@example.com" not in comparison.model_dump_json()
    assert "see@example.com" not in _rendered_text(app)


def test_accessibility_status_and_skip_link_render():
    app = _start("some jazz please")
    blob = "\n".join(block.value for block in app.markdown)
    assert "cadence-skip" in blob  # keyboard skip link to results
    assert 'aria-live="polite"' in blob  # polite screen-reader status region
    assert "New set ready with" in blob


def _toggle(app: AppTest, label: str):
    return next(item for item in app.toggle if item.label == label)


def test_feedback_tap_learns_session_taste_and_records_an_evolution_step():
    app = _start("something upbeat", catalog="fictional")
    before = _session(app).current.turn.receipt.request_id

    _current_button(app, "👍 More like this").click()
    app.run()
    state = _session(app)
    assert not app.exception
    assert len(state.snapshots) == 2
    assert state.preference.is_active  # taste accumulated, session-only
    assert state.current.turn.receipt.request_id != before
    assert any(
        snapshot.evolution and "Learned" in " ".join(snapshot.evolution.changes)
        for snapshot in state.snapshots
    )


def test_dont_learn_toggle_and_clear_learning_are_honored():
    app = _start("something upbeat", catalog="fictional")
    _current_button(app, "👍 More like this").click()
    app.run()
    assert _session(app).preference.is_active

    _current_button(app, "Clear learning").click()
    app.run()
    assert not _session(app).preference.is_active  # cleared, reversible

    _toggle(app, "Learn from my feedback").set_value(False)
    app.run()
    assert _session(app).preference.enabled is False


def test_developer_toggle_is_presentation_only():
    app = _start("some jazz please")
    before = _session(app).current.turn.receipt.request_id
    app.toggle[1].set_value(True)
    app.run()
    assert _session(app).current.turn.receipt.request_id == before
    assert len(_session(app).snapshots) == 1
    assert any("request-local receipt" in item.value for item in app.caption)
