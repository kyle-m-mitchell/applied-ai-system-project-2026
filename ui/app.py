"""Cadence's flagship Streamlit application—thin, typed, and reversible."""

from __future__ import annotations

import streamlit as st

from src.contracts import (
    CatalogDescriptor,
    CompanionAction,
    DiversityLevel,
    ExecutionPolicy,
    FeatureGoal,
    FeatureRelation,
    GuardCategory,
)
from src.research import ResearchOutcome
from src.refine import (
    REFINEMENTS,
    IntentPatch,
    apply_intent_patch,
    apply_refinement,
    directional_goal,
    remove_intent_facet,
    tempo_goal,
)
from ui.components import (
    announce,
    render_action_state,
    render_brand,
    render_context_evidence,
    render_developer_view,
    render_disclosure,
    render_evolution,
    render_first_run_story,
    render_framing,
    render_intent,
    render_mode_badges,
    render_privacy_explainer,
    render_results_anchor,
    render_skip_link,
    render_track_cards,
)
from ui.examples import examples_for_catalog
from ui.runtime import RuntimeBundle, get_runtime
from ui.state import (
    UiSession,
    commit_turn,
    current_tempo_goal,
    describe_intent_delta,
    dismiss_transient,
    feature_direction,
    start_session,
    tempo_target,
    undo,
)
from ui.theme import inject_theme


STATE_KEY = "cadence_ui_session"
LOCAL_KEY = "cadence_local_only"
DEV_KEY = "cadence_developer_view"
PENDING_SEARCH_KEY = "cadence_pending_search"
CATALOG_KEY = "cadence_catalog"
RESEARCH_KEY = "cadence_research_cache"
CATALOG_CHOICES = ("fma", "fictional")


def _state() -> UiSession:
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = UiSession(catalog_id="fma")
    return st.session_state[STATE_KEY]


def _set_state(value: UiSession) -> None:
    st.session_state[STATE_KEY] = value


def _clear_dynamic_widgets(*, include_session_artifacts: bool = False) -> None:
    for key in list(st.session_state):
        if str(key).startswith(
            (
                "console_",
                "followup_",
                "fit_",
                "search_",
                "quick_",
                "research_",
                "catalog_link_",
                "cadence_pending",
            )
        ):
            del st.session_state[key]
    if include_session_artifacts:
        st.session_state[RESEARCH_KEY] = {}


def _research_cache() -> dict[str, ResearchOutcome]:
    cache = st.session_state.get(RESEARCH_KEY)
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[RESEARCH_KEY] = cache
    return cache


def _catalog_change_callback() -> None:
    """Treat a catalog switch as a hard namespace boundary for listener state."""
    selected = st.session_state.get(CATALOG_KEY, "fma")
    if selected not in CATALOG_CHOICES:
        selected = "fma"
        st.session_state[CATALOG_KEY] = selected
    _set_state(UiSession(catalog_id=selected))
    _clear_dynamic_widgets(include_session_artifacts=True)


def _undo_callback() -> None:
    state = undo(_state())
    _set_state(state)
    _clear_dynamic_widgets()
    if state.current is not None:
        st.session_state[LOCAL_KEY] = state.current.policy.force_local


def _reset_callback() -> None:
    selected = st.session_state.get(CATALOG_KEY, "fma")
    _set_state(UiSession(catalog_id=selected))
    _clear_dynamic_widgets(include_session_artifacts=True)


def _dismiss_callback() -> None:
    _set_state(dismiss_transient(_state()))


def _capture_and_clear(widget_key: str, pending_key: str) -> None:
    """Move submitted text into an ephemeral transaction slot, then clear input."""
    st.session_state[pending_key] = st.session_state.get(widget_key, "")
    st.session_state[widget_key] = ""


def _policy(local_only: bool, diversity: DiversityLevel) -> ExecutionPolicy:
    return ExecutionPolicy(force_local=local_only, diversity=diversity)


def _run_initial(bundle: RuntimeBundle, prompt: str, local_only: bool) -> None:
    selected = _policy(local_only, DiversityLevel.BALANCED)
    with st.spinner("Cadence is checking the request and building a grounded set…"):
        turn = bundle.companion.respond_detailed(prompt, policy=selected)
    _clear_dynamic_widgets()
    state = _state()
    if state.current is not None and turn.response.action in (
        CompanionAction.SAFE_RESPONSE,
        CompanionAction.CLARIFY,
        CompanionAction.NO_MATCH,
    ):
        # A failed replacement is an attempted transaction, not permission to
        # destroy a working set. Surface it above the current mix just like a
        # failed refinement; a successful new direction still starts fresh.
        _set_state(
            commit_turn(
                state,
                turn,
                selected,
                ("Tried a different mix; kept the previous grounded set.",),
            )
        )
    else:
        _set_state(start_session(turn, selected, catalog_id=bundle.catalog_id))
    st.rerun()


def _commit_refinement(
    bundle: RuntimeBundle,
    new_intent,
    policy: ExecutionPolicy,
    changes: tuple[str, ...],
) -> None:
    state = _state()
    current = state.current
    if current is None:
        return
    category = state.guard_category
    with st.spinner("Remixing through the same guarded recommendation pipeline…"):
        turn = bundle.companion.respond_with_intent_detailed(
            new_intent,
            category=category,
            policy=policy,
        )
    _clear_dynamic_widgets()
    _set_state(commit_turn(state, turn, policy, changes))
    st.rerun()


def _render_search(bundle: RuntimeBundle, local_only: bool, *, first_run: bool) -> None:
    heading = "Describe a moment" if first_run else "Start a new direction"
    placeholder = (
        (
            "e.g. calm independent electronic music for focused writing"
            if bundle.catalog_id == "fma"
            else "e.g. clean instrumental focus music, calm and acoustic"
        )
        if first_run
        else "Describe a completely new mix…"
    )
    with st.form("search_form", clear_on_submit=True):
        prompt = st.text_input(
            heading,
            placeholder=placeholder,
            max_chars=400,
            key="search_prompt",
            help="Raw text is guarded, then cleared after submission. It is not logged or placed in the URL.",
        )
        submitted = st.form_submit_button(
            "Build my set" if first_run else "Build a new set",
            type="primary",
            icon=":material/graphic_eq:",
            width="stretch",
            on_click=_capture_and_clear,
            args=("search_prompt", PENDING_SEARCH_KEY),
        )
    if submitted:
        _run_initial(
            bundle,
            st.session_state.pop(PENDING_SEARCH_KEY, prompt),
            local_only,
        )


def _render_examples(bundle: RuntimeBundle, local_only: bool) -> None:
    st.caption("Or begin with a designed example")
    with st.container(horizontal=True, gap="small", key="example_prompts"):
        for example in examples_for_catalog(bundle.catalog_id):
            if st.button(
                example.label,
                key=f"example_{example.key}",
                help=example.description,
                type="secondary",
            ):
                _run_initial(bundle, example.prompt, local_only)


def _setting(value: int, options: tuple[str, str, str]) -> str:
    return options[value + 1]


def _direction(value: str | None, options: tuple[str, str, str]) -> int:
    if value is None:  # defense in depth; controls are also required in the UI
        return 0
    return options.index(value) - 1


def _render_quick_moves(bundle: RuntimeBundle, local_only: bool) -> None:
    state = _state()
    current = state.current
    if current is None or current.turn.response.intent is None:
        return
    st.markdown('<h2 class="cadence-eyebrow">Quick moves</h2>', unsafe_allow_html=True)
    with st.container(horizontal=True, gap="small", key="quick_moves"):
        for key, refinement in REFINEMENTS.items():
            if st.button(
                refinement.label,
                key=f"quick_{current.turn.receipt.request_id}_{key}",
                help=refinement.description,
                type="tertiary",
            ):
                before = current.turn.response.intent
                after = apply_refinement(before, key)
                policy = _policy(local_only, current.policy.diversity)
                changes = describe_intent_delta(
                    before,
                    after,
                    before_diversity=current.policy.diversity,
                    after_diversity=policy.diversity,
                )
                if not changes:
                    st.toast(
                        "That preference is already active.",
                        icon=":material/check:",
                    )
                    return
                _commit_refinement(bundle, after, policy, changes)


def _remove_facet(bundle: RuntimeBundle, local_only: bool, facet: str) -> None:
    """Drop one interpreted soft preference and rebuild through the guarded pipeline."""
    state = _state()
    current = state.current
    if current is None or current.turn.response.intent is None:
        return
    before = current.turn.response.intent
    try:
        after = remove_intent_facet(before, facet)
    except ValueError:
        st.toast("That preference is no longer active.", icon=":material/check:")
        return
    policy = _policy(local_only, current.policy.diversity)
    changes = describe_intent_delta(
        before,
        after,
        before_diversity=current.policy.diversity,
        after_diversity=policy.diversity,
    )
    if not changes:
        st.toast("That preference is already gone.", icon=":material/check:")
        return
    _commit_refinement(bundle, after, policy, changes)


def _supports_filter(descriptor: CatalogDescriptor | None, name: str) -> bool:
    """A missing descriptor is the legacy fictional implementation."""
    return descriptor is None or descriptor.capabilities.supports_filter(name)


def _supports_feature(descriptor: CatalogDescriptor | None, name: str) -> bool:
    return descriptor is None or name in descriptor.capabilities.supported_features


def _render_taste_console(bundle: RuntimeBundle, local_only: bool) -> None:
    state = _state()
    current = state.current
    if current is None or current.turn.response.intent is None:
        return
    intent = current.turn.response.intent
    descriptor = bundle.catalog_descriptor
    supports_instrumental_filter = _supports_filter(descriptor, "instrumental_only")
    supports_clean_filter = _supports_filter(descriptor, "exclude_explicit")
    supports_instrumentalness = _supports_feature(descriptor, "instrumentalness")
    token = current.turn.receipt.request_id[:10]
    st.markdown("## Taste Console")
    st.caption(
        "Soft preferences reorder eligible tracks. Must-have filters remove tracks. "
        "Changes run once when you press Remix—not while you touch a control."
    )

    with st.form(f"console_form_{token}"):
        st.markdown("**Soft preferences**")
        energy_options = ("Less", "Any", "More")
        tone_options = ("Darker", "Any", "Brighter")
        movement_options = ("Less", "Any", "More")
        texture_options = ("Less acoustic", "Any", "More acoustic")
        energy = st.segmented_control(
            "Energy",
            energy_options,
            default=_setting(feature_direction(intent, "energy"), energy_options),
            key=f"console_energy_{token}",
            required=True,
            width="stretch",
        )
        tone = st.segmented_control(
            "Mood tone",
            tone_options,
            default=_setting(feature_direction(intent, "valence"), tone_options),
            key=f"console_tone_{token}",
            required=True,
            help="The catalog's technical field is valence.",
            width="stretch",
        )
        movement = st.segmented_control(
            "Movement",
            movement_options,
            default=_setting(
                feature_direction(intent, "danceability"), movement_options
            ),
            key=f"console_movement_{token}",
            required=True,
            help="The catalog's technical field is danceability.",
            width="stretch",
        )
        texture = st.segmented_control(
            "Acoustic texture",
            texture_options,
            default=_setting(
                feature_direction(intent, "acousticness"), texture_options
            ),
            key=f"console_texture_{token}",
            required=True,
            help=(
                "This is the catalog's acousticness proxy. Low acousticness does "
                "not necessarily mean electronic music."
            ),
            width="stretch",
        )
        instrumental_character = None
        instrumental_options = ("Less", "Any", "More")
        if supports_instrumentalness:
            instrumental_character = st.segmented_control(
                "Instrumental character",
                instrumental_options,
                default=_setting(
                    feature_direction(intent, "instrumentalness"),
                    instrumental_options,
                ),
                key=f"console_instrumentalness_{token}",
                required=True,
                help=(
                    "A soft instrumentalness preference where a trustworthy value "
                    "exists. It does not claim a track is verified instrumental-only."
                ),
                width="stretch",
            )

        existing_tempo = tempo_target(intent)
        tempo_enabled = st.toggle(
            "Set a tempo target",
            value=existing_tempo is not None,
            key=f"console_tempo_enabled_{token}",
        )
        tempo = st.slider(
            "Tempo (BPM)",
            50,
            200,
            int(round(existing_tempo or 120)),
            disabled=not tempo_enabled,
            key=f"console_tempo_{token}",
        )
        original_tempo_rule = current_tempo_goal(intent)
        if original_tempo_rule is not None and original_tempo_rule.relation.value != "near":
            st.caption(
                "The original tempo rule is preserved until you move the slider. "
                f"Current rule: {original_tempo_rule.relation.value.replace('_', ' ')}."
            )

        st.markdown("**Must-have filters**")
        instrumental = st.toggle(
            "Instrumental only",
            value=intent.instrumental_only,
            key=f"console_instrumental_{token}",
            disabled=not supports_instrumental_filter,
            help=(
                "This catalog can verify the instrumental boolean."
                if supports_instrumental_filter
                else "Unavailable: this catalog cannot prove instrumental-only status. Use the soft instrumental-character control instead."
            ),
        )
        clean = st.toggle(
            "Clean only",
            value=intent.exclude_explicit,
            key=f"console_clean_{token}",
            disabled=not supports_clean_filter,
            help=(
                "This catalog can verify explicit-content status."
                if supports_clean_filter
                else "Unavailable: this catalog cannot verify lyric cleanliness, so Cadence will not guess."
            ),
        )
        unsupported: list[str] = []
        if not supports_instrumental_filter:
            unsupported.append("instrumental-only")
        if not supports_clean_filter:
            unsupported.append("clean-only")
        if unsupported:
            st.caption(
                "Unavailable for this catalog: "
                + " and ".join(unsupported)
                + ". Unknown is not treated as false."
            )

        st.markdown("**Set variety**")
        variety_options = ("Focused", "Balanced", "Exploratory")
        variety_default = current.policy.diversity.value.title()
        variety = st.segmented_control(
            "Relevance ↔ variety",
            variety_options,
            default=variety_default,
            key=f"console_variety_{token}",
            required=True,
            help="Changes MMR variety—not popularity or familiarity. The relevance floor stays fixed.",
            width="stretch",
        )
        remixed = st.form_submit_button(
            "Remix this set",
            type="primary",
            icon=":material/tune:",
            width="stretch",
        )

    if not remixed:
        return

    directions = {
        "energy": _direction(energy, energy_options),
        "valence": _direction(tone, tone_options),
        "danceability": _direction(movement, movement_options),
        "acousticness": _direction(texture, texture_options),
    }
    if supports_instrumentalness:
        directions["instrumentalness"] = _direction(
            instrumental_character, instrumental_options
        )
    goals: list[FeatureGoal] = []
    for feature, direction in directions.items():
        if feature == "instrumentalness":
            goal = (
                None
                if direction == 0
                else FeatureGoal(
                    feature=feature,
                    relation=(
                        FeatureRelation.PREFER_LOW
                        if direction < 0
                        else FeatureRelation.PREFER_HIGH
                    ),
                    cue_id=(
                        "instrumentalness_prefer_low_v1"
                        if direction < 0
                        else "instrumentalness_prefer_high_v1"
                    ),
                )
            )
        else:
            goal = directional_goal(feature, direction)
        if goal is not None:
            goals.append(goal)
    if tempo_enabled:
        if (
            original_tempo_rule is not None
            and existing_tempo is not None
            and int(tempo) == int(round(existing_tempo))
        ):
            goals.append(original_tempo_rule)
        else:
            goals.append(tempo_goal(float(tempo)))

    patch = IntentPatch(
        goals=tuple(goals),
        clear_features=tuple(FeatureGoal.NUMERIC_FEATURES),
        instrumental_only=(
            instrumental if supports_instrumental_filter else intent.instrumental_only
        ),
        exclude_explicit=clean if supports_clean_filter else intent.exclude_explicit,
    )
    after = apply_intent_patch(intent, patch)
    selected_diversity = (
        DiversityLevel(variety.lower()) if variety is not None else current.policy.diversity
    )
    policy = _policy(local_only, selected_diversity)
    changes = list(
        describe_intent_delta(
            intent,
            after,
            before_diversity=current.policy.diversity,
            after_diversity=policy.diversity,
        )
    )
    if current.policy.force_local != policy.force_local:
        changes.append(
            "Provider-free execution enabled."
            if policy.force_local
            else "Standard provider policy enabled."
        )
    if not changes:
        st.toast("That recipe already matches the current set.", icon=":material/check:")
        return
    _commit_refinement(bundle, after, policy, tuple(changes))


def _render_follow_up(bundle: RuntimeBundle, local_only: bool) -> None:
    state = _state()
    current = state.current
    if current is None or current.turn.response.intent is None:
        return
    token = current.turn.receipt.request_id[:10]
    st.markdown("## Refine in words")
    with st.form(f"followup_form_{token}", clear_on_submit=True):
        follow_up = st.text_input(
            "One musical change",
            placeholder="Try: make it calmer and more acoustic",
            max_chars=240,
            key=f"followup_text_{token}",
            help="The follow-up is guarded and then cleared. Evolution stores only controlled changes.",
        )
        pending_key = f"cadence_pending_followup_{token}"
        submitted = st.form_submit_button(
            "Apply refinement",
            icon=":material/arrow_forward:",
            width="stretch",
            on_click=_capture_and_clear,
            args=(f"followup_text_{token}", pending_key),
        )
    if not submitted:
        return

    follow_up = st.session_state.pop(pending_key, follow_up)
    policy = _policy(local_only, current.policy.diversity)
    before = current.turn.response.intent
    with st.spinner("Guarding and interpreting that refinement…"):
        turn = bundle.companion.refine_detailed(
            before,
            follow_up,
            base_category=state.guard_category,
            policy=policy,
        )
    after = turn.response.intent or before
    changes = describe_intent_delta(
        before,
        after,
        before_diversity=current.policy.diversity,
        after_diversity=policy.diversity,
    )
    _clear_dynamic_widgets()
    _set_state(commit_turn(state, turn, policy, changes))
    st.rerun()


def _render_transient(developer: bool) -> bool:
    state = _state()
    if state.transient is None:
        return False
    st.markdown('<h2 class="cadence-eyebrow">Latest follow-up</h2>', unsafe_allow_html=True)
    render_action_state(state.transient)
    render_mode_badges(state.transient)
    if developer:
        render_developer_view(
            state.transient, title="Under the hood — latest follow-up"
        )
    if state.transient.response.action is CompanionAction.SAFE_RESPONSE:
        st.button(
            "Return to the current music set",
            on_click=_dismiss_callback,
            type="primary",
        )
        return True
    st.button("Dismiss", on_click=_dismiss_callback)
    return False


def _announce_turn(turn) -> None:
    """Politely announce the outcome for assistive tech after a rerun."""
    response = turn.response
    action = response.action
    if action in (CompanionAction.RECOMMEND, CompanionAction.DEGRADED) and response.retrieval:
        announce(f"New set ready with {len(response.retrieval.hits)} tracks.")
    elif action is CompanionAction.CLARIFY:
        announce("Cadence needs one clarification to continue.")
    elif action is CompanionAction.NO_MATCH:
        announce("No confident match; your previous set is preserved.")
    elif action is CompanionAction.SAFE_RESPONSE:
        announce("A safety response is shown; no music set was built.")


def _render_results(bundle: RuntimeBundle, local_only: bool, developer: bool) -> None:
    state = _state()
    current = state.current
    if current is None:
        return
    turn = current.turn
    response = turn.response

    if _render_transient(developer):
        return

    render_results_anchor()
    _announce_turn(turn)
    render_action_state(turn)
    render_mode_badges(turn)
    if response.action in (
        CompanionAction.SAFE_RESPONSE,
        CompanionAction.CLARIFY,
        CompanionAction.NO_MATCH,
    ):
        if response.action is CompanionAction.SAFE_RESPONSE:
            st.button("Start over", on_click=_reset_callback, type="primary")
        if developer:
            render_developer_view(turn)
        return

    if response.intent is not None:
        render_intent(
            response.intent,
            token=turn.receipt.request_id[:10],
            on_remove=lambda facet: _remove_facet(bundle, local_only, facet),
        )
    render_context_evidence(turn)
    _render_quick_moves(bundle, local_only)

    results_col, console_col = st.columns([1.55, 1], gap="large")
    with results_col:
        st.markdown(
            '<h2 class="cadence-eyebrow">A set for this moment</h2>',
            unsafe_allow_html=True,
        )
        render_framing(turn)
        research_agent = (
            bundle.provider_free_research_agent if local_only else bundle.research_agent
        )
        render_track_cards(
            turn,
            catalog_descriptor=bundle.catalog_descriptor,
            developer=developer,
            research_agent=research_agent,
            research_cache=_research_cache(),
        )
    with console_col:
        _render_taste_console(bundle, local_only)

    _render_follow_up(bundle, local_only)

    controls_left, controls_right = st.columns([1, 1])
    controls_left.button(
        "Undo last change",
        disabled=len(state.snapshots) <= 1,
        on_click=_undo_callback,
        icon=":material/undo:",
        width="stretch",
    )
    controls_right.button(
        "Reset session",
        on_click=_reset_callback,
        icon=":material/restart_alt:",
        width="stretch",
    )
    render_evolution(state)

    st.markdown("## Did this set fit?")
    rating = st.feedback("faces", key=f"fit_{turn.receipt.request_id}")
    if rating is not None:
        labels = ("Missed", "Not quite", "Close", "Good", "Nailed it")
        st.caption(f"Session-only rating: {labels[rating]}. It does not train a model.")

    if developer:
        render_developer_view(turn)


def run() -> None:
    st.set_page_config(
        page_title="Cadence · Transparent music discovery",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "About": "Cadence is a privacy-first, explainable music-discovery companion."
        },
    )
    inject_theme()
    render_skip_link()

    # Cadence does not currently support user-authored share URLs. Discard any
    # parameters rather than letting free text linger in history/referrers.
    if len(st.query_params):
        st.query_params.clear()

    if CATALOG_KEY not in st.session_state:
        st.session_state[CATALOG_KEY] = "fma"
    selected_catalog = st.session_state[CATALOG_KEY]
    if selected_catalog not in CATALOG_CHOICES:
        selected_catalog = "fma"
        st.session_state[CATALOG_KEY] = selected_catalog

    state = _state()
    if state.catalog_id != selected_catalog:
        _set_state(UiSession(catalog_id=selected_catalog))
        _clear_dynamic_widgets(include_session_artifacts=True)
        state = _state()
    first_run = state.current is None
    if LOCAL_KEY not in st.session_state:
        st.session_state[LOCAL_KEY] = True
    if DEV_KEY not in st.session_state:
        st.session_state[DEV_KEY] = False
    privacy_locked = state.guard_category is GuardCategory.SENSITIVE
    if privacy_locked:
        # Make the visible control agree with the backend's monotonic lock.
        st.session_state[LOCAL_KEY] = True

    active_turn = state.transient or (state.current.turn if state.current else None)
    paused = bool(
        active_turn
        and active_turn.response.action is CompanionAction.SAFE_RESPONSE
    )
    brand_col, controls_col = st.columns([2.5, 1], gap="large", vertical_alignment="top")
    with brand_col:
        render_brand(compact=not first_run, paused=paused)
    with controls_col:
        st.selectbox(
            "Music catalog",
            CATALOG_CHOICES,
            key=CATALOG_KEY,
            format_func=lambda value: "FMA" if value == "fma" else "Fictional",
            on_change=_catalog_change_callback,
            help=(
                "FMA is selected initially. Full and Lite are verified editions of "
                "the same FMA catalog, chosen automatically by artifact availability."
            ),
        )
        local_only = st.toggle(
            "Local-only",
            key=LOCAL_KEY,
            disabled=privacy_locked,
            help="Blocks AI-provider calls for submitted turns. Hosted use still reaches the app server.",
        )
        developer = st.toggle(
            "Developer view",
            key=DEV_KEY,
            help="Shows request-local IDs, timings, provenance, and fingerprints—never prompt text.",
        )
        render_privacy_explainer()
        if privacy_locked:
            st.warning(
                "Privacy lock active for this mix. Provider calls remain blocked "
                "through refinements and undo; reset or start a new mix to clear it.",
                icon=":material/lock:",
            )
        if state.current is not None and local_only != state.current.turn.receipt.force_local:
            prior = "local-only" if state.current.turn.receipt.force_local else "standard"
            upcoming = "local-only" if local_only else "standard"
            st.info(
                f"This set used the {prior} policy. {upcoming.title()} applies to "
                "the next submitted turn.",
                icon=":material/schedule:",
            )

    try:
        bundle = get_runtime(selected_catalog)
    except Exception as exc:  # a broken catalog/config is a startup state, not a traceback
        st.error(
            "Cadence could not load the selected verified catalog. Choose the other "
            "catalog above or retry after restoring its artifact.",
            icon=":material/error:",
        )
        if developer:
            st.exception(exc)
        st.button("Retry", on_click=get_runtime.clear)
        return

    if bundle.provider_configured:
        st.caption("Cloud assist is configured but used only when local-only is off.")
    else:
        st.caption("No cloud key detected; cached/local recommendation paths remain usable.")

    if first_run:
        _render_search(bundle, local_only, first_run=True)
        _render_examples(bundle, local_only)
        render_first_run_story()
        render_disclosure(
            bundle.catalog_descriptor,
            artifact_source=bundle.catalog_artifact_source,
            warnings=bundle.catalog_warnings,
        )
        return

    _render_results(bundle, local_only, developer)
    with st.expander("Start a different mix", icon=":material/add_circle:"):
        _render_search(bundle, local_only, first_run=False)
    st.divider()
    render_disclosure(
        bundle.catalog_descriptor,
        artifact_source=bundle.catalog_artifact_source,
        warnings=bundle.catalog_warnings,
    )
