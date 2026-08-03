"""Reusable, evidence-faithful Streamlit components for Cadence."""

from __future__ import annotations

import html
from collections.abc import Callable, Iterable, MutableMapping

import streamlit as st

from src.contracts import (
    CatalogDescriptor,
    CatalogEdition,
    CatalogTrack,
    CompanionAction,
    CompanionTurn,
    EmbeddingSource,
    FieldLineage,
    FieldOrigin,
    GuardCategory,
    MusicIntent,
    OperatingMode,
    RankedCandidate,
    ResearchStatus,
    SignalComparison,
    VoiceSource,
)
from src.research import ResearchOutcome, TrackResearchAgent
from src.scoring import candidates_from_hits
from ui.state import UiSession


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_skip_link() -> None:
    """A keyboard skip link to jump straight to the results region."""
    st.markdown(
        '<a class="cadence-skip" href="#cadence-results">Skip to results</a>',
        unsafe_allow_html=True,
    )


def render_results_anchor() -> None:
    st.markdown('<div id="cadence-results"></div>', unsafe_allow_html=True)


def announce(message: str) -> None:
    """Emit a visually-hidden, polite live-region update for screen readers."""
    st.markdown(
        f'<div class="cadence-sr-only" role="status" aria-live="polite">{_e(message)}</div>',
        unsafe_allow_html=True,
    )


def render_brand(*, compact: bool = False, paused: bool = False) -> None:
    class_name = "cadence-brand cadence-compact" if compact else "cadence-brand"
    if paused:
        class_name += " cadence-paused"
    st.markdown(
        f"""
        <div class="{class_name}">
          <div>
            <div class="cadence-kicker">A transparent listening room</div>
            <h1 class="cadence-wordmark">Cadence</h1>
          </div>
          <div class="cadence-wave" aria-hidden="true">
            <span></span><span></span><span></span><span></span><span></span><span></span>
          </div>
        </div>
        <div class="cadence-deck">Tell me the moment. I’ll show you the match—and let you shape it.</div>
        """,
        unsafe_allow_html=True,
    )


def render_disclosure(
    descriptor: CatalogDescriptor | None = None,
    *,
    artifact_source: str | None = None,
    warnings: tuple[str, ...] = (),
) -> None:
    """Describe the concrete artifact in use without overstating its evidence."""
    if descriptor is None or descriptor.edition is CatalogEdition.FICTIONAL:
        body = (
            "<strong>Regression catalog.</strong> Cadence is using the validated "
            "fictional 200-track catalog. It is intentionally preserved as the "
            "behavioral baseline. There is no playback."
        )
    else:
        edition = "Full" if descriptor.edition is CatalogEdition.FULL else "Lite"
        source = f" via {_e(artifact_source)}" if artifact_source else ""
        mood = (
            " Mood profiles are experimental audio-character estimates and may abstain."
            if descriptor.calibration_status == "experimental"
            else ""
        )
        fallback = (
            " The verified 300-track fallback is active; it is an edition of FMA, "
            "not a different catalog."
            if descriptor.edition is CatalogEdition.LITE
            else ""
        )
        body = (
            f"<strong>FMA {edition}.</strong> Cadence is discovering independent music "
            f"from a verified {_e(f'{descriptor.accepted_count:,}')}-track artifact{source}."
            f"{fallback}{mood} Metadata coverage varies, and unknown values stay unknown."
        )
    st.markdown(
        '<div class="cadence-disclosure">'
        + body
        + " Session refinements, ratings, and research disappear when this browser "
        "session ends and do not train a model.</div>",
        unsafe_allow_html=True,
    )
    if warnings:
        st.caption("Catalog notice: " + " · ".join(warnings))


def render_privacy_explainer() -> None:
    with st.popover("What does local-only mean?", icon=":material/shield:"):
        st.markdown(
            "**Local-only blocks AI-provider calls.** On a hosted deployment, your "
            "request still travels from your browser to the Cadence app server. It is "
            "held only in this session so refinements and undo work; it is not sent "
            "onward to Gemini, written to Cadence logs, or placed in a URL."
        )
        st.caption(
            "If Cadence detects personal details, provider-free routing becomes sticky "
            "for every later refinement in that mix. Session state disappears when "
            "you reset the mix or the Streamlit session ends."
        )


def render_mode_badges(turn: CompanionTurn) -> None:
    receipt = turn.receipt
    with st.container(
        horizontal=True,
        gap="small",
        key=f"mode_badges_{receipt.request_id}",
    ):
        if receipt.operating_mode is OperatingMode.DEGRADED:
            st.badge(
                "Local fallback",
                color="orange",
                icon=":material/offline_bolt:",
                help="Semantic retrieval was unavailable; TF-IDF and guides answered locally.",
            )
        elif receipt.embedding_source is EmbeddingSource.CACHE:
            st.badge(
                "Cached semantic + lexical",
                color="blue",
                icon=":material/database:",
                help="The query vector came from the committed offline cache.",
            )
        elif receipt.embedding_source is EmbeddingSource.LIVE:
            st.badge(
                "Live semantic + lexical",
                color="violet",
                icon=":material/cloud:",
                help="A live embedding call was used for this submitted turn.",
            )
        elif receipt.operating_mode is not None:
            st.badge(
                "Local lexical",
                color="green",
                icon=":material/computer:",
                help="TF-IDF and context guides ran without an AI-provider call.",
            )

        if receipt.voice_source is VoiceSource.GENERATED:
            st.badge("AI-selected intro", color="violet", icon=":material/auto_awesome:")
        elif receipt.operating_mode is not None:
            st.badge("Deterministic voice", color="gray", icon=":material/rule:")

        if receipt.force_local:
            st.badge("Provider-free", color="green", icon=":material/lock:")
        elif receipt.network_used:
            st.badge("Network used", color="orange", icon=":material/wifi:")
        else:
            st.badge("No provider call", color="green", icon=":material/wifi_off:")

        if receipt.guard_category is GuardCategory.SENSITIVE:
            st.badge("Personal detail removed", color="orange", icon=":material/privacy_tip:")
        elif receipt.guard_category is GuardCategory.INJECTION:
            st.badge("Embedded instruction ignored", color="orange", icon=":material/security:")


def _goal_label(cue_id: str) -> str:
    labels = {
        "energy_low_v1": "lower energy",
        "energy_high_v1": "higher energy",
        "acoustic_high_v1": "more acoustic",
        "acoustic_low_v1": "less acoustic",
        "valence_high_v1": "brighter tone",
        "valence_low_v1": "moodier tone",
        "dance_high_v1": "more movement",
        "dance_low_v1": "less movement",
        "tempo_near_v1": "tempo target",
        "tempo_range_v1": "tempo range",
        "tempo_atleast_v1": "minimum tempo",
        "tempo_atmost_v1": "maximum tempo",
        "tempo_near_ui_v1": "tempo target",
        "instrumentalness_prefer_high_v1": "more instrumental character",
        "instrumentalness_prefer_low_v1": "less instrumental character",
    }
    if cue_id in labels:
        return labels[cue_id]
    if cue_id.startswith("ui_danceability_low"):
        return "less movement"
    if cue_id.startswith("ui_acousticness_low"):
        return "less acoustic"
    if cue_id.startswith("mood_"):
        # Experimental quadrant goal (``mood_<quadrant>_<feature>_<relation>_v1``).
        # Present the listener-facing quadrant word, not its numeric axes.
        parts = cue_id.split("_")
        return f"{parts[1]} feel" if len(parts) > 1 else "mood"
    return cue_id.replace("_v1", "").replace("ui_", "").replace("_", " ")


def render_intent(
    intent: MusicIntent,
    *,
    on_remove: Callable[[str], None] | None = None,
    token: str = "intent",
) -> None:
    """Show what Cadence interpreted.

    When ``on_remove`` is supplied, each soft preference (genre, mood, numeric
    goal) becomes a one-tap chip that drops that facet and rebuilds the set
    through the same guarded pipeline. Hard filters stay as display badges; the
    Taste Console is the single place to toggle those.
    """
    st.markdown('<h2 class="cadence-eyebrow">Cadence heard</h2>', unsafe_allow_html=True)
    removable: list[tuple[str, str, str]] = []
    seen_labels: set[str] = set()
    if intent.genre:
        label = f"Genre · {intent.genre}"
        removable.append(("genre", label, "blue"))
        seen_labels.add(label)
    if intent.mood:
        label = f"Mood · {intent.mood}"
        removable.append(("mood", label, "violet"))
        seen_labels.add(label)
    for goal in intent.feature_goals:
        label = _goal_label(goal.cue_id)
        if label in seen_labels:
            # Collapse the two numeric axes of one mood quadrant into one chip.
            continue
        seen_labels.add(label)
        removable.append((goal.cue_id, label, "primary"))

    has_hard = intent.instrumental_only or intent.exclude_explicit
    if not removable and not has_hard:
        # Never leave "Cadence heard" visually empty: say plainly that nothing
        # structured was pinned down and the raw words still drove retrieval.
        st.caption(
            "No specific genre, mood, or preference to pin down — I searched on the "
            "words in your request. Steer it with the Taste Console or a quick follow-up."
        )
        return

    with st.container(horizontal=True, gap="small", key=f"intent_badges_{token}"):
        for facet, label, color in removable:
            if on_remove is not None:
                if st.button(
                    label,
                    key=f"remove_{token}_{facet}",
                    icon=":material/close:",
                    type="tertiary",
                    help="Remove this interpreted preference and rebuild the set.",
                ):
                    on_remove(facet)
            else:
                st.badge(label, color=color)
        if intent.instrumental_only:
            st.badge("Must be instrumental", color="green", icon=":material/check_circle:")
        if intent.exclude_explicit:
            st.badge("Must be clean", color="green", icon=":material/check_circle:")

    if on_remove is not None and removable:
        st.caption(
            "Tap a preference to drop it. Green rules are hard filters—toggle those "
            "in the Taste Console. The original text still informs retrieval."
        )
    else:
        st.caption(
            "Green badges are hard eligibility rules. Colored preference badges change "
            "ordering. The original text still informs retrieval."
        )


def render_context_evidence(turn: CompanionTurn) -> None:
    result = turn.response.retrieval
    if result is None or not result.guides_used:
        return
    with st.expander(
        f"Vocabulary sources used ({len(result.guides_used)})",
        icon=":material/library_books:",
    ):
        st.caption(
            "These guides expanded catalog vocabulary. They were evidence—not "
            "preferences added to your request."
        )
        for guide in result.guides_used:
            terms = ", ".join(guide.expansion_terms[:5]) or "no expansion terms"
            st.markdown(f"- **{guide.title}:** {terms}")


def render_framing(turn: CompanionTurn) -> None:
    response = turn.response
    framing = response.intro_message or response.message
    st.markdown(
        f'<div class="cadence-framing">{_e(framing)}</div>',
        unsafe_allow_html=True,
    )


def _plain_reason(candidate: RankedCandidate, *, sensitive: bool) -> str:
    structured = [
        reason.removeprefix("structured: ")
        for reason in candidate.components.reasons
        if reason.startswith("structured: ")
    ]
    readable: list[str] = []
    for reason in structured:
        bits = reason.split()
        if len(bits) >= 2 and bits[0] == "genre":
            readable.append(f"matches the {bits[1]} genre preference")
        elif len(bits) >= 2 and bits[0] == "mood":
            readable.append(f"matches the {bits[1]} mood preference")
        elif len(bits) >= 2:
            feature = {
                "energy": "energy",
                "valence": "mood tone",
                "danceability": "movement",
                "acousticness": "texture",
                "tempo_bpm": "tempo",
                "instrumentalness": "instrumental character",
            }.get(bits[0], bits[0])
            direction = {
                "prefer_high": "leans higher",
                "prefer_low": "leans lower",
                "near": "sits near the target",
                "at_least": "meets the minimum",
                "at_most": "stays under the maximum",
                "range": "sits in the requested range",
            }.get(bits[1], bits[1].replace("_", " "))
            readable.append(f"its {feature} {direction}")
    if readable:
        lead = readable[0].capitalize()
        if len(readable) > 1:
            return f"{lead}, and {readable[1]}."
        return lead + "."
    matched = [
        reason.removeprefix("matched: ")
        for reason in candidate.components.reasons
        if reason.startswith("matched: ")
    ]
    if matched and not sensitive:
        return "Catalog language overlaps on " + ", ".join(matched[:3]) + "."
    if candidate.components.semantic is not None and candidate.components.semantic > 0.0:
        return "Its catalog description is semantically close to this request."
    return "It remained inside the grounded, relevance-filtered result set."


def _cover(track_id: int, title: str) -> str:
    hue_a = (track_id * 37 + 18) % 360
    hue_b = (hue_a + 54) % 360
    initials = "".join(word[0] for word in title.split()[:2]).upper() or "♪"
    return (
        f'<div class="cadence-cover" aria-hidden="true" style="background:'
        f'linear-gradient(145deg,hsl({hue_a} 72% 63%),hsl({hue_b} 70% 48%));">'
        f"{_e(initials)}</div>"
    )


def _signal_bar(label: str, value: float | None, color: str) -> None:
    if value is None:
        st.markdown(
            f'<div class="cadence-signal"><div class="cadence-signal-row">'
            f'<span>{_e(label)}</span><span class="cadence-na">N/A · not evaluated</span>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        return
    pct = max(0.0, min(100.0, value * 100))
    meaning = "evaluated · no match" if value == 0.0 else "ranking signal"
    st.markdown(
        f'<div class="cadence-signal" role="meter" aria-label="{_e(label)}" '
        f'aria-valuemin="0" aria-valuemax="1" aria-valuenow="{value:.3f}">'
        f'<div class="cadence-signal-row">'
        f'<span>{_e(label)}</span><span>{value:.3f} · {_e(meaning)}</span></div>'
        f'<div class="cadence-signal-track"><span class="cadence-signal-fill" '
        f'style="width:{pct:.1f}%;background:{color};"></span></div></div>',
        unsafe_allow_html=True,
    )


def _edition_label(descriptor: CatalogDescriptor | None, track: CatalogTrack) -> str:
    if descriptor is None or track.catalog_id == "fictional":
        return "Fictional · regression"
    edition = "Full" if descriptor.edition is CatalogEdition.FULL else "Lite"
    return f"FMA {edition}"


def _lineage_for(track: CatalogTrack, field: str) -> FieldLineage | None:
    return next((item for item in track.lineage if item.field_name == field), None)


def _origin_label(lineage: FieldLineage | None) -> str:
    if lineage is None:
        return "catalog value"
    labels = {
        FieldOrigin.AUTHORED: "authored",
        FieldOrigin.ARTIST_SUPPLIED: "artist-supplied",
        FieldOrigin.FMA_METADATA: "FMA metadata",
        FieldOrigin.LIBROSA_COMPUTED: "Librosa-computed",
        FieldOrigin.ECHONEST_COMPUTED: "Echo Nest-computed",
        FieldOrigin.MODEL_ESTIMATED: "model-estimated",
        FieldOrigin.DETERMINISTIC_DERIVED: "deterministically derived",
        FieldOrigin.UNKNOWN: "unknown",
    }
    label = labels[lineage.origin]
    if lineage.origin is FieldOrigin.MODEL_ESTIMATED:
        details: list[str] = []
        if lineage.confidence is not None:
            details.append(f"{lineage.confidence:.0%} confidence")
        if lineage.interval_low is not None and lineage.interval_high is not None:
            details.append(f"interval {lineage.interval_low:.2f}–{lineage.interval_high:.2f}")
        if details:
            label += " · " + " · ".join(details)
    return label


def _render_audio_features(track: CatalogTrack) -> None:
    known = (
        ("Energy", "energy", track.energy, False),
        ("Valence", "valence", track.valence, False),
        ("Acousticness", "acousticness", track.acousticness, False),
        ("Danceability", "danceability", track.danceability, False),
        ("Instrumentalness", "instrumentalness", track.instrumentalness, False),
        ("Tempo", "tempo_bpm", track.tempo_bpm, True),
    )
    visible = [item for item in known if item[2] is not None]
    if not visible:
        st.caption("Audio character: no trustworthy numeric values available.")
        return
    st.markdown("**Known audio character**")
    for label, field, raw, bpm in visible:
        assert raw is not None
        value = f"{raw:.1f} BPM" if bpm else f"{raw:.2f}"
        st.caption(f"{label}: {value} · {_origin_label(_lineage_for(track, field))}")


def _render_scoped_text(track: CatalogTrack) -> None:
    scopes = (
        ("Catalog description", track.description),
        ("Track information", track.track_information),
        ("Album information", track.album_information),
        ("Artist biography", track.artist_biography),
    )
    visible = [(label, value) for label, value in scopes if value]
    if not visible:
        st.caption("No source-supplied descriptive text is available for this track.")
        return
    for label, value in visible:
        st.markdown(f"**{label}**")
        st.write(value)


def _local_track_summary(track: CatalogTrack) -> str:
    facts: list[str] = []
    genres = track.genres or ((track.genre,) if track.genre else ())
    if genres:
        facts.append("genres: " + ", ".join(genres[:4]))
    numeric = [
        label
        for label, value in (
            ("energy", track.energy),
            ("valence", track.valence),
            ("acousticness", track.acousticness),
            ("danceability", track.danceability),
            ("instrumentalness", track.instrumentalness),
            ("tempo", track.tempo_bpm),
        )
        if value is not None
    ]
    if numeric:
        facts.append("known audio fields: " + ", ".join(numeric))
    if track.license:
        facts.append(f"license: {track.license}")
    return "; ".join(facts) or "Only the track identity is available locally."


def _render_research_outcome(outcome: ResearchOutcome, track: CatalogTrack) -> None:
    brief = outcome.brief
    if brief.status is ResearchStatus.PUBLISHED:
        if brief.identity_confidence is None:
            st.warning(
                "Identity not verified against MusicBrainz — this note comes from a web "
                "search for the title and artist and may describe a different recording. "
                "Sources are cited below.",
                icon=":material/help:",
            )
        else:
            st.success(
                "Identity resolved and every published claim has a validated citation.",
                icon=":material/fact_check:",
            )
        citations = {item.citation_id: item for item in brief.citations}
        if brief.narrative:
            # The model's grounded creative presentation, drawn from the cited
            # sources below. Escaped and shown as Cadence's note, not catalog truth.
            st.markdown(
                f'<div class="cadence-framing">{_e(brief.narrative)}</div>',
                unsafe_allow_html=True,
            )
            st.caption("Cadence's note, written from the cited sources below.")
        with st.expander("Cited points", icon=":material/format_list_bulleted:"):
            for claim in brief.claims:
                st.write(claim.text)
                st.caption(
                    "Cited by: "
                    + ", ".join(citations[item].source_domain for item in claim.citation_ids)
                )
        st.markdown("**Validated sources**")
        for index, citation in enumerate(brief.citations, start=1):
            st.link_button(
                f"{index}. {citation.title}",
                citation.url,
                key=(
                    f"research_source_{brief.track_ref.catalog_id}_"
                    f"{brief.track_ref.track_id}_{citation.citation_id}"
                ),
                icon=":material/open_in_new:",
            )
        if brief.timestamp:
            st.caption(f"Session-only research · {brief.timestamp}")
    elif brief.status is ResearchStatus.CATALOG_NOTE and brief.narrative:
        st.info(
            "Live web sources weren't available, so this is Cadence's note written "
            "from the catalog — flavor, not verified facts about the artist.",
            icon=":material/auto_awesome:",
        )
        st.markdown(
            f'<div class="cadence-framing">{_e(brief.narrative)}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Deterministic local summary: " + _local_track_summary(track))
        if brief.timestamp:
            st.caption(f"Session-only · {brief.timestamp}")
    else:
        warning = brief.warnings[0] if brief.warnings else "Research could not be verified."
        st.warning(warning, icon=":material/search_off:")
        st.caption("Deterministic local summary: " + _local_track_summary(track))

    with st.expander("Research action trace", icon=":material/account_tree:"):
        st.code("\n→ ".join(outcome.trace), language="text")
        st.caption(
            "This trace records sanitized actions and outcomes, never private reasoning. "
            "Research did not alter eligibility, ranking, or catalog fields."
        )


def render_track_cards(
    turn: CompanionTurn,
    *,
    catalog_descriptor: CatalogDescriptor | None = None,
    developer: bool = False,
    research_agent: TrackResearchAgent | None = None,
    research_cache: MutableMapping[str, ResearchOutcome] | None = None,
    on_feedback: Callable[[object, str], None] | None = None,
) -> None:
    response = turn.response
    if response.retrieval is None:
        return
    candidates = candidates_from_hits(response.retrieval.hits)
    sensitive = turn.receipt.guard_category is GuardCategory.SENSITIVE
    token = turn.receipt.request_id[:10]
    for rank, candidate in enumerate(candidates, start=1):
        track = candidate.track
        source_key = track.ref.source_id.replace(":", "_")
        with st.container(border=True, key=f"track_card_{source_key}"):
            st.markdown(
                '<div class="cadence-track-head">'
                + _cover(track.id, track.title)
                + '<div><div class="cadence-rank">'
                + f"Selection {rank:02d}</div>"
                + f'<h3 class="cadence-title">{_e(track.title)}</h3>'
                + f'<div class="cadence-artist">{_e(track.artist)}</div></div></div>',
                unsafe_allow_html=True,
            )
            with st.container(horizontal=True, gap="small"):
                st.badge(
                    _edition_label(catalog_descriptor, track),
                    color="orange" if track.catalog_id == "fma" else "gray",
                )
                if track.genre:
                    st.badge(track.genre, color="blue")
                elif track.genres:
                    st.badge(track.genres[0], color="blue")
                if track.mood:
                    st.badge(track.mood, color="violet")
                elif track.mood_profile is not None:
                    if track.mood_profile.label is not None:
                        st.badge(
                            f"{track.mood_profile.label.value} · experimental",
                            color="violet",
                        )
                    else:
                        st.badge("balanced / uncertain · experimental", color="gray")
                if track.era:
                    st.badge(track.era, color="gray")
                if (
                    response.intent
                    and response.intent.instrumental_only
                    and track.instrumental is True
                ):
                    st.badge("Instrumental confirmed", color="green")
                if (
                    response.intent
                    and response.intent.exclude_explicit
                    and track.explicit is False
                ):
                    st.badge("Clean confirmed", color="green")
            st.markdown(
                f'<div class="cadence-why"><strong>Why it fits:</strong> '
                f'{_e(_plain_reason(candidate, sensitive=sensitive))}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Why this track?", icon=":material/tune:"):
                if track.mood_profile is not None:
                    profile = track.mood_profile
                    leader = (
                        profile.label.value if profile.label is not None else "balanced/uncertain"
                    )
                    confidence = (
                        f" · input-evidence confidence {profile.confidence:.0%}"
                        if profile.confidence is not None
                        else ""
                    )
                    st.caption(
                        f"Experimental mood profile: {leader}{confidence}. This is derived "
                        "from trustworthy valence/energy axes, not an authored FMA mood."
                    )
                _render_audio_features(track)
                st.caption(
                    "Signals are ranking inputs—not confidence, probability, or a "
                    "prediction that you will like the track."
                )
                _signal_bar("Semantic similarity", candidate.components.semantic, "#82AFFF")
                _signal_bar("Keyword similarity", candidate.components.lexical, "#E4A24B")
                _signal_bar("Structured relevance", candidate.components.structured, "#BEA7E5")
                _signal_bar(
                    "Fused relevance (before diversity)",
                    candidate.components.fused,
                    "#74C69D",
                )

                structured_reasons = [
                    reason.removeprefix("structured: ")
                    for reason in candidate.components.reasons
                    if reason.startswith("structured: ")
                ]
                matched = [
                    reason.removeprefix("matched: ")
                    for reason in candidate.components.reasons
                    if reason.startswith("matched: ")
                ]
                if structured_reasons:
                    st.write("Structured evidence: " + ", ".join(structured_reasons))
                if matched and not sensitive:
                    st.write("Matched catalog terms: " + ", ".join(matched))
                if developer:
                    st.code(
                        f"track_id={track.id}\n"
                        f"source={candidate.source_type.value}\n"
                        f"fusion={candidate.components.fusion_version}\n"
                        f"content_hash={candidate.content_hash[:16]}…",
                        language="text",
                    )
            with st.expander("Source, rights & context", icon=":material/source:"):
                _render_scoped_text(track)
                st.markdown("**License**")
                st.write(track.license or "Unknown — Cadence will not invent one.")
                if catalog_descriptor and catalog_descriptor.attribution:
                    st.caption("Attribution: " + " · ".join(catalog_descriptor.attribution))
                links = tuple(
                    dict.fromkeys(
                        value
                        for value in (
                            track.track_url,
                            track.artist_url,
                            track.album_url,
                            track.source_url,
                        )
                        if value
                    )
                )
                for index, url in enumerate(links, start=1):
                    st.link_button(
                        f"Source link {index}",
                        url,
                        key=f"catalog_link_{source_key}_{index}",
                        icon=":material/open_in_new:",
                    )

            if (
                catalog_descriptor is not None
                and catalog_descriptor.capabilities.research
                and research_agent is not None
                and research_cache is not None
            ):
                st.markdown("**Optional track research**")
                st.caption(
                    "Clicking sends only this track's title and artist to the web "
                    "(MusicBrainz, then grounded search) — never your request, history, "
                    "or preferences — so it works even in local-only mode. It runs after "
                    "ranking and can't change this recommendation. A privacy lock keeps "
                    "it fully local."
                )
                cached = research_cache.get(track.ref.source_id)
                if cached is None:
                    if st.button(
                        "Research this track",
                        key=f"research_{source_key}",
                        icon=":material/travel_explore:",
                        help="Runs after ranking and cannot change this recommendation.",
                    ):
                        with st.spinner("Resolving identity and validating citations…"):
                            research_cache[track.ref.source_id] = research_agent.research(track)
                        st.rerun()
                else:
                    _render_research_outcome(cached, track)

            if on_feedback is not None:
                with st.container(horizontal=True, gap="small", key=f"fb_{token}_{track.id}"):
                    if st.button(
                        "👍 More like this", key=f"fb_like_{token}_{track.id}",
                        type="tertiary", help="Learn toward this track (this session only)",
                    ):
                        on_feedback(track, "like")
                    if st.button(
                        "👎 Fewer like this", key=f"fb_less_{token}_{track.id}",
                        type="tertiary", help="Learn away from this track's character",
                    ):
                        on_feedback(track, "less")
                    if st.button(
                        "⚑ Didn't fit", key=f"fb_missed_{token}_{track.id}",
                        type="tertiary", help="Set this one aside for the session",
                    ):
                        on_feedback(track, "missed")

            source_note = (
                "FMA catalog record · no playback"
                if track.catalog_id == "fma"
                else "Fictional regression track · no playback"
            )
            st.caption(source_note)


def render_action_state(turn: CompanionTurn) -> None:
    response = turn.response
    if response.action is CompanionAction.SAFE_RESPONSE:
        st.warning(response.message, icon=":material/health_and_safety:")
    elif response.action is CompanionAction.CLARIFY:
        st.info(response.message, icon=":material/chat_bubble:")
    elif response.action is CompanionAction.NO_MATCH:
        st.warning(response.message, icon=":material/search_off:")
    elif response.action is CompanionAction.DEGRADED:
        st.warning(
            "Semantic search was unavailable, so Cadence built this set locally "
            "from catalog language and context guides.",
            icon=":material/offline_bolt:",
        )


def render_pipeline(turn: CompanionTurn) -> None:
    receipt = turn.receipt
    response = turn.response
    action = response.action
    retrieval = turn.response.retrieval
    trace = response.trace
    evaluation = trace.evaluation if trace else None
    stages: list[tuple[str, str]] = [("Guard", receipt.guard_category.value)]

    if action is CompanionAction.SAFE_RESPONSE:
        stages.extend(
            (("Safe branch", "retrieval skipped"), ("Output", action.value))
        )
    elif action is CompanionAction.CLARIFY:
        if response.intent is not None:
            stages.append(("Intent", "needs clarification"))
        stages.append(("Output", action.value))
    else:
        source = receipt.embedding_source.value if receipt.embedding_source else "lexical"
        structured = bool(
            retrieval and any(hit.structured_score is not None for hit in retrieval.hits)
        )
        stages.extend(
            (
                ("Intent", "typed"),
                ("Retrieve", f"{len(receipt.candidate_ids)} · {source}"),
                ("Fusion", "text + structured" if structured else "text only"),
                (
                    "Diversify",
                    (
                        f"{receipt.diversity.value} · changed"
                        if trace and trace.diversity_applied
                        else (
                            f"{receipt.diversity.value} · unchanged"
                            if receipt.candidate_ids
                            else "skipped"
                        )
                    ),
                ),
            )
        )
        if action is CompanionAction.NO_MATCH:
            verdict = "rejected" if evaluation and evaluation.failures else "no candidates"
            stages.extend((("Evaluate", verdict), ("Output", action.value)))
        else:
            stages.extend(
                (
                    ("Evaluate", "passed" if evaluation and evaluation.ok else "failed"),
                    (
                        "Voice / guard",
                        receipt.voice_source.value if receipt.voice_source else "none",
                    ),
                    ("Output", f"{action.value} · {len(receipt.final_ids)}"),
                )
            )
    blocks = "".join(
        f'<div class="cadence-stage">{_e(label)}<strong>{_e(value)}</strong></div>'
        for label, value in stages
    )
    st.markdown(f'<div class="cadence-pipeline">{blocks}</div>', unsafe_allow_html=True)


def render_developer_view(
    turn: CompanionTurn, *, title: str = "Under the hood"
) -> None:
    with st.expander(title, icon=":material/schema:"):
        st.caption(
            "A request-local receipt—not a shared log. It contains controlled "
            "facets, IDs, timings, and fingerprints, never the prompt."
        )
        render_pipeline(turn)
        left, right = st.columns(2)
        left.metric("Latency", f"{turn.receipt.latency_ms:.2f} ms")
        right.metric(
            "Candidate → final",
            f"{len(turn.receipt.candidate_ids)} → {len(turn.receipt.final_ids)}",
        )
        st.json(turn.receipt.model_dump(mode="json"), expanded=False)
        if turn.response.trace is not None:
            st.markdown("**Bounded-agent trace**")
            st.json(turn.response.trace.model_dump(mode="json"), expanded=False)
        if turn.comparison is not None and turn.comparison.rows:
            render_signal_comparison(turn.comparison)


def _ranked_list(rows, *, highlight: set[int] | None = None) -> str:
    highlight = highlight or set()
    items = "".join(
        f'<li>{_e(row.title)}'
        + (' <span class="cadence-lift">lifted</span>' if row.track_id in highlight else "")
        + "</li>"
        for row in rows
    )
    return f'<ol class="cadence-legrank">{items}</ol>'


def render_signal_comparison(comparison: SignalComparison, *, top_n: int = 5) -> None:
    """Show how the candidate pool would rank under each retrieval leg.

    This is the structured-preference story made visible: the same pool, ordered
    by text alone vs by the structured preferences vs by the fused score Cadence
    actually used. Diversity (MMR) then runs on the fused order.
    """
    st.markdown("**How the pool ranked under each leg** · before diversity")
    rows = comparison.rows
    text_top = sorted(rows, key=lambda row: (-row.text, row.track_id))[:top_n]
    fused_top = sorted(rows, key=lambda row: (-row.fused, row.track_id))[:top_n]
    text_ids = {row.track_id for row in text_top}
    lifted = {row.track_id for row in fused_top if row.track_id not in text_ids}

    if not comparison.structured_active:
        st.caption(
            "No structured preference this turn, so the structured leg did not run — "
            "the fused order equals the text-only order."
        )
        columns = st.columns(2)
        legs = (("Text only · keywords + semantics", text_top, set()),
                ("Fused · what Cadence used", fused_top, set()))
    else:
        structured_top = sorted(
            (row for row in rows if row.structured is not None),
            key=lambda row: (-(row.structured or 0.0), row.track_id),
        )[:top_n]
        columns = st.columns(3)
        legs = (
            ("Text only · keywords + semantics", text_top, set()),
            ("Structured only · your preferences", structured_top, set()),
            ("Fused · what Cadence used", fused_top, lifted),
        )
    for column, (label, leg_rows, highlight) in zip(columns, legs):
        with column:
            st.caption(label)
            st.markdown(_ranked_list(leg_rows, highlight=highlight), unsafe_allow_html=True)
    if comparison.structured_active and lifted:
        st.caption(
            "“Lifted” marks tracks the structured leg pulled into the top set that "
            "text ranking alone would have missed."
        )


def _track_name_lookup(state: UiSession) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for snapshot in state.snapshots:
        result = snapshot.turn.response.retrieval
        if result:
            lookup.update({hit.track.id: hit.track.title for hit in result.hits})
    return lookup


def render_evolution(state: UiSession) -> None:
    entries = [snapshot.evolution for snapshot in state.snapshots if snapshot.evolution]
    if not entries:
        return
    lookup = _track_name_lookup(state)
    st.markdown("## How the set evolved")
    visible = entries[-5:]
    if len(entries) > len(visible):
        st.caption(f"Showing the latest {len(visible)} of {len(entries)} changes.")
    for index, entry in enumerate(reversed(visible), start=1):
        with st.container(border=True, key=f"evolution_{index}"):
            st.markdown("\n".join(f"- {change}" for change in entry.changes))
            if entry.entered_ids:
                st.caption(
                    "Entered: "
                    + ", ".join(lookup.get(track_id, f"Track {track_id}") for track_id in entry.entered_ids)
                )
            if entry.dropped_ids:
                st.caption(
                    "Left: "
                    + ", ".join(lookup.get(track_id, f"Track {track_id}") for track_id in entry.dropped_ids)
                )


def render_first_run_story() -> None:
    st.markdown("## A recommendation you can inspect")
    columns = st.columns(3)
    steps: Iterable[tuple[str, str, str]] = (
        ("01", "Describe", "Use a mood, moment, activity, genre, or musical constraint."),
        ("02", "Inspect", "See interpreted intent, evidence, operating mode, and guardrails."),
        ("03", "Shape", "Remix with typed controls, then undo without losing the original set."),
    )
    for column, (number, title, body) in zip(columns, steps):
        with column.container(border=True):
            st.caption(number)
            st.markdown(f"**{title}**")
            st.write(body)
