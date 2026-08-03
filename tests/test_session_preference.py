"""Tests for session-only taste learning (pure, deterministic, offline)."""

from __future__ import annotations

from src.contracts import CatalogTrack, RetrievalHit, SessionPreference, SourceType
from src.session_preference import (
    SESSION_WEIGHT,
    apply_fit_missed,
    apply_like,
    apply_suggest_less,
    clear_learning,
    session_ranked_hits,
    session_signal,
)


def _track(track_id: int, *, genre="folk", energy=0.5, valence=0.5,
           danceability=0.5, acousticness=0.5, instrumentalness=None) -> CatalogTrack:
    return CatalogTrack.model_validate(
        {
            "id": track_id, "catalog_id": "fictional", "title": f"T{track_id}",
            "artist": "A", "genre": genre, "energy": energy, "valence": valence,
            "danceability": danceability, "acousticness": acousticness,
            "instrumentalness": instrumentalness,
        }
    )


def _hit(track: CatalogTrack, score: float) -> RetrievalHit:
    return RetrievalHit(
        source_type=SourceType.CATALOG, source_id=f"c:{track.id}", content_hash="h",
        fields_used=("genre",), score=score, track=track,
    )


def test_like_nudges_affinity_toward_the_tracks_character_and_adds_an_exemplar():
    pref = apply_like(SessionPreference(), _track(1, genre="folk", energy=0.9))
    assert pref.feature_bias["energy"] > 0.0     # liked a high-energy track
    assert pref.genre_bias["folk"] > 0.0
    assert len(pref.exemplars) == 1 and pref.exemplars[0].energy == 0.9
    assert pref.is_active


def test_suggest_less_is_the_reverse_nudge():
    pref = apply_suggest_less(SessionPreference(), _track(1, energy=0.9))
    assert pref.feature_bias["energy"] < 0.0     # steer away from high energy
    assert pref.exemplars == ()                  # a negative tap adds no exemplar


def test_unknown_feature_contributes_nothing_and_never_crashes():
    pref = apply_like(SessionPreference(), _track(1, energy=None))
    assert "energy" not in pref.feature_bias     # Unknown is not treated as 0
    assert session_signal(_track(2, energy=None), pref) >= 0.0


def test_fit_missed_soft_suppresses_and_a_later_like_reverses_it():
    pref = apply_fit_missed(SessionPreference(), _track(7))
    assert pref.suppressed_ids == (7,)
    assert apply_like(pref, _track(7)).suppressed_ids == ()  # like clears suppression


def test_inactive_preference_leaves_ordering_and_signal_neutral():
    assert session_signal(_track(1, energy=0.9), SessionPreference()) == 0.5
    hits = [_hit(_track(1), 0.9), _hit(_track(2), 0.5)]
    assert session_ranked_hits(hits, None) == tuple(hits)
    assert session_ranked_hits(hits, SessionPreference()) == tuple(hits)  # enabled but empty


def test_learning_reorders_the_pool_toward_liked_energy_but_stays_bounded():
    pref = SessionPreference()
    for _ in range(2):  # a couple of likes on high-energy tracks
        pref = apply_like(pref, _track(99, energy=0.95))
    low = _track(1, energy=0.1)
    high = _track(2, energy=0.95)
    # Text relevance slightly favors the low-energy track; session taste flips it.
    ranked = session_ranked_hits([_hit(low, 0.62), _hit(high, 0.58)], pref)
    assert ranked[0].track.id == 2
    # Bounded: the blended score moved by no more than the session weight.
    assert abs(ranked[0].score - 0.58) <= SESSION_WEIGHT + 1e-9


def test_suppressed_track_is_demoted_but_not_removed():
    pref = apply_fit_missed(SessionPreference(), _track(1))
    ranked = session_ranked_hits([_hit(_track(1), 0.9), _hit(_track(2), 0.4)], pref)
    ids = [hit.track.id for hit in ranked]
    assert ids == [2, 1] and 1 in ids       # demoted, still present (reversible)


def test_clear_learning_keeps_the_dont_learn_choice():
    pref = apply_like(SessionPreference(enabled=False), _track(1, energy=0.9))
    assert pref.enabled is False and pref.is_active is False  # disabled = no effect
    cleared = clear_learning(pref)
    assert cleared.enabled is False and cleared.feature_bias == {}
