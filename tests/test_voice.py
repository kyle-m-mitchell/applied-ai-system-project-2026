"""Tests for Cadence's voice (deterministic + guarded optional generation)."""

from __future__ import annotations

from src.contracts import CatalogTrack, MusicIntent, RetrievalHit, SourceType, VoiceSource
from src.generation import FakeTextGenerator, FewShot, TextGenerator
from src.voice import CadenceVoice


def _hit(track_id: int, title: str) -> RetrievalHit:
    track = CatalogTrack.model_validate(
        {
            "id": track_id,
            "title": title,
            "artist": "Someone",
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.4,
            "tempo_bpm": 80,
            "valence": 0.5,
            "danceability": 0.5,
            "acousticness": 0.9,
            "description": "A placeholder track used only for voice-logic tests.",
            "tags": ("one", "two"),
            "contexts": ("alpha", "beta"),
            "instruments": ("piano",),
            "instrumental": True,
            "explicit": False,
            "era": "2020s",
        }
    )
    return RetrievalHit(
        source_type=SourceType.CATALOG,
        source_id=f"catalog:{track_id}",
        content_hash="h",
        fields_used=("genre",),
        score=0.5,
        matched_terms=("calm",),
        track=track,
    )


HITS = [_hit(1, "Focus Flow"), _hit(2, "Blue Desk Lamp")]
INTENT = MusicIntent(query="calm study beats", limit=5)


class _Inventing(TextGenerator):
    model_id = "stub"

    def generate(self, system: str, few_shot: FewShot, user: str) -> str:
        return 'You absolutely must hear "Ghost Town Radio" tonight.'


class _Boom(TextGenerator):
    model_id = "stub"

    def generate(self, system: str, few_shot: FewShot, user: str) -> str:
        raise RuntimeError("provider down")


class _PersonaViolation(TextGenerator):
    model_id = "remote-stub"
    is_remote = True

    def generate(self, system: str, few_shot: FewShot, user: str) -> str:
        return (
            "I am human, I listened to these tracks, and you should visit "
            "https://evil.example immediately."
        )


class _ContradictoryMusicClaims(TextGenerator):
    model_id = "remote-stub"

    def generate(self, system: str, few_shot: FewShot, user: str) -> str:
        return "These are all fast electronic vocals for an intense evening."


def test_deterministic_voice_is_grounded():
    result = CadenceVoice().render(HITS, INTENT, generator=None)
    assert result.source is VoiceSource.TEMPLATE
    assert result.fallback_reason is None and result.model is None
    assert "Focus Flow" in result.message and "Blue Desk Lamp" in result.message


def test_fake_generator_produces_grounded_generated_voice():
    result = CadenceVoice().render(HITS, INTENT, generator=FakeTextGenerator())
    assert result.source is VoiceSource.GENERATED
    assert result.model == "fake-generator-v1"  # records which generator ran
    assert result.text_evaluation is not None and result.text_evaluation.ok
    assert "Focus Flow" in result.message  # authoritative track list still supplied by us


def test_invented_song_is_caught_and_falls_back():
    result = CadenceVoice().render(HITS, INTENT, generator=_Inventing())
    assert result.source is VoiceSource.TEMPLATE
    assert result.fallback_reason and "grounding" in result.fallback_reason
    assert "Ghost Town Radio" not in result.message  # ungrounded framing discarded
    # the failing grounding report is preserved, not silently dropped
    assert result.text_evaluation is not None and not result.text_evaluation.ok


def test_generation_error_falls_back():
    result = CadenceVoice().render(HITS, INTENT, generator=_Boom())
    assert result.source is VoiceSource.TEMPLATE
    assert result.fallback_reason == "generation failed"
    assert "Focus Flow" in result.message


def test_persona_or_link_violation_falls_back_and_records_network_use():
    result = CadenceVoice().render(HITS, INTENT, generator=_PersonaViolation())
    assert result.source is VoiceSource.TEMPLATE
    assert result.network_used
    assert result.text_evaluation is not None and not result.text_evaluation.ok
    assert "evil.example" not in result.message


def test_generated_music_claims_are_discarded_instead_of_fact_checked_by_guessing():
    result = CadenceVoice().render(HITS, INTENT, generator=_ContradictoryMusicClaims())
    assert result.source is VoiceSource.TEMPLATE
    assert result.text_evaluation is not None and not result.text_evaluation.ok
    assert "electronic vocals" not in result.message


def test_zero_semantic_score_is_not_narrated_as_a_close_match():
    zero = HITS[0].model_copy(
        update={"semantic_score": 0.0, "matched_terms": ("study",)}
    )
    rendered = CadenceVoice().render([zero], INTENT)
    assert "close match in feel" not in rendered.message
    assert "study" in rendered.message
