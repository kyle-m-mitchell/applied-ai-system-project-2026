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


def test_deterministic_voice_is_grounded():
    message, source, fallback = CadenceVoice().render(HITS, INTENT, generator=None)
    assert source is VoiceSource.TEMPLATE
    assert fallback is None
    assert "Focus Flow" in message and "Blue Desk Lamp" in message


def test_fake_generator_produces_grounded_gemini_voice():
    message, source, fallback = CadenceVoice().render(HITS, INTENT, generator=FakeTextGenerator())
    assert source is VoiceSource.GEMINI
    assert fallback is None
    assert "Focus Flow" in message  # authoritative track list still supplied by us


def test_invented_song_is_caught_and_falls_back():
    message, source, fallback = CadenceVoice().render(HITS, INTENT, generator=_Inventing())
    assert source is VoiceSource.TEMPLATE
    assert fallback and "grounding" in fallback
    assert "Ghost Town Radio" not in message  # the ungrounded framing was discarded


def test_generation_error_falls_back():
    message, source, fallback = CadenceVoice().render(HITS, INTENT, generator=_Boom())
    assert source is VoiceSource.TEMPLATE
    assert fallback == "generation failed"
    assert "Focus Flow" in message
