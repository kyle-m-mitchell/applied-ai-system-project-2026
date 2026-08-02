"""Curated, deterministic starting points shown in the flagship UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExamplePrompt:
    key: str
    label: str
    prompt: str
    description: str


EXAMPLES: tuple[ExamplePrompt, ...] = (
    ExamplePrompt(
        "focus",
        "Deep focus",
        "clean instrumental lofi for studying, calm and acoustic",
        "Hard filters plus lower energy and acoustic texture.",
    ),
    ExamplePrompt(
        "jazz",
        "Rainy jazz",
        "wistful rainy-day jazz, mellow and acoustic",
        "A named genre with a reflective context.",
    ),
    ExamplePrompt(
        "workout",
        "Run harder",
        "high-energy songs for a hard run",
        "A directional energy preference.",
    ),
    ExamplePrompt(
        "tempo",
        "Around 118 BPM",
        "clean dance music around 118 bpm",
        "An explicit tempo goal and clean constraint.",
    ),
    ExamplePrompt(
        "privacy",
        "Privacy demo",
        "my email is listener@example.com, find me melancholy piano",
        "Shows redaction and sticky provider-free routing.",
    ),
)
