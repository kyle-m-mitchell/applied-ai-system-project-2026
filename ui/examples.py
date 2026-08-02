"""Curated, deterministic starting points shown in the flagship UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExamplePrompt:
    key: str
    label: str
    prompt: str
    description: str


FICTIONAL_EXAMPLES: tuple[ExamplePrompt, ...] = (
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


FMA_EXAMPLES: tuple[ExamplePrompt, ...] = (
    ExamplePrompt(
        "fma_focus",
        "Independent focus",
        "calm electronic music for focused writing",
        "A calm, lower-energy search over independently released music.",
    ),
    ExamplePrompt(
        "fma_folk",
        "Acoustic morning",
        "bright acoustic folk for a slow morning",
        "Genre and trustworthy audio-character preferences without guessed filters.",
    ),
    ExamplePrompt(
        "fma_run",
        "Run harder",
        "high-energy dance music for a hard run",
        "A directional energy and movement preference.",
    ),
    ExamplePrompt(
        "fma_instrumental",
        "More instrumental",
        "electronic music with more instrumental character",
        "A soft instrumentalness preference where trustworthy values exist.",
    ),
    ExamplePrompt(
        "fma_privacy",
        "Privacy demo",
        "my email is listener@example.com, find me somber piano music",
        "Shows redaction and sticky provider-free routing.",
    ),
)


def examples_for_catalog(catalog_id: str) -> tuple[ExamplePrompt, ...]:
    """Return examples that stay inside the selected catalog's capabilities."""
    return FMA_EXAMPLES if catalog_id == "fma" else FICTIONAL_EXAMPLES


# Backward-compatible public name for tests and downstream imports that use the
# original, fully evidenced regression catalog.
EXAMPLES = FICTIONAL_EXAMPLES
