"""Experimental, evidence-preserving valence/arousal mood mapping.

FMA does not provide an authored track-level mood field.  This module therefore
keeps its result separate from ``CatalogTrack.mood``: it derives four soft
quadrant scores from energy and valence and labels the result only when the
leading quadrant is meaningfully ahead of the runner-up.

The math is intentionally tiny, deterministic, and dependency-free so the ETL,
SQLite builder, evaluator, and UI can all use exactly the same implementation.
It is an interpretation of numeric evidence, not a claim about what the artist
intended a listener to feel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Final, Mapping


MOOD_METHOD_VERSION: Final = "cadence-va-quadrant-v1"
MOOD_TEMPERATURE: Final = 0.15
MOOD_AMBIGUITY_MARGIN: Final = 0.10

MOOD_AXIS_DIRECTIONS: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        "upbeat": MappingProxyType({"energy": "prefer_high", "valence": "prefer_high"}),
        "calm": MappingProxyType({"energy": "prefer_low", "valence": "prefer_high"}),
        "intense": MappingProxyType({"energy": "prefer_high", "valence": "prefer_low"}),
        "somber": MappingProxyType({"energy": "prefer_low", "valence": "prefer_low"}),
    }
)


def _unit_number(name: str, value: float) -> float:
    """Validate one real, finite number on the closed unit interval."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return result


def _sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    forward = math.exp(value)
    return forward / (1.0 + forward)


@dataclass(frozen=True, slots=True)
class MoodComputation:
    """One experimental four-quadrant calculation.

    ``confidence`` describes the confidence of the *input evidence* when the
    caller has it.  It is deliberately not inferred from the winning quadrant's
    score: a decisive transformation of uncertain model estimates is still
    uncertain evidence.
    """

    upbeat: float
    calm: float
    intense: float
    somber: float
    label: str | None
    confidence: float | None
    margin: float
    method_version: str = MOOD_METHOD_VERSION
    experimental: bool = True

    @property
    def scores(self) -> Mapping[str, float]:
        """Return an immutable label-to-score view in canonical order."""
        return MappingProxyType(
            {
                "upbeat": self.upbeat,
                "calm": self.calm,
                "intense": self.intense,
                "somber": self.somber,
            }
        )

    def as_profile_kwargs(self) -> dict[str, object]:
        """Return the exact payload accepted by the shared ``MoodProfile`` contract."""
        return {
            "upbeat": self.upbeat,
            "calm": self.calm,
            "intense": self.intense,
            "somber": self.somber,
            "label": self.label,
            "confidence": self.confidence,
            "method_version": self.method_version,
            "experimental": self.experimental,
        }


def compute_mood_profile(
    energy: float | None,
    valence: float | None,
    *,
    energy_confidence: float | None = None,
    valence_confidence: float | None = None,
    temperature: float = MOOD_TEMPERATURE,
    ambiguity_margin: float = MOOD_AMBIGUITY_MARGIN,
) -> MoodComputation | None:
    """Derive an experimental mood profile, or abstain when an axis is unknown.

    The four scores are the product of two soft binary axes and therefore sum to
    one (within floating-point precision).  A label is returned only when the
    top score leads the second score by at least ``ambiguity_margin``.

    Missing evidence returns ``None``. Invalid evidence raises ``ValueError`` so
    a corrupt upstream row cannot quietly become a plausible-looking mood.
    """
    if energy is None or valence is None:
        return None
    energy_value = _unit_number("energy", energy)
    valence_value = _unit_number("valence", valence)

    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a positive real number")
    temperature_value = float(temperature)
    if not math.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be a positive finite number")
    margin_required = _unit_number("ambiguity_margin", ambiguity_margin)

    evidence_confidences: list[float] = []
    if energy_confidence is not None:
        evidence_confidences.append(_unit_number("energy_confidence", energy_confidence))
    if valence_confidence is not None:
        evidence_confidences.append(_unit_number("valence_confidence", valence_confidence))
    confidence = min(evidence_confidences) if evidence_confidences else None

    high_arousal = _sigmoid((energy_value - 0.5) / temperature_value)
    positive = _sigmoid((valence_value - 0.5) / temperature_value)
    scores = {
        "upbeat": high_arousal * positive,
        "calm": (1.0 - high_arousal) * positive,
        "intense": high_arousal * (1.0 - positive),
        "somber": (1.0 - high_arousal) * (1.0 - positive),
    }
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    margin = ranked[0][1] - ranked[1][1]
    label = ranked[0][0] if margin >= margin_required else None

    return MoodComputation(
        upbeat=scores["upbeat"],
        calm=scores["calm"],
        intense=scores["intense"],
        somber=scores["somber"],
        label=label,
        confidence=confidence,
        margin=margin,
    )


def mood_axis_directions(label: str) -> Mapping[str, str]:
    """Map a quadrant request to transparent energy/valence directions.

    This lets the intent layer rank on the underlying evidence rather than
    treating an experimental label as an observed categorical fact.
    """
    if not isinstance(label, str):
        raise ValueError("experimental mood label must be text")
    normalized = label.strip().lower()
    try:
        return MOOD_AXIS_DIRECTIONS[normalized]
    except KeyError as exc:
        choices = ", ".join(MOOD_AXIS_DIRECTIONS)
        raise ValueError(f"unknown experimental mood {label!r}; choose {choices}") from exc
