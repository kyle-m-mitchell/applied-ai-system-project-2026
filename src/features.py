"""Public, reusable feature-comparison utilities.

Extracted from the legacy scorer so there is exactly **one** correct
implementation of "how alike are these two categorical/numeric features?" —
shared by the structured scorer, MMR diversity, and (Feature 8) the coming
structured-preference leg.

The categorical helpers all embody the same guard. A family lookup that misses
returns ``None``, and in Python ``None == None`` is ``True``. Comparing two
*missed* lookups directly would declare two unrelated unknowns "the same family"
(the bug that used to live in MMR). Every helper here refuses to treat an absent
family as a match.

Numeric ``None`` is load-bearing too: it means **not evaluated**, which is
distinct from a real ``0.0`` (**evaluated, no overlap**). Callers decide how to
treat "not evaluated"; the utilities never invent a value for a missing one.
"""

from __future__ import annotations

from collections.abc import Mapping


def normalize_category(value: str | None) -> str | None:
    """Canonicalize a category for matching: strip + lowercase. ``None`` stays ``None``.

    An empty/whitespace-only string also collapses to ``None`` so it can never be
    mistaken for a real category.
    """
    if value is None:
        return None
    text = value.strip().lower()
    return text or None


def same_family(a: str | None, b: str | None, mapping: Mapping[str, str]) -> bool:
    """True only when both categories resolve to the **same, known** family.

    A missing family is ``None``; two misses are not a match (the ``None == None``
    trap). Requires a real, shared family — never absence-equals-absence.
    """
    a_norm = normalize_category(a)
    b_norm = normalize_category(b)
    if a_norm is None or b_norm is None:
        return False
    family_a = mapping.get(a_norm)
    family_b = mapping.get(b_norm)
    return family_a is not None and family_a == family_b


def categorical_score(
    pref: str | None,
    value: str | None,
    mapping: Mapping[str, str],
    *,
    family_credit: float = 0.5,
) -> float:
    """Categorical match: ``1.0`` exact, ``family_credit`` same-family, else ``0.0``.

    Matching is case/whitespace-insensitive, so "Lofi" or " lofi " still lines up
    with a catalog "lofi" instead of silently scoring 0.
    """
    pref_norm = normalize_category(pref)
    value_norm = normalize_category(value)
    if pref_norm is None or value_norm is None:
        return 0.0
    if pref_norm == value_norm:
        return 1.0
    if same_family(pref_norm, value_norm, mapping):
        return family_credit
    return 0.0


def numeric_closeness(target: float | None, value: float | None) -> float | None:
    """Closeness of two 0-1 features: ``1 - |target - value|``, clamped to ``[0, 1]``.

    Returns ``None`` when either side is absent — **not evaluated**, distinct from
    a real ``0.0`` (**evaluated, no overlap**). The caller decides how to treat an
    unknown; this helper never fabricates one.
    """
    if target is None or value is None:
        return None
    return max(0.0, 1.0 - abs(target - value))


def normalize_unit(value: float, low: float, high: float) -> float:
    """Map ``value`` onto ``0-1`` over ``[low, high]``, clamped. For BPM-like scales.

    Kept explicit (no silent domain assumptions): the caller states the real range
    so a feature on a non-``0-1`` scale is converted honestly rather than clamped
    blindly.
    """
    if high <= low:
        raise ValueError("high must exceed low")
    return min(1.0, max(0.0, (value - low) / (high - low)))
