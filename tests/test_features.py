"""Tests for the shared feature-comparison utilities."""

from __future__ import annotations

import pytest

from src.features import (
    categorical_score,
    normalize_category,
    normalize_unit,
    numeric_closeness,
    same_family,
)

# A tiny family map: two "mellow" members, plus an isolated genre.
FAMILIES = {"lofi": "mellow", "ambient": "mellow", "metal": "heavy"}


def test_normalize_category_strips_lowercases_and_empties_to_none():
    assert normalize_category(" Lofi ") == "lofi"
    assert normalize_category(None) is None
    assert normalize_category("   ") is None  # whitespace-only is not a category


def test_same_family_requires_a_real_shared_family():
    assert same_family("lofi", "ambient", FAMILIES) is True   # both "mellow"
    assert same_family("lofi", "metal", FAMILIES) is False    # different families
    # The core guard: two UNKNOWN genres both map to None, and None == None is
    # True — this must NOT read as a shared family.
    assert same_family("polka", "klezmer", FAMILIES) is False
    assert same_family("polka", None, FAMILIES) is False


def test_categorical_score_exact_family_and_miss():
    assert categorical_score("Lofi", "lofi", FAMILIES) == 1.0      # exact (normalized)
    assert categorical_score("lofi", "ambient", FAMILIES) == 0.5   # same family
    assert categorical_score("lofi", "metal", FAMILIES) == 0.0     # unrelated
    assert categorical_score("polka", "klezmer", FAMILIES) == 0.0  # both unknown, not a match
    assert categorical_score(None, "lofi", FAMILIES) == 0.0        # nothing to compare
    assert categorical_score("jazz", "blues", FAMILIES, family_credit=0.25) == 0.0


def test_numeric_closeness_distinguishes_none_from_zero():
    assert numeric_closeness(0.8, 0.8) == 1.0
    assert numeric_closeness(0.2, 0.9) == pytest.approx(0.3)
    assert numeric_closeness(0.0, 1.0) == 0.0        # evaluated, no overlap
    assert numeric_closeness(None, 0.5) is None      # not evaluated
    assert numeric_closeness(0.5, None) is None


def test_normalize_unit_maps_and_clamps():
    assert normalize_unit(125, 50, 200) == pytest.approx(0.5)  # midpoint
    assert normalize_unit(20, 50, 200) == 0.0                  # below range clamps
    assert normalize_unit(500, 50, 200) == 1.0                 # above range clamps
    with pytest.raises(ValueError, match="high must exceed low"):
        normalize_unit(1.0, 5.0, 5.0)
