"""Tests for the Part 3 level manager."""

from __future__ import annotations

import pytest

from p3backend.level_manager import (
    TOTAL_LEVELS,
    instructions_for,
    level_for,
    level_is_unlocked,
    next_level_id,
    stars_for_accuracy,
)


def test_level_for_structure():
    for lid in [1, 5, 11, 21, TOTAL_LEVELS]:
        lvl = level_for(lid)
        assert lvl.level_id == lid
        assert lvl.chapter in ("beginner", "intermediate", "advanced")
        assert lvl.sign_name
        assert lvl.sign_key
    assert level_for(11).chapter == "intermediate"
    assert level_for(21).chapter == "advanced"


def test_level_for_out_of_range():
    with pytest.raises(ValueError):
        level_for(0)
    with pytest.raises(ValueError):
        level_for(TOTAL_LEVELS + 1)


def test_stars_thresholds():
    assert stars_for_accuracy(0.95) == 3
    assert stars_for_accuracy(0.90) == 3
    assert stars_for_accuracy(0.85) == 2
    assert stars_for_accuracy(0.75) == 1
    assert stars_for_accuracy(0.60) == 0


def test_unlock_logic():
    assert level_is_unlocked([], 1)
    assert not level_is_unlocked([], 2)
    assert level_is_unlocked([1], 2)
    assert not level_is_unlocked([2], 3)   # level 2 must be complete too
    assert level_is_unlocked([1, 2], 3)


def test_next_level():
    assert next_level_id(1) == 2
    assert next_level_id(TOTAL_LEVELS) is None


def test_instructions():
    assert "thực hiện" in instructions_for(level_for(1))


def test_beginner_levels_are_core_word_curriculum():
    """Level 1..10 must be the curated everyday signs, not number entries."""
    expected = ["chào", "bạn", "đi", "cơm", "mới",
                "hoa", "áo", "mũ", "sao", "bơi"]
    for lid, word in enumerate(expected, start=1):
        assert level_for(lid).word == word, f"level {lid}"


def test_all_level_words_have_reference_data():
    from vslshared.template_reference import load_reference_set

    ref = load_reference_set()
    for lid in range(1, TOTAL_LEVELS + 1):
        word = level_for(lid).word
        assert word in ref, f"level {lid} -> {word!r}"


def test_no_level_teaches_dataset_meta_entries():
    """Opening levels must not be raw numeric/math/parenthetical signs."""
    noise = ("1 000 000 000", "(một", "＞", "＜", "Toán học")
    for lid in (1, 2, 3):
        word = level_for(lid).word
        assert not any(n in word for n in noise), word