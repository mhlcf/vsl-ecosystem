"""Level & quest management for Part 3.

Levels are organised into three chapters aligned with the prompt's
difficulty buckets: Beginner (1-10), Intermediate (11-20), Advanced
(21-30). Each level teaches one sign from the shared vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass

import re

from vslshared.config import (
    LEVELS_PER_DIFFICULTY,
    STAR_1_THRESHOLD,
    STAR_2_THRESHOLD,
    STAR_3_THRESHOLD,
)
from vslshared.template_reference import load_reference_set
from vslshared.vocab import load_vocabulary, vocabulary_by_difficulty

# Geometry/grammar characters that mark a "dataset-meta" sign (math, numbers,
# units, parenthetical notes) rather than a word worth teaching first.
_NOISE_RE = re.compile(r"\d|[><%+=×÷()/().!？?：:]|％|＞|＜")


@dataclass(frozen=True)
class Level:
    level_id: int
    difficulty: int          # 1 = beginner, 2 = intermediate, 3 = advanced
    chapter: str             # human chapter name
    sign_name: str           # vocabulary key, e.g. "xin chào"
    sign_key: str            # normalized key, e.g. "xin_chào"

    @property
    def word(self) -> str:
        return self.sign_name


TOTAL_LEVELS = LEVELS_PER_DIFFICULTY * 3

# Beginner curriculum: common everyday signs that exist in the reference
# library, so level 1 is a real word instead of "1 000 000 000 (một tỉ)".
_CORE_LEVELS = [
    "chào", "bạn", "đi", "cơm", "mới",
    "hoa", "áo", "mũ", "sao", "bơi",
]


def _key(sign_name: str) -> str:
    return sign_name.strip().replace(" ", "_")


def _vocab_sorted(difficulty: int) -> list[dict]:
    """Signs for a difficulty bucket the engine can animate.

    Signs without a reference template (no motion data in the dataset) are
    skipped first, guaranteeing every game level is score-able *and*
    playable by the Part 2 avatar with the shared library. Dataset-meta
    signs (numbers/units/math/parenthetical notes) are skipped too, so the
    game teaches real words instead of "1 000 000 000 (một tỉ)".
    """
    ref = load_reference_set()
    def viable(e):
        name = str(e["sign_name"]).strip()
        return name in ref and not _NOISE_RE.search(name)

    pool = [e for e in vocabulary_by_difficulty(difficulty) if viable(e)]
    if len(pool) < LEVELS_PER_DIFFICULTY:
        with_ref = [e for e in vocabulary_by_difficulty(difficulty)
                    if str(e["sign_name"]).strip() in ref]
        pool = pool or with_ref or vocabulary_by_difficulty(difficulty)
    return pool


def level_for(level_id: int) -> Level:
    """Resolve a 1-based ``level_id`` to a concrete sign."""
    if not (1 <= level_id <= TOTAL_LEVELS):
        raise ValueError(f"level_id must be in [1, {TOTAL_LEVELS}]")
    difficulty = (level_id - 1) // LEVELS_PER_DIFFICULTY + 1
    index = (level_id - 1) % LEVELS_PER_DIFFICULTY
    if difficulty == 1:
        sign_name = _CORE_LEVELS[index]
    else:
        entry = _vocab_sorted(difficulty)[index]
        sign_name = str(entry["sign_name"])
    chapter = {1: "beginner", 2: "intermediate", 3: "advanced"}[difficulty]
    return Level(
        level_id=level_id,
        difficulty=difficulty,
        chapter=chapter,
        sign_name=sign_name,
        sign_key=_key(sign_name),
    )


def stars_for_accuracy(accuracy: float) -> int:
    """Map an average accuracy to 0-3 stars."""
    if accuracy >= STAR_3_THRESHOLD:
        return 3
    if accuracy >= STAR_2_THRESHOLD:
        return 2
    if accuracy >= STAR_1_THRESHOLD:
        return 1
    return 0


def level_is_unlocked(completed_levels: list[int], level_id: int) -> bool:
    """A level is unlocked when all earlier levels are completed."""
    completed = set(completed_levels)
    return all(lid in completed for lid in range(1, level_id))


def next_level_id(level_id: int) -> int | None:
    if level_id < TOTAL_LEVELS:
        return level_id + 1
    return None


def instructions_for(level: Level) -> str:
    """Short Vietnamese instruction shown before the attempt."""
    return f"Hãy thực hiện cử chỉ '{level.sign_name}' trước camera."