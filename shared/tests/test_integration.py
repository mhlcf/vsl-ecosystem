"""Cross-part integration tests.

Verifies the contract between Part 3 (scoring/levels) and Part 2 (avatar):
both must resolve their signs through the same shared vocabulary + reference
template library, so a sign the game teaches can be animated by the avatar.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vslshared.config import TRAIN_DIR
from vslshared.template_reference import iter_sequences, load_reference_set
from vslshared.vocab import load_vocabulary, vocabulary_by_difficulty

_PART3 = Path(__file__).resolve().parents[2] / "part3-gamification"


@pytest.fixture(scope="module")
def reference() -> dict:
    return load_reference_set()


def _part3_level_for(level_id: int):
    """Import the Part 3 level manager inline (avoids cross-package name clash)."""
    import sys

    if str(_PART3) not in sys.path:
        sys.path.insert(0, str(_PART3))
    from backend.level_manager import level_for

    return level_for(level_id)


def test_part3_level_words_have_reference_data(reference):
    """Every game level must map to a sign the scoring/avatar can animate."""
    total = int(_part3_level_for(30).level_id)
    for lid in range(1, total + 1):
        word = _part3_level_for(lid).sign_name
        assert word in reference, f"level {lid} -> {word!r} has no reference data"


def test_part2_avatar_library_overlaps_part3_scoring(reference):
    """The avatar library is the reference set; Part 3 scores against it."""
    vocab = load_vocabulary()
    sign_names = {str(e["sign_name"]) for e in vocab}
    assert len(sign_names) > 3000
    shared = sign_names & set(reference.keys())
    assert len(shared) >= 3000


def test_reference_templates_are_201_dim(reference):
    for sign, tpl in list(reference.items())[:50]:
        assert tpl.shape == (30, 201), (sign, tpl.shape)


def test_dataset_sample_frames_matching_keypoint_convention():
    from vslshared.template_reference import iter_vocab_dirs

    n = 0
    for _, _ in list(iter_vocab_dirs(TRAIN_DIR))[:5]:
        for seq in iter_sequences(TRAIN_DIR, "b", limit=2):
            assert seq.shape[1] == 201
            n += 1
    assert n > 0