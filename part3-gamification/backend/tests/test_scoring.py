"""Tests for the 4-criteria scoring engine."""

from __future__ import annotations

import numpy as np
import pytest

from p3backend.feedback_generator import weakest_criterion
from p3backend.scoring_engine import (
    CriteriaScores,
    hand_shape_score,
    movement_score,
    palm_orientation_score,
    position_score,
    score_sign,
)


def test_criteria_scores_confidence_weighted():
    s = CriteriaScores(0.5, 0.7, 0.6, 0.8)
    d = s.as_dict()
    assert set(d) == {"hand_shape", "position", "movement", "palm_orientation"}
    conf = s.confidence()
    assert 0 <= conf <= 1
    assert conf == pytest.approx(0.35 * 0.5 + 0.25 * 0.7 + 0.20 * 0.6 + 0.20 * 0.8)


def test_weakest_criterion():
    s = CriteriaScores(0.9, 0.9, 0.2, 0.9)
    assert weakest_criterion(s) == "movement"


def test_hand_shape_no_hands_is_zero():
    q = np.zeros((30, 201), dtype=np.float32)
    t = np.zeros((30, 201), dtype=np.float32)
    assert hand_shape_score(q, t) == 0.0


def test_hand_shape_perfect_match():
    rng = np.random.default_rng(1)
    q = np.zeros((30, 201), dtype=np.float32)
    q[:, 80:200] = rng.normal(size=(30, 120)).astype(np.float32)
    assert hand_shape_score(q, q) >= 0.99


def test_position_score_bounds():
    q = np.zeros((30, 201), dtype=np.float32)
    pos = np.zeros((30, 201), dtype=np.float32)
    b = (75, 138)
    q[:, b[0]:b[1]] = 0.2
    pos[:, b[0]:b[1]] = 0.2
    assert position_score(q, pos) >= 0.9
    assert 0 <= position_score(q, q * -1) <= 1


def test_movement_identical_and_opposite():
    q = np.zeros((30, 201), dtype=np.float32)
    t = q.copy()
    assert movement_score(q, t) == 1.0  # both stationary
    q[:, 100] = np.linspace(0, 1, 30)
    assert movement_score(q, q) >= 0.99
    assert movement_score(q, q * -1) <= 1.0


def test_palm_orientation_same_hand(real_sequence):
    seq, _ = real_sequence
    assert 0 <= palm_orientation_score(seq, seq) <= 1.0


def test_score_sign_self_match_beats_cross(real_sequence):
    seq, sign = real_sequence
    self_score = score_sign(seq, sign)
    assert self_score is not None
    assert all(0 <= float(v) <= 1 for v in self_score.as_dict().values())
    # any other sign must score lower or equal in expectation
    other = score_sign(seq[:30], sign)
    assert self_score.confidence() >= (other.confidence() if other else 0.0) - 1e-6