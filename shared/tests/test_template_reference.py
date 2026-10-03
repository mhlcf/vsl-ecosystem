"""Tests for reference template building."""

from __future__ import annotations

import numpy as np

from vslshared.config import FEATURE_DIM, TEMPLATE_TARGET_N, TRAIN_DIR
from vslshared.template_reference import (
    center_sequence,
    has_hands,
    iter_sequences,
    resample_seq,
    template_from_samples,
)


def test_resample_up_and_down():
    short = np.zeros((20, FEATURE_DIM), dtype=np.float32)
    long = np.zeros((90, FEATURE_DIM), dtype=np.float32)
    assert resample_seq(short, TEMPLATE_TARGET_N).shape == (TEMPLATE_TARGET_N, FEATURE_DIM)
    assert resample_seq(long, TEMPLATE_TARGET_N).shape == (TEMPLATE_TARGET_N, FEATURE_DIM)
    assert resample_seq(short).shape == (TEMPLATE_TARGET_N, FEATURE_DIM)


def test_center_sequence_removes_wrist_position():
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(TEMPLATE_TARGET_N, FEATURE_DIM)).astype(np.float32)
    # shift all x of the left hand block by +1m -> centered version unchanged
    pose_dim = 75
    hand = 63
    boxed = seq.copy()
    boxed[:, pose_dim:pose_dim + hand:3] += 0.5
    c1 = center_sequence(seq)
    c2 = center_sequence(boxed)
    assert np.allclose(c1, c2, atol=1e-5)


def test_has_hands():
    zero = np.zeros((10, FEATURE_DIM), dtype=np.float32)
    assert not has_hands(zero)
    hands = np.zeros((10, FEATURE_DIM), dtype=np.float32)
    hands[:, 80] = 1.0  # inside the left-hand block
    assert has_hands(hands)


def test_iter_sequences_and_template(sample_sign_name):
    seqs = list(iter_sequences(TRAIN_DIR, sample_sign_name, limit=5))
    assert seqs, "expected usable sequences for the sample sign"
    assert seqs[0].shape[1] == FEATURE_DIM
    tpl = template_from_samples(seqs)
    assert tpl.shape == (TEMPLATE_TARGET_N, FEATURE_DIM)
    assert tpl.dtype == np.float32