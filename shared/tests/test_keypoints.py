"""Tests for the keypoint extraction schema."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vslshared.config import FEATURE_DIM, HAND_DIM, POSE_DIM
from vslshared.keypoints import (
    extract_keypoints,
    normalize,
    split_components,
)


def _lm_list(n: int):
    xmlns = SimpleNamespace
    return [
        xmlns(x=float(i) / n, y=float(i % 7) / 7, z=-0.001 * i) for i in range(n)
    ]


def _fake_results(n_pose=25, n_hand=21, include_hands=True):
    res = SimpleNamespace()
    res.pose_landmarks = SimpleNamespace(landmark=_lm_list(n_pose))
    res.left_hand_landmarks = (
        SimpleNamespace(landmark=_lm_list(n_hand)) if include_hands else None
    )
    res.right_hand_landmarks = (
        SimpleNamespace(landmark=_lm_list(n_hand)) if include_hands else None
    )
    return res


def test_extract_keypoints_full():
    vec = extract_keypoints(_fake_results())
    assert vec.shape == (FEATURE_DIM,)
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec))


def test_extract_keypoints_no_hands():
    vec = extract_keypoints(_fake_results(include_hands=False))
    assert vec.shape == (FEATURE_DIM,)
    assert np.all(vec[POSE_DIM:] == 0)


def test_split_components_dims():
    seq = np.zeros((10, FEATURE_DIM), dtype=np.float32)
    pose, left, right = split_components(seq)
    # a representative feature exists in pose block to avoid vacuous pass
    seq[:, 0] = 1.0
    pose, left, right = split_components(seq)
    assert pose.shape == (10, POSE_DIM)
    assert left.shape == (10, HAND_DIM)
    assert right.shape == (10, HAND_DIM)
    assert float(pose[:, 0].sum()) == 10.0


def test_normalize():
    seq = np.full((5, FEATURE_DIM), 100.0, dtype=np.float32)
    out = normalize(seq, np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM))
    assert out.shape == seq.shape
    assert np.allclose(out, seq)