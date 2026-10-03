"""Tests for the avatar controller (animation sequencing)."""

from __future__ import annotations

import numpy as np
import pytest

from p2backend.avatar_controller import (
    animation_frames,
    build_animation_sequence,
    controls,
    load_animation_library,
    normalize_sign_key,
    resolve_sign,
)


def test_normalize_sign_key():
    assert normalize_sign_key("Xin Chào") == "xin_chao"
    assert normalize_sign_key("Cảm ơn") == "cam_on"


def test_resolve_sign_found_in_vocabulary():
    # 'b' is a real label in the reference vocabulary
    entry = resolve_sign("b")
    assert entry is not None
    assert "animation_id" in entry


def test_resolve_sign_unknown_returns_none():
    assert resolve_sign("@#$%^") is None or resolve_sign("xyz_zzz") is None


def test_animation_frames_shape():
    lib = load_animation_library()
    sign = next(iter(lib))             # e.g. '0 (số không)'
    frames = animation_frames(sign)
    assert frames is not None
    assert frames.ndim == 2
    assert frames.shape[1] == 201


def test_build_animation_sequence_known_word():
    entry = resolve_sign("b")           # real label with motion data
    assert entry is not None
    sign_name = entry["sign_name"]
    steps = build_animation_sequence([sign_name], duration_ms=1500)
    assert len(steps) == 1
    step = steps[0]
    assert step["word"] == sign_name
    assert step["duration_ms"] == 1500
    assert step["animation_id"] is not None
    assert step["parameters"]["frame_count"] > 0
    assert step["parameters"]["template_key"].startswith("reference://")
    assert step["parameters"]["frames"] is not None


def test_build_animation_sequence_missing_sign():
    steps = build_animation_sequence(["không_tồn_tại_word"], duration_ms=1000)
    assert len(steps) == 1
    assert steps[0]["animation_id"] is None or steps[0]["parameters"]["frame_count"] == 0


def test_controls_metadata():
    steps = build_animation_sequence(["b", "10"], duration_ms=2000)
    ctrl = controls(steps)
    assert ctrl["total_clips"] == 2
    assert ctrl["estimated_duration_ms"] >= 4000


def test_neutral_pose_shape():
    from p2backend.avatar_controller import neutral_pose

    pose = neutral_pose()
    assert pose.ndim == 2
    assert pose.shape[1] == 201
    assert pose.shape[0] == 30
    assert pose.dtype == np.float32


def test_build_sequence_merges_multiword_phrase():
    """'chúc mừng' is one reference-backed phrase → single animation step."""
    steps = build_animation_sequence(["chúc", "mừng"], duration_ms=1000)
    assert len(steps) == 1
    assert steps[0]["word"] == "chúc mừng"
    assert steps[0]["parameters"]["frame_count"] > 0


def test_build_sequence_merges_only_known_phrases():
    steps = build_animation_sequence(["chúc", "mừng", "năm", "mới"],
                                     duration_ms=1000)
    signed = [s["word"] for s in steps if s["parameters"]["frame_count"] > 0]
    assert "chúc mừng" in signed


def test_build_sequence_missing_sign_is_frameless_and_holdable():
    steps = build_animation_sequence(["không_có_mẫu_xyz"], duration_ms=1000)
    assert steps[0]["parameters"]["frame_count"] == 0
    from p2backend.avatar_controller import neutral_pose

    assert neutral_pose().shape == (30, 201)