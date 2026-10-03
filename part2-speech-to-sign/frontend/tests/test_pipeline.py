"""Headless tests for the Part 2 frontend pipeline + UI smoke."""

from __future__ import annotations

import numpy as np
import pytest

from p2frontend.pipeline import process_text


def test_process_text_greeting():
    result = process_text("Xin chào, bạn khỏe không?")
    structure = result["vsl_structure"]
    assert structure["topic"] == "bạn"
    assert structure["words"] == ["xin", "chào", "bạn", "khỏe", "không"]
    assert len(result["animation_sequence"]) == len(structure["words"])
    assert len(result["clips"]) == len(result["animation_sequence"])


def test_process_text_known_sign_has_clips():
    # 'b' is a real vocabulary sign with reference motion data
    result = process_text("b")
    steps = result["animation_sequence"]
    assert steps
    assert steps[0]["animation_id"] is not None


def test_pipeline_never_returns_empty_clips():
    """Frameless signs become a neutral hold; clips stay aligned to steps."""
    result = process_text("xyzzy hoặc đó")
    steps = result["animation_sequence"]
    assert len(result["clips"]) == len(steps)
    assert all(c is not None and len(c) > 0 for c in result["clips"])
    assert result["missing_animation"]
    assert set(result["missing_animation"]) <= set(result["vsl_structure"]["words"])
    for step in steps:
        assert step["parameters"]["frame_count"] > 0


def test_pipeline_reports_no_missing_for_known_sign():
    result = process_text("b")
    assert result["missing_animation"] == []


def test_speech_to_sign_app_builds_headless(qapp):
    from p2frontend.main import SpeechToSignApp

    win = SpeechToSignApp()
    assert win.windowTitle()
    assert win.viewport is not None
    assert win.history is not None
    win.stop()


def test_viewport_plays_clip_headless(qapp):
    from p2frontend.main import SpeechToSignApp

    win = SpeechToSignApp()
    zeros = np.zeros((30, 201), dtype=np.float32)
    win.viewport.play_clips([zeros], fps=15)
    assert win.viewport._playback.clips
    win.viewport.pause()
    # playback stepping must not raise outside a running event loop
    win.viewport._playback.frame_changed.connect(lambda _: None)
    win.viewport._playback.play()
    win.viewport._playback.stop()
    win.stop()


def test_viewport_replay_no_clips_headless(qapp):
    from p2frontend.main import SpeechToSignApp

    win = SpeechToSignApp()
    win.viewport.replay()       # empty playback must not crash
    win.stop()