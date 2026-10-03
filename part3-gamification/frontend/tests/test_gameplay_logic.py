"""Headless tests for the in-process gameplay decision loop."""

from __future__ import annotations

import numpy as np
import pytest

from p3backend.level_manager import Level
from p3backend.user_manager import UserManager
from vslshared.config import TRAIN_DIR
from vslshared.template_reference import iter_sequences

from p3frontend.main import VSLApp


class StubCamera:
    """Fake CameraWorker returning a fixed dataset sequence."""

    def __init__(self, sign: str) -> None:
        self._seq = next(iter_sequences(TRAIN_DIR, sign, limit=1))
        self._last_window = None

    def window(self) -> np.ndarray:
        win = self._seq if self._seq.shape[0] >= 60 else np.concatenate(
            [self._seq] * int(np.ceil(60 / self._seq.shape[0])))[:60]
        self._last_window = win
        return win

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


def _make_app(qapp, user_id: int) -> VSLApp:
    app = VSLApp()
    app.session.user_id = user_id
    app.session.username = "tester"
    return app


@pytest.mark.parametrize("sign,expected_correct", [("b", True), ("10", True)])
def test_score_window_detects_correct(qapp, sign, expected_correct):
    um = UserManager()
    user = um.get_or_create_user(f"gameuser_{sign}")
    app = _make_app(qapp, user["id"])
    app.session.level = Level(level_id=1, difficulty=1, chapter="test",
                              sign_name=sign, sign_key=sign)
    app.gameplay._camera = StubCamera(sign)
    app.gameplay._score_timer.stop()

    app.gameplay._score_window()

    if expected_correct is False:
        assert app.session.completed is False
        return
    assert app.session.completed is True
    assert app.session.last_stars >= 1
    # navigation reached the results screen
    assert app.stack.currentWidget() is app.results
    app.gameplay._camera.close()


def test_score_window_never_correct_for_wrong_sign(qapp):
    um = UserManager()
    user = um.get_or_create_user("wrongsign_user")
    app = _make_app(qapp, user["id"])
    app.session.level = Level(level_id=1, difficulty=1, chapter="test",
                              sign_name="b", sign_key="b")
    wrong = next(iter_sequences(TRAIN_DIR, "10", limit=1))

    class Wrong(StubCamera):
        def window(self) -> np.ndarray:
            return wrong if wrong.shape[0] >= 60 else np.concatenate(
                [wrong] * int(np.ceil(60 / wrong.shape[0])))[:60]

    app.gameplay._camera = Wrong("10")
    app.gameplay._score_timer.stop()
    app.gameplay._score_window()

    assert app.session.completed is False
    # feedback should advise fixing the weakest criterion
    assert app.gameplay.feedback_label.text() != ""
    app.gameplay._camera.close()


def test_results_refresh_renders_breakdown(qapp):
    um = UserManager()
    user = um.get_or_create_user("results_user")
    app = _make_app(qapp, user["id"])
    app.session.level = Level(level_id=1, difficulty=1, chapter="test",
                              sign_name="b", sign_key="b")
    app.gameplay._camera = StubCamera("b")
    app.gameplay._score_timer.stop()
    app.gameplay._score_window()
    assert app.session.completed

    app.show_screen("results")
    assert app.results.banner.text() != ""
    for bar in app.results.cards.__dict__.get("_bars", {}).values():
        assert bar.value() <= 100

def test_draw_overlay_guards_empty_input():
    """draw_overlay must tolerate None / empty / all-zero keypoint vectors."""
    import numpy as np

    from p3frontend.camera import draw_overlay

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for bad in (None, np.array([]), np.zeros((201,), dtype=np.float32),
                np.zeros((30, 201), dtype=np.float32)):
        out = draw_overlay(frame, bad)
        assert out.shape == frame.shape
        assert out.dtype == frame.dtype
