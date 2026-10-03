"""Webcam worker + skeleton/feedback overlay for the Part 3 game.

Runs the camera capture in a background ``QThread`` so the UI stays
responsive, and renders the MediaPipe skeleton with red highlighting on
the region that needs fixing (position/zone from the scoring engine).
"""

from __future__ import annotations

import threading

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from vslshared.keypoints import HolisticExtractor, extract_keypoints

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (17, 19), (19, 15),
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (11, 23),
    (12, 24), (23, 24), (23, 25), (24, 26),
]
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
_BONE_COLOR = (200, 255, 200)
_JOINT_COLOR = (0, 255, 0)
_ERROR_COLOR = (0, 0, 255)
_CORRECT_COLOR = (0, 255, 0)


def _scale(points: np.ndarray, w: int, h: int) -> np.ndarray:
    """Normalized (x, y, z) points -> (x, y) pixel coordinates."""
    xy = points[:, :2]
    return np.column_stack([xy[:, 0] * w, (1.0 - xy[:, 1]) * h]).astype(int)


def draw_overlay(frame: np.ndarray, keypoints: np.ndarray,
                 error_hand: str | None = None,
                 show_correct: bool = False) -> np.ndarray:
    """Draw skeleton + green/red zones on a copy of the BGR frame."""
    import cv2

    out = frame.copy()
    if keypoints is None or len(keypoints) == 0:
        return out
    if keypoints.ndim == 2:
        keypoints = keypoints[-1]
    if keypoints.shape[0] < 75 or not np.any(keypoints):
        return out

    pose = keypoints[:75].reshape(25, 3)
    left = keypoints[75:138].reshape(21, 3)
    right = keypoints[138:].reshape(21, 3)

    h, w = _frame_size(out)
    for hand in (left, right):
        pts = _scale(hand, w, h)
        for a, b in _HAND_CONNECTIONS:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), _BONE_COLOR, 2)
        for p in pts:
            cv2.circle(out, tuple(p), 2, _JOINT_COLOR, -1)

    p_pts = _scale(pose, w, h)
    for a, b in _POSE_CONNECTIONS:
        cv2.line(out, tuple(p_pts[a]), tuple(p_pts[b]), _BONE_COLOR, 1, cv2.LINE_AA)

    if error_hand in ("left", "both"):
        _highlight_hand(out, left, w, h)
    if error_hand in ("right", "both"):
        _highlight_hand(out, right, w, h)
    if show_correct and keypoints.shape[0] >= 75:
        cv2.circle(out, (w - 40, 40), 18, _CORRECT_COLOR, 4)
    return out


def _frame_size(frame: np.ndarray) -> tuple[int, int]:
    return frame.shape[0], frame.shape[1]


def _highlight_hand(out, hand: np.ndarray, w: int, h: int) -> None:
    import cv2

    pts = _scale(hand, w, h)
    center = pts.mean(axis=0).astype(int)
    radius = int(0.35 * min(w, h))
    cv2.circle(out, tuple(center), radius, _ERROR_COLOR, 4)


class CameraWorker(QObject):
    """Captures webcam frames, extracts keypoints, emits annotated frames."""

    frame_ready = pyqtSignal(np.ndarray)     # RGB frame (and copy)
    disconnected = pyqtSignal()

    def __init__(self, camera_index: int = 0, parent=None) -> None:
        super().__init__(parent)
        self._index = camera_index
        self._running = False
        self._capture_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._window: list[np.ndarray] = []
        self._extractor = HolisticExtractor()

    # -- public API ----------------------------------------------------------
    def start(self) -> bool:
        import cv2

        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            return False
        self._cap = cap
        self._running = True
        self._capture_thread = threading.Thread(target=self._run, daemon=True)
        self._capture_thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)
        self._capture_thread = None
        if hasattr(self, "_cap"):
            self._cap.release()

    def window(self) -> np.ndarray:
        with self._lock:
            if not self._window:
                return np.zeros((1, 201), dtype=np.float32)
            return np.stack(self._window, axis=0).astype(np.float32)

    def reset(self) -> None:
        with self._lock:
            self._window.clear()

    def close(self) -> None:
        self.stop()
        self._extractor.close()

    # -- worker loop ---------------------------------------------------------
    def _run(self) -> None:                      # pragma: no cover - threading
        import cv2

        while self._running:
            try:
                ret, frame = self._cap.read()
            except Exception:
                break
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                results = self._extractor.process(rgb)
                vec = extract_keypoints(results)
            except Exception:
                continue
            with self._lock:
                self._window.append(vec)
                if len(self._window) > 60:
                    self._window = self._window[-60:]
            annotated = draw_overlay(frame, vec)
            self.frame_ready.emit(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        self.disconnected.emit()
        self._running = False