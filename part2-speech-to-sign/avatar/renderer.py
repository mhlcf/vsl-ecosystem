"""Keypoint-driven 3D avatar renderer.

Animates the (T, 201) MediaPipe keypoint sequences (pose + two hands) in a
matplotlib 3D viewport embedded in Qt. No external 3D models are needed —
the NPZ sequences *are* the animation library, so any sign that has motion
data in ``av:reference`` can be played back.

Clipping: the view is configured once; each frame draws the limbs + joints
and updates in place for smooth Qt integration.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from vslshared.config import HAND_DIM, POSE_DIM

POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (15, 17), (17, 19), (19, 15),
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27),
    (27, 29), (29, 31), (31, 27), (26, 28), (28, 30), (30, 32), (32, 28),
]
# The 201-dim schema keeps only the FIRST 25 MediaPipe pose landmarks
# (indices 0..24: face + shoulders + arms + hips; no legs) — see
# vslshared.keypoints._landmarks_flat. Bones referencing 25..32 are dropped.
POSE_EDGES_25 = [e for e in POSE_EDGES if max(e) < 25]
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

_POSE_COLOR = (0.55, 0.62, 0.72)
_LEFT_HAND_COLOR = (0.96, 0.42, 0.34)     # warm: "tay trái"
_RIGHT_HAND_COLOR = (0.35, 0.72, 0.95)    # cool: "tay phải"

# Visibility tuning: sign-language watching needs chunky joints/bones.
_POSE_MARKERSIZE = 7
_POSE_LINEWIDTH = 3.5
_HAND_LINEWIDTH = 4.5
_HAND_JOINT_SIZE = 120           # scatter area in points^2 (mid-finger joints)
_HAND_TIP_SIZE = 190             # fingertips + wrist: bigger to read hand shape
_HAND_TIP_IDX = (0, 4, 8, 12, 16, 20)
_HAND_JOINT_EDGE = "white"
_HAND_JOINT_EDGEWIDTH = 1.5
# Face (11 MediaPipe points in a tiny area) is drawn small + faded so it
# reads as background, not a blob competing with the hands.
_FACE_IDX = list(range(0, 11))
_BODY_IDX = list(range(11, 25))
_FACE_JOINT_SIZE = 25
_FACE_ALPHA = 0.55

# Default camera: near-frontal with a slight tilt so fingers separate
# in depth instead of stacking exactly on top of each other.
_CAM_ELEV = 10
_CAM_AZIM = -90
# Tight framing around the signing space (measured data ranges:
# x in [-0.06, 0.93], y-flipped in [-0.23, 1.13], z in [-1.32, 0.40]).
_VIEW_LIMITS = ((-0.15, 1.05), (-0.3, 1.2), (-1.5, 0.6))


class AvatarPlayback(QObject):
    """Drives clip playback independently of the Qt widget."""

    frame_changed = pyqtSignal(int)        # global frame index
    finished = pyqtSignal()

    def __init__(self, fps: int = 30, parent=None) -> None:
        super().__init__(parent)
        self._clips: list[np.ndarray] = []
        self._bounds: list[tuple[int, int]] = []
        self._current = 0
        self._global_frame = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._set_fps(fps)

    def load_clips(self, clips: list[np.ndarray], fps: int | None = None) -> None:
        self.stop()
        self._clips = [np.asarray(c, dtype=np.float32) for c in clips]
        acc = 0
        self._bounds = []
        for c in self._clips:
            self._bounds.append((acc, acc + c.shape[0]))
            acc += c.shape[0]
        if fps:
            self._set_fps(fps)
        self._current = 0
        self._global_frame = 0

    @property
    def clips(self) -> list[np.ndarray]:
        return self._clips

    @property
    def current_frame(self) -> int:
        return self._global_frame

    def current_keypoints(self) -> np.ndarray | None:
        if not self._clips:
            return None
        clip_idx = self._current
        start, _ = self._bounds[clip_idx]
        return self._clips[clip_idx][self._global_frame - start]

    def play(self) -> None:
        if self._clips:
            self._timer.start()

    def play_new(self, clips: list[np.ndarray], fps: int | None = None) -> None:
        self.load_clips(clips, fps)
        self.play()

    def pause(self) -> None:
        self._timer.stop()

    def stop(self) -> None:
        self._timer.stop()
        self._current = 0
        self._global_frame = 0

    def _tick(self) -> None:
        if not self._clips:
            return
        start, end = self._bounds[self._current]
        if self._global_frame >= end - 1:
            self._current += 1
            if self._current >= len(self._clips):
                self.finished.emit()
                self.stop()
                return
            start, end = self._bounds[self._current]
            self._global_frame = start
        self._global_frame += 1
        self.frame_changed.emit(self._global_frame)

    def _set_fps(self, fps: int) -> None:
        self._timer.setInterval(max(10, int(1000 / max(fps, 1))))


class KeypointAvatar:
    """Pure drawing logic: render one (T,201)-style keypoint frame.

    Separated from the widget so it is unit-testable with matplotlib's
    Agg backend (no GUI event loop required).
    """

    def __init__(self, figure) -> None:
        self.fig = figure
        self.ax = figure.add_subplot(projection="3d")
        self.ax.view_init(elev=_CAM_ELEV, azim=_CAM_AZIM)
        self.ax.set_axis_off()

        self._hand_meshes: list[list] = [[], []]
        self._hand_joints: list = []
        self._pose_mesh = None
        self._pose_joints = None
        self._face_joints = None
        self._head_ring = None

    def rebuild(self, frame: np.ndarray) -> None:
        """(Re)create all drawing artists from one keypoint frame."""
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        pose, left, right = split_frame(frame)
        self.ax.clear()
        self.ax.view_init(elev=_CAM_ELEV, azim=_CAM_AZIM)
        self.ax.set_axis_off()
        self.ax.set_xlim(*_VIEW_LIMITS[0])
        self.ax.set_ylim(*_VIEW_LIMITS[1])
        self.ax.set_zlim(*_VIEW_LIMITS[2])

        p = transform(pose)
        self._pose_mesh = Line3DCollection(
            _segments(p, POSE_EDGES_25), color=_POSE_COLOR,
            linewidth=_POSE_LINEWIDTH)
        self.ax.add_collection3d(self._pose_mesh)
        body = p[_BODY_IDX]
        self._pose_joints = self.ax.scatter(
            body[:, 0], body[:, 1], body[:, 2], color=_POSE_COLOR,
            s=_POSE_MARKERSIZE ** 2, depthshade=False,
            edgecolors=_HAND_JOINT_EDGE, linewidths=_HAND_JOINT_EDGEWIDTH)
        face = p[_FACE_IDX]
        self._face_joints = self.ax.scatter(
            face[:, 0], face[:, 1], face[:, 2], color=_POSE_COLOR,
            s=_FACE_JOINT_SIZE, alpha=_FACE_ALPHA, depthshade=False,
            edgecolors=_HAND_JOINT_EDGE, linewidths=0.5)
        ring = _head_ring(p[0], _head_radius(p))
        self._head_ring = self.ax.plot(
            ring[:, 0], ring[:, 1], ring[:, 2], "-",
            color=_POSE_COLOR, linewidth=2, alpha=0.6)[0]

        self._hand_meshes = [[], []]
        for idx, hand in enumerate((left, right)):
            h = transform(hand)
            color = _LEFT_HAND_COLOR if idx == 0 else _RIGHT_HAND_COLOR
            mesh = Line3DCollection(
                _segments(h, HAND_EDGES), color=color,
                linewidth=_HAND_LINEWIDTH, alpha=0.9)
            self.ax.add_collection3d(mesh)
            mid_idx = [i for i in range(len(h)) if i not in _HAND_TIP_IDX]
            joints_mid = self.ax.scatter(
                h[mid_idx, 0], h[mid_idx, 1], h[mid_idx, 2],
                color=color, s=_HAND_JOINT_SIZE, depthshade=False,
                edgecolors=_HAND_JOINT_EDGE, linewidths=_HAND_JOINT_EDGEWIDTH)
            tip_idx = list(_HAND_TIP_IDX)
            joints_tip = self.ax.scatter(
                h[tip_idx, 0], h[tip_idx, 1], h[tip_idx, 2],
                color=color, s=_HAND_TIP_SIZE, depthshade=False,
                edgecolors=_HAND_JOINT_EDGE, linewidths=_HAND_JOINT_EDGEWIDTH)
            self._hand_meshes[idx] = [mesh, joints_mid, joints_tip]

    def draw_frame(self, frame: np.ndarray) -> None:
        """Fast in-place update of an existing scene."""
        pose, left, right = split_frame(frame)
        p = transform(pose)
        if self._pose_mesh is not None:
            self._pose_mesh.set_segments(_segments(p, POSE_EDGES_25))
        if self._pose_joints is not None:
            body = p[_BODY_IDX]
            self._pose_joints._offsets3d = (body[:, 0], body[:, 1], body[:, 2])
        if self._face_joints is not None:
            face = p[_FACE_IDX]
            self._face_joints._offsets3d = (face[:, 0], face[:, 1], face[:, 2])
        if self._head_ring is not None:
            ring = _head_ring(p[0], _head_radius(p))
            self._head_ring.set_data(ring[:, 0], ring[:, 1])
            self._head_ring.set_3d_properties(ring[:, 2])
        for idx, hand in enumerate((left, right)):
            if not self._hand_meshes[idx]:
                continue
            h = transform(hand)
            mesh, joints_mid, joints_tip = self._hand_meshes[idx]
            mesh.set_segments(_segments(h, HAND_EDGES))
            mid_idx = [i for i in range(len(h)) if i not in _HAND_TIP_IDX]
            joints_mid._offsets3d = (h[mid_idx, 0], h[mid_idx, 1],
                                     h[mid_idx, 2])
            tip_idx = list(_HAND_TIP_IDX)
            joints_tip._offsets3d = (h[tip_idx, 0], h[tip_idx, 1],
                                     h[tip_idx, 2])

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.fig)


def _segments(points: np.ndarray, edges: list[tuple[int, int]]) -> list:
    """Build (N, 2, 3) bone segments from joint positions + edge index pairs."""
    pts = np.asarray(points).reshape(-1, 3)
    return [(pts[a], pts[b]) for a, b in edges]


def _head_ring(nose: np.ndarray, radius: float, n: int = 32) -> np.ndarray:
    """Circle around the nose suggesting the head (first mannequin brick)."""
    radius = min(max(float(radius), 1e-3), 0.15)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return np.column_stack([
        nose[0] + radius * np.cos(t),
        nose[1] + radius * np.sin(t),
        np.full(n, nose[2]),
    ])


def _head_radius(p: np.ndarray) -> float:
    """Head radius from nose->ear distance measured in the image (XY) plane.

    The depth (z) channel of MediaPipe face landmarks is too noisy for
    sizing — the 3D nose->ear distance can exceed the shoulder width.
    """
    d = max(float(np.linalg.norm(p[0, :2] - p[7, :2])),
            float(np.linalg.norm(p[0, :2] - p[8, :2])))
    return d


def split_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split keypoints into (pose, left hand, right hand) column blocks.

    Accepts a single ``(201,)`` vector or a ``(T, 201)`` sequence.
    """
    if frame.ndim == 1:
        frame = frame[np.newaxis, :]
    return (
        frame[:, :POSE_DIM],
        frame[:, POSE_DIM:POSE_DIM + HAND_DIM],
        frame[:, POSE_DIM + HAND_DIM:],
    )


def transform(xyz: np.ndarray) -> np.ndarray:
    """MediaPipe normalized coords -> plot coords (y flipped for display).

    ``xyz`` may be a single vector or a (T, 3) block; the result keeps
    the same leading shape (a mediapipe block is already (N, 3)).
    """
    out = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    out = out.copy()
    out[:, 1] = 1.0 - out[:, 1]
    return out