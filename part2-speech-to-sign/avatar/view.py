"""Qt viewport embedding the keypoint-driven 3D avatar."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .renderer import AvatarPlayback, KeypointAvatar


class AvatarViewport(QWidget):
    """matplotlib 3D canvas + play/pause controls for a clip sequence."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._canvas = None
        self._avatar: KeypointAvatar | None = None
        self._playback = AvatarPlayback(parent=self)
        self._playback.frame_changed.connect(self._on_frame)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.placeholder = QWidget(self)
        layout.addWidget(self.placeholder, 1)

        controls = QHBoxLayout()
        self.play_btn = QPushButton("▶ Phát")
        self.play_btn.clicked.connect(self.play)
        self.pause_btn = QPushButton("Tạm dừng")
        self.pause_btn.clicked.connect(self.pause)
        self.replay_btn = QPushButton("↺ Lại")
        self.replay_btn.clicked.connect(self.replay)
        self.speed_btn = QPushButton("Tốc độ 1.0×")
        self.speed_btn.clicked.connect(self._cycle_speed)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.replay_btn)
        controls.addWidget(self.speed_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._speeds = [1.0, 0.5, 2.0]
        self._speed = 1.0

    # -- public API ----------------------------------------------------------
    def play_clips(self, clips: list[np.ndarray], fps: int = 30) -> None:
        n_frames = sum(int(c.shape[0]) for c in clips)
        if n_frames == 0:
            return
        if self._avatar is None:
            self._build_canvas()
        assert self._avatar is not None
        self._avatar.rebuild(clips[0][0] if clips[0].ndim == 2 else clips[0][-1])
        self._playback.play_new(clips, fps)
        self.play_btn.setText("▶ Phát")

    def play(self) -> None:
        self._playback.play()

    def pause(self) -> None:
        self._playback.pause()

    def replay(self) -> None:
        self._playback.stop()
        self._playback.load_clips(self._playback.clips)
        self._playback.play()

    def _cycle_speed(self) -> None:
        self._speed = self._speeds[(self._speeds.index(self._speed) + 1)
                                   % len(self._speeds)]
        self.speed_btn.setText(f"Tốc độ {self._speed:.1f}×")
        if self._canvas is not None:
            self._canvas.grab()  # no-op to keep the canvas alive

    # -- internals -----------------------------------------------------------
    def _build_canvas(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as Canvas,
        )
        from matplotlib.figure import Figure

        canvas = Canvas(Figure(figsize=(6, 6), dpi=100))
        self._canvas = canvas
        self._avatar = KeypointAvatar(canvas.figure)
        parent = self.layout().itemAt(0).widget()
        parent.deleteLater()
        self.layout().removeWidget(parent)
        self.layout().insertWidget(0, canvas, 1)

    def _on_frame(self, _global_frame: int) -> None:
        kp = self._playback.current_keypoints()
        if kp is None or self._avatar is None:
            return
        self._avatar.draw_frame(kp)
        if self._canvas is not None:
            self._canvas.draw_idle()