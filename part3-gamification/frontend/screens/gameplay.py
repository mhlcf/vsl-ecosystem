"""Gameplay screen: live webcam, target word, 4-criteria scoring, feedback."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from backend.feedback_generator import generate_feedback
from backend.level_manager import stars_for_accuracy
from backend.scoring_engine import classify_live

try:
    from frontend.camera import CameraWorker
except ModuleNotFoundError:
    from ..camera import CameraWorker

CRITERIA_XLABELS = (("hand_shape", "Hình dạng tay"),
                    ("position", "Vị trí"),
                    ("movement", "Chuyển động"),
                    ("palm_orientation", "Hướng lòng bàn tay"))


class GameplayScreen(QWidget):
    ACCEPTABLE_FRAMES = 60

    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        self._camera: CameraWorker | None = None
        self._score_timer = QTimer(self)
        self._score_timer.setInterval(150)
        self._score_timer.timeout.connect(self._score_window)
        self._build()

    # -- layout --------------------------------------------------------------
    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)

        left = QVBoxLayout()
        self.camera_view = QLabel("Khởi động camera...")
        self.camera_view.setObjectName("card")
        self.camera_view.setMinimumSize(640, 480)
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.camera_view, 1)
        root.addLayout(left, 3)

        right = QFrame()
        right.setObjectName("card")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(24, 24, 24, 24)

        self.word_label = QLabel("")
        self.word_label.setObjectName("word")
        self.word_label.setWordWrap(True)
        rl.addWidget(self.word_label)

        self.instruction = QLabel("")
        self.instruction.setObjectName("subtitle")
        self.instruction.setWordWrap(True)
        rl.addWidget(self.instruction)

        rl.addWidget(QLabel("Chấm điểm theo từng tiêu chí"))
        self.bars: dict[str, QProgressBar] = {}
        for crit, label in CRITERIA_XLABELS:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat(f"{label}: %p%")
            self.bars[crit] = bar
            rl.addWidget(bar)

        self.feedback_label = QLabel("")
        self.feedback_label.setObjectName("feedback")
        self.feedback_label.setWordWrap(True)
        rl.addWidget(self.feedback_label)

        self.attempt_label = QLabel("")
        self.attempt_label.setObjectName("stat")
        rl.addWidget(self.attempt_label)

        self.correct_label = QLabel("")
        self.correct_label.setObjectName("stat")
        rl.addWidget(self.correct_label)

        buttons = QHBoxLayout()
        self.stop_btn = QPushButton("Dừng")
        self.stop_btn.clicked.connect(self._on_stop)
        buttons.addWidget(self.stop_btn)
        rl.addLayout(buttons)
        rl.addStretch(1)

        root.addWidget(right, 2)

    # -- lifecycle -----------------------------------------------------------
    def refresh(self) -> None:
        level = self.app.session.level
        if level is None:
            self.app.show_screen("levels")
            return
        self.word_label.setText(level.word)
        self.instruction.setText(
            f"Cấp {level.level_id} • Hãy ký từ “{level.word}”. "
            "Giơ tay trước camera và lặp lại cử chỉ cho đến khi đạt 3 sao.")
        self.attempt_label.setText("Khởi động…")
        self.feedback_label.setText("")
        self.correct_label.setText("")
        for bar in self.bars.values():
            bar.setValue(0)
        self._start_camera()

    def _start_camera(self) -> None:
        self._stop_camera()
        cam = CameraWorker()
        if not cam.start():
            self.feedback_label.setText(
                "Không tìm thấy webcam. Kiểm tra kết nối và thử lại.")
            self.attempt_label.setText("Camera lỗi")
            return
        self._camera = cam
        cam.frame_ready.connect(self._on_frame)
        cam.disconnected.connect(lambda: self._on_camera_lost())
        self._score_timer.start()
        self.attempt_label.setText(
            f"Đang thu {self.ACCEPTABLE_FRAMES} khung hình để chấm điểm…")

    def _stop_camera(self) -> None:
        self._score_timer.stop()
        if self._camera is not None:
            self._camera.close()
            self._camera = None
        self.camera_view.setPixmap(QPixmap())

    def _on_camera_lost(self) -> None:
        self._score_timer.stop()
        self.feedback_label.setText("Mất tín hiệu camera.")

    def stop(self) -> None:
        self._stop_camera()

    # -- frames / scoring ----------------------------------------------------
    def _on_frame(self, rgb) -> None:
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.camera_view.setPixmap(QPixmap.fromImage(img))

    def _score_window(self) -> None:
        if self._camera is None or self.app.session.level is None:
            return
        if self.app.session.completed:
            return
        win = self._camera.window()
        if win.shape[0] < self.ACCEPTABLE_FRAMES:
            self.attempt_label.setText(
                f"Đang thu dữ liệu… {win.shape[0]}/{self.ACCEPTABLE_FRAMES}")
            return

        self.app.session.attempts += 1
        level = self.app.session.level
        res = classify_live(win, level.word)
        if res is None:
            self.feedback_label.setText("Chưa có mẫu cho từ này.")
            self._camera.reset()
            return

        for crit, value in res["accuracy_breakdown"].items():
            self.bars[crit].setValue(int(value * 100))

        scores = res["accuracy"]
        text, hint = generate_feedback(res["accuracy_breakdown"], res["is_correct"])
        self.feedback_label.setText(text)
        self.attempt_label.setText(f"Lần thử #{self.app.session.attempts} • điểm {scores:.2f}")

        if res["is_correct"]:
            self._on_correct(res)
        self._camera.reset()

    def _on_correct(self, res: dict) -> None:
        self._score_timer.stop()
        self.app.session.completed = True
        stars = stars_for_accuracy(res["accuracy"])
        self.correct_label.setText(
            f"Chính xác! (điểm {res['accuracy']:.2f} → {stars} sao)")
        self.app.finish_level(res, stars)

    def _on_stop(self) -> None:
        self._stop_camera()
        self.app.show_screen("levels")