"""Results screen after completing a level: stars, breakdown, actions."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from backend.level_manager import TOTAL_LEVELS, level_is_unlocked, level_for
from backend.user_manager import UserManager
from vslshared.db import Database


class ResultsScreen(QWidget):
    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        self._users = UserManager(Database())
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 40, 48, 40)

        self.banner = QLabel("")
        self.banner.setObjectName("word")
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.banner)

        self.stars_label = QLabel("")
        self.stars_label.setObjectName("title")
        self.stars_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stars_label.setStyleSheet("font-size:34px; color:#ffd166;")
        root.addWidget(self.stars_label)

        self.level_label = QLabel("")
        self.level_label.setObjectName("subtitle")
        self.level_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.level_label)

        root.addSpacing(20)
        self.cards = QFrame()
        self.cards.setObjectName("card")
        cl = QVBoxLayout(self.cards)
        cl.addWidget(QLabel("Chi tiết điểm số"))
        for crit, label in (("hand_shape", "Hình dạng tay"),
                            ("position", "Vị trí"),
                            ("movement", "Chuyển động"),
                            ("palm_orientation", "Hướng lòng bàn tay")):
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat(f"{label}: %p%")
            self.cards.__dict__.setdefault("_bars", {})[crit] = bar
            cl.addWidget(bar)
        root.addWidget(self.cards)

        self.message = QLabel("")
        self.message.setObjectName("feedback")
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.message)

        buttons = QHBoxLayout()
        self.retry_btn = QPushButton("Chơi lại")
        self.retry_btn.setObjectName("primary")
        self.retry_btn.clicked.connect(self._on_retry)
        self.next_btn = QPushButton("Cấp tiếp theo →")
        self.next_btn.clicked.connect(self._on_next)
        self.levels_btn = QPushButton("Danh sách cấp")
        self.levels_btn.clicked.connect(lambda: self.app.show_screen("levels"))
        buttons.addWidget(self.retry_btn)
        buttons.addWidget(self.next_btn)
        buttons.addWidget(self.levels_btn)
        buttons.addStretch(1)
        root.addLayout(buttons)
        root.addStretch(1)

    def refresh(self) -> None:
        session = self.app.session
        if not session.level or not session.completed:
            self.app.show_screen("levels")
            return

        res = session.last_scores
        stars = session.last_stars
        self.banner.setText("Hoàn thành cấp độ!")
        self.stars_label.setText("★" * stars + "☆" * (3 - stars))
        self.level_label.setText(
            f"Cấp {session.level.level_id} • {session.level.word}")
        breakdown = res.get("accuracy_breakdown", {})
        for crit, bar in self.cards.__dict__.get("_bars", {}).items():
            bar.setValue(int(breakdown.get(crit, 0) * 100))
        self.message.setText(
            f"Điểm số: {res.get('accuracy', 0):.2f} • {session.attempts} lần thử")

        completed = self._users.completed_levels(session.user_id)
        nxt = session.level.level_id + 1
        self.next_btn.setEnabled(nxt <= TOTAL_LEVELS)
        if nxt <= TOTAL_LEVELS:
            self.next_btn.setText(f"Cấp tiếp theo (cấp {nxt}) →")

    def _on_retry(self) -> None:
        level = self.app.session.level
        if level:
            self.app.start_level(level.level_id)

    def _on_next(self) -> None:
        nxt = self.app.session.level.level_id + 1
        if nxt <= TOTAL_LEVELS:
            self.app.start_level(nxt)