"""Level select screen: grid of levels with locked/unlocked + stars."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from backend.level_manager import TOTAL_LEVELS, level_is_unlocked, level_for
from backend.user_manager import UserManager
from vslshared.db import Database


class _LevelCard(QFrame):
    def __init__(self, app, level_id: int, unlocked: bool, stars: int) -> None:
        super().__init__()
        self.app = app
        self.level_id = level_id
        self.setObjectName("card")
        layout = QVBoxLayout(self)

        number = QLabel(f"Cấp {level_id}")
        number.setObjectName("subtitle")
        layout.addWidget(number)

        level = level_for(level_id)
        name = QLabel(level.word)
        name.setWordWrap(True)
        name.setStyleSheet("font-size:16px; font-weight:600;")
        layout.addWidget(name)

        stars_label = QLabel("★" * stars + "☆" * (3 - stars))
        stars_label.setObjectName("subtitle")
        stars_label.setStyleSheet("color:#ffd166;")
        layout.addWidget(stars_label)

        self.btn = QPushButton("Chơi" if unlocked else "Khoá")
        self.btn.setEnabled(unlocked)
        self.btn.clicked.connect(lambda: self.app.start_level(self.level_id))
        layout.addWidget(self.btn)
        layout.addStretch(1)


class LevelSelectScreen(QWidget):
    RUBRIC = "S4 (luôn thành thạo >4 lần/chương). Hoàn thành để mở cấp kế tiếp."

    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        self._users = UserManager(Database())
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 24, 40, 24)

        head = QHBoxLayout()
        back = QPushButton("← Trang chủ")
        back.clicked.connect(lambda: self.app.show_screen("dashboard"))
        head.addWidget(back)
        title = QLabel("Chọn cấp độ")
        title.setObjectName("title")
        head.addWidget(title, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        head.setStretch(0, 0)
        head.setStretch(1, 1)
        head.setStretch(2, 0)
        root.addLayout(head)

        hint = QLabel(self.RUBRIC)
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        grid_widget = QWidget()
        self.grid = QGridLayout(grid_widget)
        self.scroll.setWidget(grid_widget)
        root.addWidget(self.scroll, 1)

    def refresh(self) -> None:
        user = self.app.session.user_id
        stars_by_level: dict[int, int] = {}
        try:
            stars_by_level = {
                level_id: stars for level_id, stars in self._users.stars_per_level(user)
            }
        except Exception:
            stars_by_level = {}
        completed = self._users.completed_levels(user)

        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cards: list[_LevelCard] = []
        for lid in range(1, TOTAL_LEVELS + 1):
            unlocked = level_is_unlocked(completed, lid)
            stars = stars_by_level.get(lid, 0)
            cards.append(_LevelCard(self.app, lid, unlocked, stars))
        cols = 4
        for idx, card in enumerate(cards):
            self.grid.addWidget(card, idx // cols, idx % cols)