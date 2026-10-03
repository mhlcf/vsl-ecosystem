"""Dashboard screen: user greeting, progress summary, entry actions."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from backend.user_manager import UserManager
from vslshared.db import Database


class DashboardScreen(QWidget):
    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        self._users = UserManager(Database())
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 32, 48, 32)

        title = QLabel("Học VSL cùng AI")
        title.setObjectName("title")
        subtitle = QLabel("Luyện tập ngôn ngữ ký hiệu Việt Nam qua trò chơi")
        subtitle.setObjectName("subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)
        root.addSpacing(24)

        # identity panel
        panel = QFrame()
        panel.setObjectName("card")
        pl = QVBoxLayout(panel)
        pl.addWidget(QLabel("Chọn biệt danh để bắt đầu"))
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("nhập tên của bạn...")
        self.continue_btn = QPushButton("Vào học")
        self.continue_btn.setObjectName("primary")
        self.continue_btn.clicked.connect(self._on_continue)
        pl.addWidget(self.username_input)
        pl.addWidget(self.continue_btn)
        root.addWidget(panel)

        self.status = QLabel("")
        self.status.setObjectName("subtitle")
        root.addWidget(self.status)
        root.addStretch(1)

    def _on_continue(self) -> None:
        name = self.username_input.text().strip()
        if not name:
            self.status.setText("Vui lòng nhập tên.")
            return
        user = self._users.get_or_create_user(name)
        self.app.session.user_id = user["id"]
        self.app.session.username = user["username"]
        self.app.new_game()

    def refresh(self) -> None:
        if self.app.session.username:
            self.status.setText(f"Xin chào, {self.app.session.username}!")
        self.username_input.clear()
        self.username_input.setFocus()