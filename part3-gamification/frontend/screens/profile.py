"""Profile screen: overall stats + export progress to CSV."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from backend.user_manager import UserManager
from vslshared.config import DATA_DIR
from vslshared.db import Database


class ProfileScreen(QWidget):
    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        self._users = UserManager(Database())
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 32, 48, 32)

        title = QLabel("Hồ sơ học tập")
        title.setObjectName("title")
        root.addWidget(title)

        self.stat_frame = QFrame()
        self.stat_frame.setObjectName("card")
        self.stat_layout = QVBoxLayout(self.stat_frame)
        self.stat_user = QLabel("")
        self.stat_user.setObjectName("title")
        self.stat_layout.addWidget(self.stat_user)
        for key in ("levels", "stars", "points"):
            lbl = QLabel("")
            lbl.setObjectName("stat")
            self.stat_layout.addWidget(lbl)
            setattr(self, f"stat_{key}", lbl)
        root.addWidget(self.stat_frame)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("subtitle")
        self.progress_label.setWordWrap(True)
        root.addWidget(self.progress_label)

        buttons = QHBoxLayout()
        export_btn = QPushButton("Xuất CSV")
        export_btn.clicked.connect(self._export_csv)
        back_btn = QPushButton("← Trang chủ")
        back_btn.clicked.connect(lambda: self.app.show_screen("dashboard"))
        buttons.addWidget(export_btn)
        buttons.addWidget(back_btn)
        buttons.addStretch(1)
        root.addLayout(buttons)
        root.addStretch(1)

    def refresh(self) -> None:
        uid = self.app.session.user_id
        if not uid:
            self.app.show_screen("dashboard")
            return
        stats = self._users.stats(uid)
        user = self._users.get_user(uid)
        name = user["username"] if user else str(uid)
        self.stat_user.setText(f"Người chơi: {name}")
        self.stat_levels.setText(
            f"Cấp đã hoàn thành: {stats['total_levels_completed']}")
        self.stat_stars.setText(f"Tổng sao: {stats['total_stars']}")
        self.stat_points.setText(f"Tổng điểm: {stats['total_points']}")
        rows = [f"Cấp {p['level_id']}: {p['stars_earned']}★ ({p['best_accuracy']:.2f})"
                for p in stats["progress"]]
        self.progress_label.setText(
            "Tiến trình: " + (" • ".join(rows) if rows else "chưa hoàn thành cấp nào"))

    def _export_csv(self) -> None:
        uid = self.app.session.user_id
        stats = self._users.stats(uid)
        default = str(DATA_DIR / "progress.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Lưu tiến trình", default, "CSV (*.csv)")
        if not path:
            return
        lines = ["user_id,level_id,stars_earned,best_accuracy,attempts,completed_at"]
        for p in stats["progress"]:
            lines.append(
                f"{uid},{p['level_id']},{p['stars_earned']},"
                f"{p['best_accuracy']:.4f},{p['attempts']},{p['completed_at']}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")