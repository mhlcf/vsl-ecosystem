"""PyQt6 application shell for Part 3 (Gamification Learning VSL)."""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from backend.level_manager import Level, level_for

from .camera import CameraWorker
from .screens.dashboard import DashboardScreen
from .screens.gameplay import GameplayScreen
from .screens.level_select import LevelSelectScreen
from .screens.profile import ProfileScreen
from .screens.results import ResultsScreen
from .state import GameSession

CANVAS_STYLESHEET = """
QMainWindow { background: #101216; }
QLabel#title { font-size: 22px; font-weight: 600; color: #e8e8e8; }
QLabel#subtitle { color: #8a8f98; }
QLabel#word { font-size: 28px; font-weight: 700; color: #ffd166; }
QLabel#feedback { font-size: 16px; color: #f4f4f4; }
QPushButton { background: #1f242e; color: #e8e8e8; border: 1px solid #333a46;
              border-radius: 8px; padding: 8px 16px; font-size: 14px; }
QPushButton:hover { background: #2b3240; }
QPushButton:disabled { color: #6a6f78; }
QPushButton#primary { background: #2d6cdf; border: none; color: white; }
QPushButton#primary:hover { background: #3b7bea; }
QFrame#card { background: #171b22; border: 1px solid #262c36; border-radius: 10px; }
QProgressBar { border: none; border-radius: 6px; background: #262c36; height: 10px; }
QProgressBar::chunk { background: #2d6cdf; border-radius: 6px; }
QLabel#stat { color: #c7ccd4; }
"""


class VSLApp(QMainWindow):
    """Single window hosting the five game screens via a stack."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Học VSL - Gamification")
        self.resize(1180, 760)
        self.setStyleSheet(CANVAS_STYLESHEET)

        self.session = GameSession()
        self.camera: CameraWorker | None = None

        self.stack = QStackedWidget(self)
        self.dashboard = DashboardScreen(self)
        self.level_select = LevelSelectScreen(self)
        self.gameplay = GameplayScreen(self)
        self.results = ResultsScreen(self)
        self.profile = ProfileScreen(self)
        for screen in (self.dashboard, self.level_select,
                       self.gameplay, self.results, self.profile):
            self.stack.addWidget(screen)
        self.setCentralWidget(self.stack)

    # -- navigation ----------------------------------------------------------
    def show_screen(self, name: str) -> None:
        mapping = {
            "dashboard": self.dashboard,
            "levels": self.level_select,
            "gameplay": self.gameplay,
            "results": self.results,
            "profile": self.profile,
        }
        widget = mapping[name]
        self.stack.setCurrentWidget(widget)
        refresh = getattr(widget, "refresh", None)
        if refresh:
            refresh()

    # -- game flow -----------------------------------------------------------
    def start_level(self, level_id: int) -> None:
        self.session.level = level_for(level_id)
        self.session.attempts = 0
        self.session.completed = False
        self.show_screen("gameplay")

    def finish_level(self, scores: dict, stars: int) -> None:
        self.session.last_scores = scores
        self.session.last_stars = stars
        self.session.completed = True
        self.show_screen("results")

    def new_game(self) -> None:
        self.session.level = None
        self.show_screen("levels")


def run() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("VSL Gamification")
    win = VSLApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()