"""Part 3 UI screenshots, rendered offscreen (no display/webcam needed).

Usage:  venv/bin/python tools/make_part3_screenshots.py [outdir]
Outputs PNGs into docs/screenshots/ (default).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = (Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/screenshots")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_fd, _db = tempfile.mkstemp(suffix=".db")
os.close(_fd)
os.unlink(_db)
os.environ["VSL_DB_PATH"] = _db  # seed data goes to a throwaway DB

sys.path.insert(0, str(ROOT / "part3-gamification"))

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)

from backend.level_manager import TOTAL_LEVELS, level_for  # noqa: E402
from backend.user_manager import UserManager  # noqa: E402
from frontend.main import VSLApp  # noqa: E402
from vslshared.db import Database  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)

um = UserManager(Database())
user = um.get_or_create_user("Minh")
uid = user["id"]
for lid in (1, 2, 3):
    um.record_completion(uid, lid, stars=min(lid, 3),
                         accuracy=0.5 + lid * 0.1, attempts=lid)

win = VSLApp()
win.session.user_id = uid
win.session.username = "Minh"
win.show()

QApplication.processEvents()

# dashboard (set the name in the input for a nicer shot)
win.show_screen("dashboard")
win.dashboard.username_input.setText("Minh")
QApplication.processEvents()
win.grab().save(str(OUT / "part3_dashboard.png"))

# level select (levels 1-3 unlocked with stars)
win.show_screen("levels")
QApplication.processEvents()
win.grab().save(str(OUT / "part3_levels.png"))

# gameplay idle (level 5; camera will report "not found" offscreen)
win.start_level(5)
QApplication.processEvents()
win.grab().save(str(OUT / "part3_gameplay.png"))

# results (seeded scores)
win.session.level = level_for(3)
win.session.attempts = 2
win.session.last_scores = {
    "accuracy": 0.78,
    "accuracy_breakdown": {
        "hand_shape": 0.82, "position": 0.71,
        "movement": 0.66, "palm_orientation": 0.93,
    },
}
win.session.last_stars = 2
win.session.completed = True
win.finish_level(win.session.last_scores, 2)
QApplication.processEvents()
win.grab().save(str(OUT / "part3_results.png"))

# profile
win.show_screen("profile")
QApplication.processEvents()
win.grab().save(str(OUT / "part3_profile.png"))

print(f"Part 3 screenshots written to {OUT}")
win.stop() if hasattr(win, "stop") else None