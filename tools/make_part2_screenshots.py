"""Part 2 UI screenshots, rendered offscreen (matplotlib keypoint avatar).

Usage:  venv/bin/python tools/make_part2_screenshots.py [outdir]
Outputs PNGs into docs/screenshots/ (default).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = (Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/screenshots")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# keep part2's own packages resolvable (no pytest conftest here)
sys.path.insert(0, str(ROOT / "part2-speech-to-sign"))

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)

from frontend.main import SpeechToSignApp  # noqa: E402
from frontend.pipeline import process_text  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)

win = SpeechToSignApp()
win.show()
QApplication.processEvents()
win.text_input.setText("Tôi không thích cà phê")

result = process_text("Tôi không thích cà phê")
win._on_result(result)

# pump a few ticks so the matplotlib canvas draws avatar frames
for _ in range(40):
    QApplication.processEvents()
    app.processEvents()

win.grab().save(str(OUT / "part2_speech.png"))

# dedicated avatar frame (rendered via the canvas itself -> always valid)
try:
    canvas = win.viewport._canvas
    if canvas is not None:
        canvas.figure.savefig(
            str(OUT / "part2_avatar.png"), dpi=120, facecolor="#101216")
    else:
        print("avatar canvas not built; skipping avatar.png")
except Exception as exc:  # noqa: BLE001
    print(f"avatar export skipped: {exc}")

print(f"Part 2 screenshots written to {OUT}")