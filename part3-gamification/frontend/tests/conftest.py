"""Headless fixtures for Part 3 frontend tests."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# offscreen Qt + isolated DB must be set before any PyQt/backend import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VSL_DB_PATH", str(Path(tempfile.mkdtemp()) / "frontend_test.db"))

_PART3 = Path(__file__).resolve().parents[2]      # part3-gamification
for p in (str(_PART3), str(Path(_PART3).parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app