"""Shared fixtures for Part 3 backend tests."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# isolate the SQLite database for the test session (set before config import)
os.environ.setdefault("VSL_DB_PATH", str(Path(tempfile.mkdtemp()) / "test_vsl.db"))

# make `backend` importable when pytest collects this directory directly
_PART3 = Path(__file__).resolve().parents[1]      # part3-gamification
for p in (str(_PART3), str(_PART3 / "..")):
    if p not in sys.path:
        sys.path.insert(0, p)

from vslshared.config import TRAIN_DIR


@pytest.fixture(scope="session")
def real_sequence() -> np.ndarray:
    """A real ``(T, 201)`` sequence for the first dataset sign."""
    from vslshared.template_reference import iter_sequences

    sign = next(iter(sorted(p for p in os.listdir(TRAIN_DIR) if (TRAIN_DIR / p).is_dir())))
    seq = next(iter_sequences(TRAIN_DIR, sign, limit=1), None)
    assert seq is not None, "no usable sequences in dataset"
    return seq, sign


@pytest.fixture()
def api_client():
    from fastapi.testclient import TestClient

    from p3backend.app import app

    with TestClient(app) as client:
        yield client