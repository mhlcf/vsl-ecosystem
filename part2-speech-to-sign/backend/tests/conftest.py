"""Test fixtures for Part 2 backend tests (aliases come from root conftest)."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from p2backend.app import app

    with TestClient(app) as c:
        yield c