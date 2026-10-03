"""Database entry-point for Part 3; re-exports the shared SQLite layer."""

from __future__ import annotations

from vslshared.db import SCHEMA, Database

__all__ = ["Database", "SCHEMA"]


def get_db() -> Database:
    return Database()