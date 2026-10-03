"""Local SQLite storage shared by part2 and part3.

Implements the schema from the project prompt (section 4.3.5):
users, user_progress, sign_vocabulary, game_sessions, attempt_logs.
Small typed wrapper around sqlite3 with safe parameterised queries.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import DB_PATH, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    level_id INTEGER NOT NULL,
    stars_earned INTEGER DEFAULT 0,
    best_accuracy REAL DEFAULT 0,
    attempts INTEGER DEFAULT 0,
    completed_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE (user_id, level_id)
);

CREATE TABLE IF NOT EXISTS sign_vocabulary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sign_name TEXT NOT NULL,
    viet_translation TEXT,
    difficulty_level INTEGER NOT NULL DEFAULT 1,
    animation_id TEXT
);

CREATE TABLE IF NOT EXISTS game_sessions (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    level_id INTEGER NOT NULL,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS attempt_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    sign_id INTEGER,
    accuracy_scores TEXT,
    correct INTEGER DEFAULT 0,
    timestamp TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES game_sessions(id)
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Thin thread-safe wrapper around a local SQLite file."""

    def __init__(self, path: Path | str | None = None) -> None:
        ensure_dirs()
        self.path: Path = Path(path) if path else DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock: Any = __import__("threading").Lock()
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self) -> None:
        """Create all tables if they do not exist yet."""
        with self._lock:
            with self._connect() as conn:
                conn.executescript(SCHEMA)

    # -- generic helpers -----------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Run a write query, returning lastrowid."""
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(sql, params)
                conn.commit()
                return int(cur.lastrowid or 0)

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(sql, params).fetchone()
                return dict(row) if row else None

    def query_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]

    # -- users ---------------------------------------------------------------
    def create_user(self, username: str, email: str | None = None) -> int:
        ts = now_iso()
        if not self._user_exists(username):
            return self.execute(
                "INSERT INTO users (username, email, created_at, updated_at)"
                " VALUES (?, ?, ?, ?)",
                (username, email, ts, ts),
            )
        return self.get_user_by_name(username)["id"]

    def _user_exists(self, username: str) -> bool:
        row = self.query_one("SELECT 1 FROM users WHERE username = ?", (username,))
        return row is not None

    def get_user_by_name(self, username: str) -> dict | None:
        return self.query_one("SELECT * FROM users WHERE username = ?", (username,))

    def get_user(self, user_id: int) -> dict | None:
        return self.query_one("SELECT * FROM users WHERE id = ?", (user_id,))

    # -- vocabulary ----------------------------------------------------------
    def upsert_vocab(self, sign_name: str, difficulty: int, viet: str | None = None,
                     animation_id: str | None = None) -> None:
        cur = self.query_one("SELECT id FROM sign_vocabulary WHERE sign_name = ?",
                             (sign_name,))
        if cur:
            self.execute(
                "UPDATE sign_vocabulary SET difficulty_level = ?, "
                "viet_translation = ?, animation_id = ? WHERE id = ?",
                (difficulty, viet, animation_id, cur["id"]),
            )
        else:
            self.execute(
                "INSERT INTO sign_vocabulary (sign_name, viet_translation,"
                " difficulty_level, animation_id) VALUES (?, ?, ?, ?)",
                (sign_name, viet, difficulty, animation_id),
            )

    def vocab_list(self) -> list[dict]:
        return self.query_all("SELECT * FROM sign_vocabulary ORDER BY id")

    def vocab_by_difficulty(self, difficulty: int) -> list[dict]:
        return self.query_all(
            "SELECT * FROM sign_vocabulary WHERE difficulty_level = ? ORDER BY id",
            (difficulty,),
        )

    # -- sessions ------------------------------------------------------------
    def start_session(self, user_id: int, level_id: int) -> str:
        session_id = uuid.uuid4().hex[:12]
        self.execute(
            "INSERT INTO game_sessions (id, user_id, level_id, started_at)"
            " VALUES (?, ?, ?, ?)",
            (session_id, user_id, level_id, now_iso()),
        )
        return session_id

    def end_session(self, session_id: str) -> None:
        self.execute(
            "UPDATE game_sessions SET ended_at = ? WHERE id = ?",
            (now_iso(), session_id),
        )

    # -- progress ------------------------------------------------------------
    def record_progress(self, user_id: int, level_id: int, stars: int,
                        accuracy: float, attempts: int) -> None:
        existing = self.query_one(
            "SELECT id, stars_earned, best_accuracy FROM user_progress"
            " WHERE user_id = ? AND level_id = ?",
            (user_id, level_id),
        )
        best_accuracy = max(
            accuracy, float(existing["best_accuracy"] or 0) if existing else 0
        )
        best_stars = max(stars, int(existing["stars_earned"] or 0) if existing else 0)
        if existing:
            self.execute(
                "UPDATE user_progress SET stars_earned = ?, best_accuracy = ?,"
                " attempts = attempts + ?, completed_at = ? WHERE id = ?",
                (best_stars, best_accuracy, attempts, now_iso(), existing["id"]),
            )
        else:
            self.execute(
                "INSERT INTO user_progress (user_id, level_id, stars_earned,"
                " best_accuracy, attempts, completed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, level_id, stars, accuracy, attempts, now_iso()),
            )

    def get_progress(self, user_id: int) -> list[dict]:
        return self.query_all(
            "SELECT * FROM user_progress WHERE user_id = ? ORDER BY level_id",
            (user_id,),
        )

    def log_attempt(self, session_id: str, sign_id: int | None,
                    scores: dict, correct: bool) -> None:
        import json

        self.execute(
            "INSERT INTO attempt_logs (session_id, sign_id, accuracy_scores,"
            " correct, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, sign_id, json.dumps(scores), int(correct), now_iso()),
        )