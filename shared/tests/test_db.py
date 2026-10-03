"""Tests for the shared SQLite storage."""

from __future__ import annotations

import pytest

from vslshared.db import Database


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "test.db")


def test_create_user_and_get(db):
    uid = db.create_user("lan", "lan@example.com")
    user = db.get_user(uid)
    assert user["username"] == "lan"
    assert db.get_user_by_name("lan")["id"] == uid


def test_create_user_idempotent(db):
    u1 = db.create_user("bich")
    u2 = db.create_user("bich")
    assert u1 == u2
    assert len(db.query_all("SELECT * FROM users")) == 1


def test_vocab_upsert(db):
    db.upsert_vocab("xin chào", difficulty=1, viet="Xin chào", animation_id="anim_00001")
    db.upsert_vocab("xin chào", difficulty=2)  # update
    rows = db.vocab_list()
    assert len(rows) == 1
    assert rows[0]["difficulty_level"] == 2
    assert len(db.vocab_by_difficulty(2)) == 1


def test_session_lifecycle(db):
    uid = db.create_user("hoa")
    db.upsert_vocab("ăn", difficulty=1)
    sid = db.start_session(uid, level_id=1)
    db.log_attempt(sid, sign_id=None, scores={"hand_shape": 0.9, "movement": 0.8}, correct=True)
    db.end_session(sid)
    session = db.query_one("SELECT * FROM game_sessions WHERE id = ?", (sid,))
    assert session["user_id"] == uid
    logs = db.query_all("SELECT * FROM attempt_logs WHERE session_id = ?", (sid,))
    assert len(logs) == 1
    assert "hand_shape" in logs[0]["accuracy_scores"]


def test_progress_keeps_best(db):
    uid = db.create_user("mai")
    db.record_progress(uid, level_id=1, stars=2, accuracy=0.75, attempts=3)
    db.record_progress(uid, level_id=1, stars=3, accuracy=0.85, attempts=1)
    db.record_progress(uid, level_id=1, stars=1, accuracy=0.40, attempts=1)
    prog = db.get_progress(uid)
    assert prog[0]["stars_earned"] == 3
    assert prog[0]["best_accuracy"] == pytest.approx(0.85)
    assert prog[0]["attempts"] == 5