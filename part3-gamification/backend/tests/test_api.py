"""Tests for user management and the game API flow."""

from __future__ import annotations

import numpy as np
import pytest

from p3backend.user_manager import UserManager
from vslshared.template_reference import iter_sequences
from vslshared.config import TRAIN_DIR


def test_get_or_create_user_roundtrip():
    um = UserManager()
    u1 = um.get_or_create_user("api_test_user")
    u2 = um.get_or_create_user("api_test_user")
    assert u1["id"] == u2["id"]
    stats = um.stats(u1["id"])
    assert stats["total_levels_completed"] >= 0
    assert stats["total_stars"] >= 0


def test_completed_levels_and_record():
    um = UserManager()
    user = um.get_or_create_user("progress_user")
    assert um.completed_levels(user["id"]) == []
    um.record_completion(user["id"], 1, stars=2, accuracy=0.85, attempts=4)
    assert um.completed_levels(user["id"]) == [1]


def test_start_submit_complete_flow(api_client):
    # create user via the shared db (no auth endpoint in MVP)
    um = UserManager()
    user = um.get_or_create_user("flow_user")

    start = api_client.post("/api/v1/game/start-session",
                            json={"user_id": user["id"], "level_id": 1})
    assert start.status_code == 200
    body = start.json()
    assert body["word_name"]

    session_id = body["session_id"]
    sign = body["word_name"]
    seq = next(iter_sequences(TRAIN_DIR, sign, limit=1), None)
    if seq is None:  # sign not in dataset slice of tests -> skip submit
        pytest.skip(f"no dataset sequence for {sign!r}")

    submit = api_client.post("/api/v1/game/submit-pose", json={
        "session_id": session_id,
        "frame_data": {"sequence": seq.tolist()},
    })
    assert submit.status_code == 200
    resp = submit.json()
    assert resp["status"] == "processing"
    for key in ("hand_shape", "position", "movement", "palm_orientation"):
        assert 0 <= resp["accuracy_breakdown"][key] <= 1
    assert isinstance(resp["feedback"], str)

    complete = api_client.post("/api/v1/game/complete-level", json={
        "session_id": session_id,
        "attempts": 3,
        "average_accuracy": resp["confidence"],
    })
    assert complete.status_code == 200
    done = complete.json()
    assert done["stars_earned"] in (0, 1, 2, 3)
    assert done["level_completed"] is (done["stars_earned"] > 0)


def test_submit_bad_session(api_client):
    r = api_client.post("/api/v1/game/submit-pose",
                        json={"session_id": "nope", "frame_data": {"sequence": [[0] * 201] * 60}})
    assert r.status_code == 404


def test_bad_frame_data(api_client):
    um = UserManager()
    user = um.get_or_create_user("badframe_user")
    start = api_client.post("/api/v1/game/start-session",
                            json={"user_id": user["id"], "level_id": 1}).json()
    r = api_client.post("/api/v1/game/submit-pose", json={
        "session_id": start["session_id"],
        "frame_data": "!!!not-base64!!!",
    })
    assert r.status_code == 400


def test_stats_endpoint(api_client):
    um = UserManager()
    user = um.get_or_create_user("stats_user")
    r = api_client.get(f"/api/v1/users/{user['id']}/stats")
    assert r.status_code == 200
    assert r.json()["username"] == "stats_user"