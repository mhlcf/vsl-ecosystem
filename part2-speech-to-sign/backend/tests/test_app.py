"""Integration tests for the Part 2 FastAPI endpoints."""

from __future__ import annotations

import base64

import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "stt_model" in body


def test_translate_text_endpoint(client):
    r = client.post("/api/v1/translate-text", json={
        "text": "Xin chào, bạn khỏe không?",
        "language": "vi",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["original_text"].startswith("Xin chào")
    structure = body["vsl_structure"]
    assert structure["topic"] == "bạn"
    assert structure["words"] == ["xin", "chào", "bạn", "khỏe", "không"]
    assert body["animation_sequence"]
    step = body["animation_sequence"][0]
    assert "animation_id" in step and "duration_ms" in step
    assert body["controls"]["total_clips"] == len(body["animation_sequence"])


def test_search_signs(client):
    r = client.get("/api/v1/signs/b")
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "b"
    assert isinstance(body["results"], list)


def test_process_speech_bad_base64(client):
    r = client.post("/api/v1/process-speech", json={
        "audio_file": "!!!not-base64!!!",
        "language": "vi",
        "duration_ms": 3000,
    })
    assert r.status_code == 400
    assert "detail" in r.json()


def test_process_speech_empty_audio(client):
    r = client.post("/api/v1/process-speech", json={
        "audio_file": base64.b64encode(b"").decode(),
        "language": "vi",
        "duration_ms": 3000,
    })
    assert r.status_code == 400


def test_process_speech_bad_wav(client):
    garbage = base64.b64encode(b"this is not audio at all").decode()
    r = client.post("/api/v1/process-speech", json={
        "audio_file": garbage,
        "language": "vi",
        "duration_ms": 3000,
    })
    assert r.status_code in (400, 503)
    assert "detail" in r.json()