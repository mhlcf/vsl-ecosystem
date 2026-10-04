import time

import pytest

from sentence_builder import (
    SentenceComposer,
    WordBuffer,
    clean_label,
    rule_sentence,
)


def wait_idle(composer: SentenceComposer, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while composer._busy and time.time() < deadline:
        time.sleep(0.02)
    assert not composer._busy, "compose thread did not finish"


def test_clean_label_strips_gloss():
    assert clean_label("Anh (nước Anh)") == "Anh"
    assert clean_label("1 000 000 (một triệu)") == "1 000 000"
    assert clean_label("cà phê") == "cà phê"


def test_rule_sentence():
    assert rule_sentence(["Anh", "thích", "cà phê"]) == "Anh thích cà phê."
    assert rule_sentence([]) == ""


def test_word_buffer_commits_stable_word_and_dedupes():
    buf = WordBuffer(threshold=0.0, stable_frames=3)
    assert buf.feed("cà phê", 0.9) is None
    assert buf.feed("cà phê", 0.9) is None
    assert buf.feed("cà phê", 0.9) == "cà phê"
    for _ in range(6):
        buf.feed("cà phê", 0.9)
    assert buf.words == ["cà phê"]
    assert buf.feed("thích", 0.9) is None
    assert buf.feed("thích", 0.9) is None
    assert buf.feed("thích", 0.9) == "thích"
    assert buf.words == ["cà phê", "thích"]
    buf.undo()
    assert buf.words == ["cà phê"]
    buf.clear()
    assert buf.words == []


def test_word_buffer_resets_on_no_hand_or_low_confidence():
    buf = WordBuffer(threshold=0.5, stable_frames=3)
    assert buf.feed("Anh", 0.9) is None
    assert buf.feed("Anh", 0.9) is None
    buf.feed(None, 0.9)                      # hand lost -> stability resets
    assert buf.feed("Anh", 0.9) is None
    assert buf.feed("Anh", 0.9) is None
    assert buf.feed("Anh", 0.9) == "Anh"
    buf.clear()
    assert buf.feed("Anh", 0.4) is None      # below threshold -> ignored
    assert buf.words == []


def test_compose_fallback_without_key(monkeypatch):
    monkeypatch.setenv("VSL_LLM_ENABLED", "1")
    monkeypatch.delenv("VSL_LLM_API_KEY", raising=False)
    composer = SentenceComposer()
    assert composer.ready is False
    assert composer.compose(["Anh", "thích", "cà phê"]) is True
    wait_idle(composer)
    assert composer.status == "fallback"
    assert composer.sentence == "Anh thích cà phê."


def test_compose_uses_api_when_call_succeeds(monkeypatch):
    monkeypatch.setenv("VSL_LLM_ENABLED", "1")
    monkeypatch.setenv("VSL_LLM_API_KEY", "test-key")
    composer = SentenceComposer()
    assert composer.ready is True
    monkeypatch.setattr(
        SentenceComposer, "_call_api",
        lambda self, words: "Anh thích cà phê.",
    )
    assert composer.compose(["Anh", "thích", "cà phê"]) is True
    wait_idle(composer)
    assert composer.status == "api"
    assert composer.sentence == "Anh thích cà phê."
    assert composer.compose(["Anh", "thích", "cà phê"]) is False  # duplicate skipped
    assert composer.compose(["Anh", "thích", "cà phê"], force=True) is True


def test_compose_api_error_falls_back(monkeypatch):
    monkeypatch.setenv("VSL_LLM_ENABLED", "1")
    monkeypatch.setenv("VSL_LLM_API_KEY", "test-key")
    composer = SentenceComposer()
    monkeypatch.setattr(SentenceComposer, "_call_api", lambda self, words: None)
    assert composer.compose(["tôi", "đói"]) is True
    wait_idle(composer)
    assert composer.status == "fallback"
    assert composer.sentence == "Tôi đói."
