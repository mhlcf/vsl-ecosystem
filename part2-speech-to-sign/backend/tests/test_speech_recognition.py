"""Tests for the speech recognition module (no real model downloads)."""

from __future__ import annotations

import numpy as np
import pytest

from p2backend.speech_recognition import (
    STTUnavailableError,
    SpeechRecognizer,
    write_wav,
)


class FakeSegment:
    start = 0.0
    end = 1.0
    text = "Xin chào"


class FakeInfo:
    language = "vi"
    language_probability = 0.93


class FakeWhisper:
    """Stand-in for faster_whisper.WhisperModel."""

    def transcribe(self, samples, language="vi", beam_size=5, vad_filter=True):
        def _segments():
            yield FakeSegment()

        return _segments(), FakeInfo()


def test_load_deferred_then_fake_model():
    rec = SpeechRecognizer()
    assert not rec.loaded
    rec._model = FakeWhisper()
    assert rec.loaded


def test_transcribe_raw_with_fake_model():
    rec = SpeechRecognizer()
    rec._model = FakeWhisper()
    samples = np.zeros(16000, dtype=np.float32)
    result = rec.transcribe_raw(samples, 16000)
    assert result.language == "vi"
    assert result.confidence > 0.9
    assert "xìn" in result.text.lower() or result.text.lower() == "xin chào"


def test_transcribe_requires_model():
    rec = SpeechRecognizer()
    with pytest.raises(STTUnavailableError):
        rec.transcribe_raw(np.zeros(100), 16000)


def test_load_double_failure_raises_unavailable(monkeypatch):
    """CUDA + CPU both failing must surface STTUnavailableError, never raw."""
    import faster_whisper

    def _boom(*args, **kwargs):
        raise RuntimeError(
            "library 'libcusparse.so.12' is not found or cannot be loaded")

    monkeypatch.setattr(faster_whisper, "WhisperModel", _boom)
    rec = SpeechRecognizer()
    with pytest.raises(STTUnavailableError, match="Không tải được mô hình"):
        rec.load()


def test_write_and_read_wav_roundtrip(tmp_path):
    samples = np.sin(np.linspace(0, 10, 16000)).astype(np.float32) * 0.5
    path = write_wav(tmp_path / "tone.wav", samples, 16000)
    import wave

    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getnframes() == 16000