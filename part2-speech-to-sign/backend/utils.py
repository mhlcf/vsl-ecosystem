"""Helper utilities for Part 2: audio decode, temp files, normalizers."""

from __future__ import annotations

import base64
import binascii
import io
import uuid
import wave
from pathlib import Path

import numpy as np

from .config import get_settings, DEFAULT_SAMPLE_RATE


def decode_audio_base64(payload: str) -> tuple[bytes, str | None]:
    """Return ``(raw_bytes, kind)`` where kind is None when valid."""
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return b"", "invalid_base64"
    if not data:
        return b"", "empty_audio"
    if len(data) > get_settings().audio_max_bytes():
        return b"", "audio_too_large"
    return data, None


def infer_wav_props(raw: bytes) -> tuple[int, int] | None:
    """Best-effort WAV header read -> (sample_rate, channels); None if not WAV."""
    try:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            return wf.getframerate(), wf.getnchannels()
    except (wave.Error, EOFError):
        return None


def save_audio_temp(raw: bytes, suffix: str = ".wav") -> Path:
    """Persist raw audio bytes into the settings temp dir."""
    name = uuid.uuid4().hex + suffix
    path = get_settings().temp_dir / name
    path.write_bytes(raw)
    return path


def read_wav_samples(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a 16-bit mono WAV -> (float32 [-1,1], sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, rate


def resample_to_16k(samples: np.ndarray, source_rate: int) -> np.ndarray:
    """Linear-interpolate audio to the Whisper sample rate (16 kHz)."""
    if source_rate == DEFAULT_SAMPLE_RATE:
        return samples.astype(np.float32)
    n_out = int(round(len(samples) * DEFAULT_SAMPLE_RATE / source_rate))
    x_old = np.linspace(0.0, 1.0, len(samples))
    x_new = np.linspace(0.0, 1.0, n_out)
    return np.interp(x_new, x_old, samples).astype(np.float32)