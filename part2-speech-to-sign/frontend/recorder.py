"""Microphone recorder (sounddevice) producing float32 mono 16 kHz."""

from __future__ import annotations

import threading

import numpy as np


class MicrophoneRecorder:
    """Records until ``stop()``; returns float32 mono samples at 16 kHz."""

    SAMPLE_RATE = 16000

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        import sounddevice as sd

        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._on_audio,
        )
        self._stream.start()

    def _on_audio(self, indata, frames, time, status) -> None:
        with self._lock:
            self._chunks.append(indata[:, 0].copy())

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._chunks)