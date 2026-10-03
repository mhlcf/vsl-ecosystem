"""Speech-to-Text module (Whisper, Vietnamese).

Wraps ``faster-whisper`` so the model is loaded lazily on first use. The
recognizer exposes a small, testable surface:

    recognizer.load()
    result = recognizer.transcribe_wav(path)
    result = recognizer.transcribe_raw(samples, sample_rate)

If the model download/cache is unavailable, ``load`` raises
``STTUnavailableError``; callers may fall back to the rule-based grammar
path (see ``grammar_translator``) so the rest of the pipeline still works.
"""

from __future__ import annotations

import logging
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import get_settings

logger = logging.getLogger("vsl.part2.stt")


class STTUnavailableError(RuntimeError):
    """Raised when the Whisper model cannot be loaded (offline / no cache)."""


@dataclass
class STTResult:
    text: str
    language: str = "vi"
    confidence: float = 0.0
    duration_ms: int = 0
    segments: list[dict] = field(default_factory=list)


class SpeechRecognizer:
    """Lazy-loaded faster-whisper wrapper for Vietnamese STT."""

    def __init__(self, settings: Part2Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None  # type: ignore[assignment]

    # -- lifecycle -----------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> "SpeechRecognizer":
        if self._model is not None:
            return self
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - env dependent
            raise STTUnavailableError("faster-whisper is not installed") from exc
        cfg = self._settings
        logger.info("loading whisper '%s' (device=%s)", cfg.whisper_model_size,
                    cfg.whisper_device)
        try:
            self._model = WhisperModel(
                cfg.whisper_model_size,
                device=cfg.whisper_device,
                compute_type=cfg.whisper_compute_type,
            )
        except Exception as exc:  # network / CUDA issues (e.g. libcusparse.so.12)
            logger.warning("failed to load whisper on %s: %s", cfg.whisper_device, exc)
            try:
                self._model = WhisperModel(cfg.whisper_model_size, device="cpu",
                                           compute_type="int8")
            except Exception as cpu_exc:
                raise STTUnavailableError(
                    "Không tải được mô hình Whisper "
                    f"({cfg.whisper_device} lỗi, CPU cũng thất bại: {cpu_exc}). "
                    "Hãy kiểm tra driver CUDA/LD_LIBRARY_PATH hoặc dùng nhập văn bản."
                ) from cpu_exc
        return self

    def close(self) -> None:
        self._model = None

    # -- transcription -------------------------------------------------------
    def transcribe_wav(self, wav_path: str | Path) -> STTResult:
        """Transcribe a 16-bit mono WAV file."""
        wav_path = Path(wav_path)
        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        duration_ms = int(1000 * len(samples) / rate)
        return self.transcribe_raw(samples, rate, duration_ms=duration_ms)

    def transcribe_raw(self, samples: np.ndarray, sample_rate: int,
                       duration_ms: int = 0) -> STTResult:
        if self._model is None:
            raise STTUnavailableError("model not loaded; call load() first")
        cfg = self._settings
        language = cfg.stt_language
        segments, info = self._model.transcribe(
            samples,
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        text_parts: list[str] = []
        seg_list: list[dict] = []
        for seg in segments:
            text_parts.append(seg.text.strip())
            seg_list.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            })
        text = " ".join(text_parts).strip()
        confidence = float(getattr(info, "language_probability", 0.0) or 0.0)
        detected_lang = str(getattr(info, "language", language) or language)
        return STTResult(
            text=text,
            language=detected_lang,
            confidence=round(confidence, 3),
            duration_ms=duration_ms or int((seg_list[-1]["end"] * 1000)
                                           if seg_list else 0),
            segments=seg_list,
        )


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Persist float32 samples as a 16-bit mono WAV file."""
    import struct

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(
        np.int16, copy=False).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return path