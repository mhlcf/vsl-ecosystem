"""Part 2 configuration: Whisper, grammar rules, avatar defaults.

All tunables are environment-overridable, mirroring the Part 3 pattern so
the app runs offline with ``ollama`` or a pure rule-based fallback.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vslshared.config import DATA_DIR

DEFAULT_SAMPLE_RATE = 16000
SUPPORTED_LANGUAGES = ("vi", "en")

_MAX_AUDIO_BYTES = 25 * 1024 * 1024   # 25 MB safety cap for base64 payloads


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


@dataclass(frozen=True)
class Part2Settings:
    # speech recognition
    whisper_model_size: str = os.environ.get("WHISPER_MODEL_SIZE", "small")
    whisper_device: str = os.environ.get("WHISPER_DEVICE", "cuda")
    whisper_compute_type: str = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16")
    stt_language: str = os.environ.get("VSL_STT_LANGUAGE", "vi")
    stt_min_silence_ms: int = _env_int("VSL_STT_MIN_SILENCE_MS", 500)
    stt_diarize: bool = False

    # local LLM (optional) for grammar refinement; falls back to rule-based
    llm_base_url: str = os.environ.get("VSL_LLM_BASE_URL", "http://localhost:11434")
    llm_model: str = os.environ.get("VSL_LLM_MODEL", "llama3.2")
    llm_enabled: bool = _env_bool("VSL_LLM_ENABLED", False)

    # avatar defaults
    avatar_fps: int = _env_int("VSL_AVATAR_FPS", 30)
    default_word_duration_ms: int = _env_int("VSL_WORD_DURATION_MS", 2000)

    # temporary folder for decoded audio
    temp_dir: Path = Path(os.environ.get("VSL_TMP_DIR", tempfile.gettempdir()))

    def audio_max_bytes(self) -> int:
        return _MAX_AUDIO_BYTES

    def output_dir(self) -> Path:
        out = DATA_DIR / "part2"
        out.mkdir(parents=True, exist_ok=True)
        return out


def get_settings() -> Part2Settings:
    return Part2Settings()