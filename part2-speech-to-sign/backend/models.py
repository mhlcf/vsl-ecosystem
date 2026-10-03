"""Pydantic request/response models for the Part 2 API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .config import SUPPORTED_LANGUAGES


class ProcessSpeechRequest(BaseModel):
    audio_file: str = Field(description="base64-encoded WAV/MP3 audio")
    language: str = Field(default="vi", description="language code for STT")
    duration_ms: int = Field(default=3000, ge=100, le=60_000)


class VSLWord(BaseModel):
    word: str
    text: str


class VSLStructure(BaseModel):
    topic: str
    comment: str
    words: list[str]


class AnimationParameters(BaseModel):
    template_key: str | None = None
    frames: list[list[float]] | None = None
    fps: int = 30
    frame_count: int = 0


class AnimationStep(BaseModel):
    word: str
    animation_id: str | None = None
    duration_ms: int
    parameters: AnimationParameters


class ProcessSpeechResponse(BaseModel):
    status: str = "success"
    original_text: str
    vsl_structure: VSLStructure
    animation_sequence: list[AnimationStep]
    controls: dict
    confidence: float = 0.0
    language: str = "vi"


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str


class STranslateRequest(BaseModel):
    text: str
    language: str = "vi"