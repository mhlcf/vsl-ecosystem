"""Part 2 FastAPI server: Speech-to-Sign via Whisper + VSL grammar + avatar.

Endpoints
    POST /api/v1/process-speech   full pipeline: audio → text → VSL → animation
    POST /api/v1/translate-text    text-only pipeline (no STT)
    GET  /api/v1/signs/{q}         search signs in the vocabulary
    GET  /health                   liveness + backend state
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import avatar_controller
from .config import get_settings
from .grammar_translator import refine_with_llm, translate_to_vsl
from .models import (
    AnimationStep,
    AnimationParameters,
    ErrorResponse,
    ProcessSpeechRequest,
    ProcessSpeechResponse,
    STranslateRequest,
    VSLStructure,
)
from .speech_recognition import STTUnavailableError, SpeechRecognizer
from .utils import (
    decode_audio_base64,
    infer_wav_props,
    read_wav_samples,
    resample_to_16k,
    save_audio_temp,
)

logger = logging.getLogger("vsl.part2")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

_settings = get_settings()
_recognizer = SpeechRecognizer(_settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("part2 backend starting (whisper deferred: first process-speech call)")
    yield
    _recognizer.close()


app = FastAPI(
    title="VSL Speech-to-Sign API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# core pipeline
# ---------------------------------------------------------------------------
def _run_pipeline(original_text: str, confidence: float,
                  language: str) -> ProcessSpeechResponse:
    structure = translate_to_vsl(original_text)
    structure = refine_with_llm(structure, original_text)
    steps = avatar_controller.build_animation_sequence(structure["words"])
    return ProcessSpeechResponse(
        status="success",
        original_text=original_text,
        vsl_structure=VSLStructure(**structure),
        animation_sequence=[_step_to_model(s) for s in steps],
        controls=avatar_controller.controls(steps),
        confidence=round(confidence, 3),
        language=language,
    )


def _step_to_model(step: dict) -> AnimationStep:
    params = dict(step["parameters"])
    frames = params.get("frames")
    if isinstance(frames, np.ndarray):
        params["frames"] = frames.round(4).tolist() if frames.shape[0] <= 30 else None
    else:
        params["frames"] = None
    return AnimationStep(
        word=step["word"],
        animation_id=step["animation_id"],
        duration_ms=step["duration_ms"],
        parameters=AnimationParameters(**params),
    )


@app.get("/health", include_in_schema=False)
def health() -> dict:
    return {
        "status": "ok",
        "stt_model": _settings.whisper_model_size,
        "stt_loaded": _recognizer.loaded,
        "llm_enabled": _settings.llm_enabled,
    }


@app.post("/api/v1/process-speech", response_model=ProcessSpeechResponse,
          responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def process_speech(req: ProcessSpeechRequest) -> ProcessSpeechResponse:
    raw, kind = decode_audio_base64(req.audio_file)
    if kind:
        raise HTTPException(status_code=400, detail=f"bad audio payload: {kind}")

    props = infer_wav_props(raw)
    if props is None and not raw.startswith(b"ID3") and not raw.startswith(b"\xff\xfb"):
        raise HTTPException(status_code=400,
                            detail="unsupported or invalid audio format (WAV expected)")

    tmp = save_audio_temp(raw)
    try:
        samples, rate = read_wav_samples(tmp)
    except Exception:
        raise HTTPException(status_code=400, detail="could not parse WAV audio")
    finally:
        tmp.unlink(missing_ok=True)

    samples = resample_to_16k(samples, rate)
    if req.language not in ("vi", "en"):
        raise HTTPException(status_code=400, detail="unsupported language")

    try:
        _recognizer.load()
        result = _recognizer.transcribe_raw(samples, 16000, duration_ms=req.duration_ms)
    except STTUnavailableError:
        raise HTTPException(status_code=503,
                            detail="Whisper model unavailable (offline?) — "
                                   "use /api/v1/translate-text instead")

    return _run_pipeline(result.text, result.confidence, result.language)


@app.post("/api/v1/translate-text", response_model=ProcessSpeechResponse)
def translate_text(req: STranslateRequest) -> ProcessSpeechResponse:
    """Grammar + animation only (no audio). Useful for offline/demo/testing."""
    return _run_pipeline(req.text, confidence=0.0, language=req.language)


@app.get("/api/v1/signs/{q}")
def search_signs(q: str) -> dict:
    """Quick vocabulary search used by the frontend autocomplete."""
    q = q.strip().lower()
    if not q:
        return {"query": q, "results": []}
    results = []
    for entry in avatar_controller._vocab():
        sn = entry["sign_name"].lower()
        if q in sn or avatar_controller.normalize_sign_key(entry["sign_name"]).startswith(
                avatar_controller.normalize_sign_key(q)):
            results.append({
                "sign_name": entry["sign_name"],
                "animation_id": entry["animation_id"],
                "difficulty": entry["difficulty"],
            })
        if len(results) >= 20:
            break
    return {"query": q, "results": results}

def main() -> None:
    """Run the Part 2 API server (default port 8001)."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Part 2 Speech-to-Sign API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(8001))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
