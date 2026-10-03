"""In-process speech-to-sign pipeline the frontend calls (like the API).

Imports the backend modules lazily so the module works in two contexts:

* standalone app  — ``backend`` package (part2-speech-to-sign/backend on path)
* shared test run — ``p2backend`` alias registered by the test conftest
"""

from __future__ import annotations

import sys

_settings = None
_recognizer = None


def _load(module: str):
    """Import a part2 backend module under whichever name is available."""
    from importlib import import_module

    if "p2backend" in sys.modules:
        return import_module(f"p2backend.{module}")
    return import_module(f"backend.{module}")


def get_settings():
    global _settings
    if _settings is None:
        _settings = _load("config").get_settings()
    return _settings


def stt_available() -> bool:
    return _recognizer is not None and _recognizer.loaded


def ensure_recognizer():
    global _recognizer
    if _recognizer is None:
        stt = _load("speech_recognition")
        _recognizer = stt.SpeechRecognizer(get_settings())
    return _recognizer.load()


def process_audio(samples, sample_rate: int) -> dict:
    """Full pipeline: audio -> text -> VSL structure -> animation steps."""
    result = ensure_recognizer().transcribe_raw(samples, sample_rate)
    return finish_pipeline(result.text, result.confidence, result.language)


def process_text(text: str) -> dict:
    """Grammar + animation only (used when STT is unavailable)."""
    return finish_pipeline(text, 0.0, "vi")


def finish_pipeline(original_text: str, confidence: float, language: str) -> dict:
    grammar = _load("grammar_translator")
    avatar = _load("avatar_controller")
    structure = grammar.translate_to_vsl(original_text)
    structure = grammar.refine_with_llm(structure, original_text)
    steps = avatar.build_animation_sequence(structure["words"])
    missing: list[str] = []
    frames = []
    for step in steps:
        f = step["parameters"].get("frames")
        if f is not None and len(f) > 0:
            frames.append(f)
        else:
            missing.append(step["word"])
            frames.append(avatar.neutral_pose())
    # keep steps/frames consistent for consumers that zip them
    _fps = get_settings().avatar_fps
    for step, clip in zip(steps, frames):
        if step["parameters"]["frame_count"] == 0:
            step["parameters"]["frames"] = clip
            step["parameters"]["frame_count"] = int(len(clip))
            step["parameters"]["fps"] = _fps
    return {
        "original_text": original_text,
        "vsl_structure": structure,
        "animation_sequence": steps,
        "clips": frames,
        "missing_animation": missing,
        "confidence": round(confidence, 3),
        "language": language,
        "controls": {
            "fps": _fps,
            "total_clips": len(steps),
        },
    }