"""Animation sequencing & control for the 3D avatar.

Resolves a VSL word list into a playable sequence of clips. Each clip is
backed by real keypoint motion from the Part 1 dataset (the NPZ files /
reference templates), so no external 3D assets are required:

    word → animation_id (vocabulary) → template key ("reference://name") → frames
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np

from vslshared.template_reference import load_reference_set, resample_seq
from vslshared.vocab import load_vocabulary

from .config import get_settings

TEMPLATE_TARGET_N = 30

_vocab_cache: list[dict] | None = None


def _vocab() -> list[dict]:
    global _vocab_cache
    if _vocab_cache is None:
        _vocab_cache = load_vocabulary()
    return _vocab_cache


def _strip_diacritics(text: str) -> str:
    ascii_ = unicodedata.normalize("NFD", text)
    return "".join(c for c in ascii_ if unicodedata.category(c) != "Mn")


def normalize_sign_key(text: str) -> str:
    return _strip_diacritics(text.lower().strip()).replace(" ", "_")


_reference_cache: dict | None = None


def load_animation_library() -> dict[str, np.ndarray]:
    """``{sign_name: (T, 201) reference template}`` for the avatar."""
    global _reference_cache
    if _reference_cache is None:
        _reference_cache = load_reference_set()
    return _reference_cache


def resolve_sign(word: str) -> dict | None:
    """Best-effort word → vocabulary sign entry.

    Matching order: exact → diacritic/key-normalized → substring, preferring
    entries that actually have animation data in the reference library.
    """
    word = word.strip().strip(".-,;:!?").lower()
    vocab = _vocab()
    if not word:
        return None
    ref = load_animation_library()
    norm = normalize_sign_key(word)

    def key(e):
        return e["sign_name"] in ref

    exact = [e for e in vocab if e["sign_name"].strip().lower() == word]
    if exact:
        return max(exact, key=key)
    normed = [e for e in vocab if normalize_sign_key(e["sign_name"]) == norm]
    if normed:
        return max(normed, key=key)
    sub = [e for e in vocab if norm in normalize_sign_key(e["sign_name"])]
    if sub:
        with_ref = [e for e in sub if e["sign_name"] in ref]
        return max(with_ref or sub, key=key)
    return None


def animation_frames(sign_name: str, n: int = TEMPLATE_TARGET_N) -> np.ndarray | None:
    """Return ``(n, 201)`` keypoint frames for a sign (reference template)."""
    lib = load_animation_library()
    tpl = lib.get(sign_name)
    if tpl is None:
        return None
    if tpl.shape[0] == n:
        return tpl
    return resample_seq(tpl, n)


def neutral_pose(n: int = TEMPLATE_TARGET_N) -> np.ndarray:
    """A static, dataset-average pose (the mean of all reference templates).

    Used as a visible "hold" placeholder for signs that have no motion data,
    so the avatar timeline stays continuous instead of silently stopping.
    """
    lib = load_animation_library()
    if lib:
        mean = np.mean(np.stack(list(lib.values())), axis=0)  # (T, 201)
        return np.broadcast_to(mean[0], (n, mean.shape[1])).copy()
    return np.zeros((n, 201), dtype=np.float32)


def build_animation_sequence(words: list[str],
                             duration_ms: int | None = None) -> list[dict]:
    """Turn VSL words into an ordered list of animation steps.

    Multi-word signs are matched greedily (longest run wins), so "chúc mừng"
    becomes a single clip backed by one reference template instead of two
    failed single-word lookups. Steps whose sign has no motion data still get
    an ``animation_id`` but ``parameters.frames`` is ``None`` (the frontend
    shows a neutral hold and tells the user which signs are missing).
    """
    cfg = get_settings()
    per_word = duration_ms or cfg.default_word_duration_ms
    steps: list[dict] = []
    i, n = 0, len(words)
    while i < n:
        entry = None
        span = 1
        for j in range(n, i, -1):          # longest run first
            cand = resolve_sign(" ".join(words[i:j]))
            if cand is not None and cand["sign_name"] in load_animation_library():
                entry, span = cand, j - i
                break
        if entry is None:                  # shortest single-word fallback
            entry = resolve_sign(words[i])
            span = 1
        sign_name = entry["sign_name"] if entry else None
        phrased = " ".join(words[i:i + span])
        frames = animation_frames(sign_name) if sign_name else None
        steps.append({
            "word": phrased,
            "animation_id": entry["animation_id"] if entry else None,
            "sign_name": sign_name,
            "duration_ms": int(per_word * span),
            "parameters": {
                "template_key": f"reference://{sign_name}" if sign_name else None,
                "frames": frames,
                "fps": cfg.avatar_fps,
                "frame_count": int(frames.shape[0]) if frames is not None else 0,
            },
        })
        i += span
    return steps


def controls(steps: list[dict]) -> dict:
    """Avatar state-machine metadata: play speed, repeat, transition."""
    cfg = get_settings()
    return {
        "mode": "sequence",
        "repeat": False,
        "fps": cfg.avatar_fps,
        "transition": "hold",
        "total_clips": len(steps),
        "estimated_duration_ms": sum(int(s["duration_ms"]) for s in steps),
    }