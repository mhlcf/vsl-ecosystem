"""Reference template builder.

Builds per-sign "ideal" keypoint sequences from the existing Part 1 dataset
(the ``*.npz`` files under ``part1-sign-classifier/train``). These templates power:

  * the Part 3 scoring engine (4-criteria comparison against the user's pose)
  * the Part 2 avatar animation library (playback of recorded motion)
  * the template-matching classifier fallback (when no trained model is trusted)

A quality gate drops samples that have no detectable hands, so partially
captured clips do not poison the templates.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import (
    DATA_DIR,
    HAND_DIM,
    POSE_DIM,
    TEMPLATE_MIN_SAMPLES,
    TEMPLATE_TARGET_N,
    TRAIN_DIR,
    ensure_dirs,
)

CACHE_NAME = "reference_templates.npz"


# ---------------------------------------------------------------------------
# sequence helpers
# ---------------------------------------------------------------------------
def resample_seq(sequence: np.ndarray, n: int = TEMPLATE_TARGET_N) -> np.ndarray:
    """Resample ``(T, D)`` to ``(n, D)`` by linear interpolation in time."""
    seq = np.asarray(sequence, dtype=np.float32)
    t, d = seq.shape
    if t == n:
        return seq.copy()
    target = np.linspace(0, t - 1, n)
    src = np.arange(t)
    return np.stack(
        [np.interp(target, src, seq[:, j]) for j in range(d)], axis=1
    ).astype(np.float32)


def _wrist_block(hand: np.ndarray) -> np.ndarray:
    """Return the per-frame wrist (landmark 0 x,y,z) tiled per landmark.

    ``hand`` is ``(T, 63)``; column triple 0:3 is the wrist of each frame.
    Tiles the ``(T, 3)`` triplet 21 times so subtracting yields a ``(T, 63)``
    position-invariant hand that keeps shape, orientation and relative motion.
    """
    wrists = hand[:, 0:3]                              # (T, 3)
    return np.tile(wrists, (1, HAND_DIM // 3))         # (T, 63)


def center_sequence(sequence: np.ndarray) -> np.ndarray:
    """Subtract each wrist landmark so results are position invariant.

    Keeps hand shape, orientation and relative motion while removing where
    in the frame the hands happen to be.
    """
    seq = np.asarray(sequence, dtype=np.float32).copy()
    pose, left, right = (
        seq[:, :POSE_DIM],
        seq[:, POSE_DIM:POSE_DIM + HAND_DIM],
        seq[:, POSE_DIM + HAND_DIM:],
    )
    left_centered = left - _wrist_block(left)
    right_centered = right - _wrist_block(right)
    return np.concatenate([pose, left_centered, right_centered], axis=1).astype(np.float32)


def has_hands(sequence: np.ndarray) -> bool:
    """True when at least one hand block has meaningful signal."""
    if sequence.shape[1] < POSE_DIM + HAND_DIM * 2:
        return False
    hands = sequence[:, POSE_DIM:]
    return float(np.linalg.norm(hands)) > 1e-3


# ---------------------------------------------------------------------------
# dataset iteration
# ---------------------------------------------------------------------------
def iter_vocab_dirs(data_dir: Path) -> Iterator[tuple[str, Path]]:
    """Yield ``(sign_name, directory)`` for each label folder in a split dir."""
    if not data_dir.exists():
        return
    for entry in sorted(os.listdir(data_dir)):
        d = data_dir / entry
        if d.is_dir():
            yield entry, d


def iter_sequences(data_dir: Path, sign_name: str,
                   limit: int | None = None) -> Iterator[np.ndarray]:
    """Yield validated ``(T, 201)`` sequences for one label."""
    d = data_dir / sign_name
    if not d.is_dir():
        return
    n = 0
    for fname in sorted(os.listdir(d)):
        if not fname.endswith(".npz"):
            continue
        try:
            data = np.load(d / fname)
            seq = data["sequence"].astype(np.float32)
        except Exception:
            continue
        if seq.ndim != 2 or seq.shape[1] < POSE_DIM + HAND_DIM * 2:
            continue
        if not has_hands(seq):
            continue
        yield seq
        n += 1
        if limit is not None and n >= limit:
            break


# ---------------------------------------------------------------------------
# template averaging
# ---------------------------------------------------------------------------
def template_from_samples(samples: list[np.ndarray]) -> np.ndarray:
    """Average resampled, centered samples into one ``(N, 201)`` template."""
    resampled = [resample_seq(s, TEMPLATE_TARGET_N) for s in samples]
    return np.median(resampled, axis=0).astype(np.float32)


def build_reference_set(data_dir: Path | None = None,
                        max_samples_per_sign: int = 5,
                        min_samples: int = TEMPLATE_MIN_SAMPLES,
                        progress_report: bool = True) -> dict[str, np.ndarray]:
    """Build ``{sign_name: template}`` from the dataset directory."""
    root = data_dir or TRAIN_DIR
    counts: dict[str, int] = {}
    templates: dict[str, np.ndarray] = {}
    for sign_name, _ in iter_vocab_dirs(Path(root)):
        samples = list(iter_sequences(root, sign_name, limit=max_samples_per_sign))
        if len(samples) < min_samples:
            continue
        templates[sign_name] = template_from_samples(samples)
        counts[sign_name] = len(samples)
        if progress_report:
            print(f"\rbuilt {len(templates)}/{counts[sign_name]} {sign_name}", end="")
    if progress_report:
        print()
    return templates


def save_reference_set(templates: dict[str, np.ndarray],
                       out_path: Path | str | None = None) -> Path:
    """Persist templates to a single ``.npz`` cache file."""
    ensure_dirs()
    path = Path(out_path) if out_path else DATA_DIR / "reference" / CACHE_NAME
    keys = sorted(templates)
    arrays: dict[str, np.ndarray] = {
        "signs": np.array(keys, dtype=object),
        **{f"t_{i}": templates[k] for i, k in enumerate(keys)},
    }
    np.savez_compressed(path, **arrays)
    return path


def load_reference_set(cache_path: Path | str | None = None) -> dict[str, np.ndarray]:
    """Load a previously saved reference set (empty dict if absent)."""
    path = Path(cache_path) if cache_path else DATA_DIR / "reference" / CACHE_NAME
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    signs = [str(s) for s in data["signs"].tolist()]
    return {name: data[f"t_{i}"].astype(np.float32) for i, name in enumerate(signs)}


def get_or_build_reference_set(force: bool = False) -> dict[str, np.ndarray]:
    """Return cached templates, building them on first use."""
    cached = load_reference_set()
    if cached and not force:
        return cached
    templates = build_reference_set()
    if templates:
        save_reference_set(templates)
    return templates