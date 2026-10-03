"""4-criteria scoring engine.

Compares a live keypoint window against the reference template for the
target sign and reports the four hand-shape factors from the prompt:
    hand_shape       — relative landmark geometry (position invariant)
    position         — where the hands are in space
    movement         — velocity/trajectory similarity
    palm_orientation — palm facing direction (`hướng lòng bàn tay`)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vslshared.config import (
    HAND_DIM,
    POSE_DIM,
    TRAIN_DIR,
)
from vslshared.template_reference import (
    center_sequence,
    iter_sequences,
    load_reference_set,
    resample_seq,
    template_from_samples,
)

N_FRAMES = 30
EPS = 1e-8

POSE_SHOULDER_LEFT = 11
POSE_SHOULDER_RIGHT = 12

HAND_INDEX_MCP = 5
HAND_PINKY_MCP = 17
HAND_MIDDLE_TIP = 12

# closeness weights for the composite confidence
WEIGHTS = {
    "hand_shape": 0.35,
    "position": 0.25,
    "movement": 0.20,
    "palm_orientation": 0.20,
}

_CACHE: dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class CriteriaScores:
    hand_shape: float
    position: float
    movement: float
    palm_orientation: float

    def as_dict(self) -> dict[str, float]:
        return {
            "hand_shape": round(float(self.hand_shape), 3),
            "position": round(float(self.position), 3),
            "movement": round(float(self.movement), 3),
            "palm_orientation": round(float(self.palm_orientation), 3),
        }

    def confidence(self) -> float:
        return float(sum(WEIGHTS[k] * getattr(self, k) for k in WEIGHTS))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _hand_blocks(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (seq[:, POSE_DIM:POSE_DIM + HAND_DIM],
            seq[:, POSE_DIM + HAND_DIM:])


def _hand_centroids(seq: np.ndarray, scale: float) -> np.ndarray:
    """Mean of present hand landmarks per frame -> ``(T, 2, 3)`` normalized by scale."""
    left, right = _hand_blocks(seq)
    out = np.zeros((seq.shape[0], 2, 3), dtype=np.float32)
    for i, hand in enumerate((left, right)):
        norms = np.linalg.norm(hand.reshape(-1, 21, 3), axis=2)  # (T, 21)
        present = norms > 1e-4
        sums = np.where(present.reshape(*present.shape, 1), hand.reshape(-1, 21, 3), 0.0).sum(axis=1)
        counts = present.sum(axis=1).clip(min=1)
        out[:, i] = sums / counts[:, None]
    return out / max(scale, EPS)


def _pose_scale(seq: np.ndarray) -> float:
    """Distance between shoulders as a natural scale for normalized positions."""
    ls = seq[:, POSE_SHOULDER_LEFT * 3: POSE_SHOULDER_LEFT * 3 + 3]
    rs = seq[:, POSE_SHOULDER_RIGHT * 3: POSE_SHOULDER_RIGHT * 3 + 3]
    d = np.linalg.norm(ls - rs, axis=1).mean()
    return float(d) if d > 1e-4 else 1.0


def _palm_normals(seq: np.ndarray) -> np.ndarray:
    """Per frame, per hand palm normal ``(T, 2, 3)``, zeros when unavailable."""
    left, right = _hand_blocks(seq)
    normals = np.zeros((seq.shape[0], 2, 3), dtype=np.float32)
    for i, hand in enumerate((left, right)):
        ld = hand.reshape(-1, 21, 3)
        wrist = ld[:, 0]
        index_mcp = ld[:, HAND_INDEX_MCP]
        pinky_mcp = ld[:, HAND_PINKY_MCP]
        mid_tip = ld[:, HAND_MIDDLE_TIP]
        u = index_mcp - wrist
        v = pinky_mcp - wrist
        n = np.cross(u, v)
        fallback = mid_tip - wrist
        n_n = np.linalg.norm(n, axis=-1, keepdims=True)
        f_n = np.linalg.norm(fallback, axis=-1, keepdims=True)
        ok = (n_n > EPS) & (f_n > EPS)
        normals[:, i] = np.where(ok, n / (n_n + EPS), fallback / (f_n + EPS))
    return normals


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
def _landmark_unit(triplets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize landmark triplets; return (units, presence) ``(T, 42, 3)``."""
    units = np.zeros_like(triplets)
    norms = np.linalg.norm(triplets, axis=-1)
    ok = norms > EPS
    units[ok] = triplets[ok] / norms[ok][:, None]
    return units, ok


def _hand_landmarks(seq: np.ndarray) -> np.ndarray:
    """Wrist-centered hand landmarks as ``(T, 42, 3)`` concatenating both hands."""
    centered = center_sequence(seq).astype(np.float32)
    left, right = _hand_blocks(centered)
    t = centered.shape[0]
    return np.concatenate(
        [left.reshape(t, 21, 3), right.reshape(t, 21, 3)], axis=1
    ).astype(np.float32)


def hand_shape_score(query: np.ndarray, template: np.ndarray) -> float:
    """Presence-aware per-landmark cosine of wrist-centered hand geometry."""
    q_land = _hand_landmarks(query)
    t_land = _hand_landmarks(template)
    q_u, q_ok = _landmark_unit(q_land)
    t_u, t_ok = _landmark_unit(t_land)
    both = q_ok & t_ok
    denom = int(both.sum())
    if denom == 0:
        return 0.0
    dots = (q_u * t_u).sum()  # only where both present (zeroed elsewhere)
    return float(max(0.0, min(1.0, dots / denom)))


def position_score(query: np.ndarray, template: np.ndarray) -> float:
    scale = max(_pose_scale(query), _pose_scale(template))
    cq = _hand_centroids(query, scale)
    ct = _hand_centroids(template, scale)
    dist = np.linalg.norm(cq - ct, axis=-1)               # (T, 2)
    per_frame = 1.0 - np.clip(dist.mean(axis=-1), 0.0, 1.0)
    return float(np.clip(per_frame.mean(), 0.0, 1.0))


def movement_score(query: np.ndarray, template: np.ndarray) -> float:
    scale = max(_pose_scale(query), _pose_scale(template))
    cq = _hand_centroids(query, scale)
    ct = _hand_centroids(template, scale)
    vq = np.diff(cq, axis=0)
    vt = np.diff(ct, axis=0)
    scores = []
    for h in range(2):
        for t in range(len(vq)):
            qn, tn_ = np.linalg.norm(vq[t, h]), np.linalg.norm(vt[t, h])
            if qn < EPS and tn_ < EPS:
                scores.append(1.0)
            elif qn < EPS or tn_ < EPS:
                scores.append(0.0)
            else:
                scores.append(float((vq[t, h] * vt[t, h]).sum()) / (qn * tn_))
    return float(np.mean(scores)) if scores else 0.0


def palm_orientation_score(query: np.ndarray, template: np.ndarray) -> float:
    nq = _palm_normals(query)
    nt = _palm_normals(template)
    scores = []
    for h in range(2):
        qn = np.linalg.norm(nq[:, h], axis=-1)
        tn = np.linalg.norm(nt[:, h], axis=-1)
        both = (qn > EPS) & (tn > EPS)
        if not bool(both.any()):
            continue
        dots = (nq[:, h] * nt[:, h]).sum(axis=-1)
        scores.append(float((dots[both] / (qn[both] * tn[both])).mean()))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# template resolution + top-level scoring
# ---------------------------------------------------------------------------
def _get_reference_template(sign_name: str) -> np.ndarray | None:
    global _CACHE
    if sign_name in _CACHE:
        return _CACHE[sign_name]
    ref = load_reference_set()
    tpl = ref.get(sign_name)
    if tpl is None:
        samples = list(iter_sequences(TRAIN_DIR, sign_name, limit=5))
        tpl = template_from_samples(samples) if samples else None
    if tpl is not None:
        _CACHE[sign_name] = tpl
    return tpl


def score_sign(sequence: np.ndarray, sign_name: str) -> CriteriaScores | None:
    """Score a live ``(T, 201)`` sequence against the reference template."""
    template = _get_reference_template(sign_name)
    if template is None:
        return None
    query = resample_seq(np.asarray(sequence, dtype=np.float32), N_FRAMES)
    tpl = resample_seq(template, N_FRAMES)
    return CriteriaScores(
        hand_shape=hand_shape_score(query, tpl),
        position=position_score(query, tpl),
        movement=movement_score(query, tpl),
        palm_orientation=palm_orientation_score(query, tpl),
    )


def classify_live(sequence: np.ndarray, sign_name: str,
                  threshold: float | None = None) -> dict | None:
    """Score a sequence and annotate it like the game API does.

    Returns ``None`` when the target sign has no reference template.
    Otherwise a dict with ``accuracy``, ``accuracy_breakdown``,
    ``is_correct`` and the configured ``threshold`` used for the verdict.
    """
    from .config import get_settings

    if threshold is None:
        threshold = get_settings().confidence_threshold
    scores = score_sign(sequence, sign_name)
    if scores is None:
        return None
    confidence = scores.confidence()
    return {
        "accuracy": round(float(confidence), 3),
        "accuracy_breakdown": scores.as_dict(),
        "is_correct": bool(confidence >= threshold),
        "threshold": float(threshold),
    }