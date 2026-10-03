"""Sign classification with two interchangeable backends.

* ``ModelBackend``  — the trained Part 1 LSTM (used when the weights load).
* ``TemplateBackend`` — position-invariant template matching over the
  reference set; used as a trusted-on-our-own-terms fallback when the
  trained model is not available or not trusted.

Both expose the same ``predict`` interface so the rest of the ecosystem
never cares which one is active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .config import (
    CONFIDENCE_THRESHOLD,
    MODEL_PATH,
    SCALER_PATH,
    SEQ_LENGTH,
    TEMPLATE_TARGET_N,
    ensure_dirs,
)
from .models import Scaler, VSLModel, load_vsl_model
from .template_reference import (
    center_sequence,
    get_or_build_reference_set,
    resample_seq,
)
from .vocab import idx_to_label


@dataclass
class Prediction:
    label: str
    index: int
    confidence: float
    top_k: list[tuple[str, float]]
    accepted: bool

    @property
    def is_known(self) -> bool:
        return self.accepted


def _pad_or_trim(sequence: np.ndarray, length: int = SEQ_LENGTH) -> np.ndarray:
    """Pad (zeros) or trim a ``(T, D)`` sequence to a fixed length."""
    seq = np.asarray(sequence, dtype=np.float32)
    t, d = seq.shape
    if t == length:
        return seq
    if t > length:
        return seq[:length]
    out = np.zeros((length, d), dtype=np.float32)
    out[:t] = seq
    return out


class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, sequence: np.ndarray, threshold: float = CONFIDENCE_THRESHOLD) -> Prediction:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class ModelBackend(BaseClassifier):
    """Part 1 LSTM inference wrapper."""

    def __init__(self,
                 model_path: str | None = None,
                 scaler_path: str | None = None,
                 index_to_label: dict[int, str] | None = None) -> None:
        ensure_dirs()
        self.model_path = str(model_path or MODEL_PATH)
        self.scaler_path = str(scaler_path or SCALER_PATH)
        self.ix2label = index_to_label if index_to_label is not None else idx_to_label()
        self.num_classes = len(self.ix2label)
        self.scaler = Scaler.from_npz(self.scaler_path)
        self.model, self.device = load_vsl_model(self.model_path, self.num_classes)

    @staticmethod
    def is_available(model_path: str | None = None,
                     scaler_path: str | None = None) -> bool:
        mp = str(model_path or MODEL_PATH)
        sp = str(scaler_path or SCALER_PATH)
        from pathlib import Path

        return Path(mp).exists() and Path(sp).exists()

    def predict(self, sequence: np.ndarray,
                threshold: float = CONFIDENCE_THRESHOLD) -> Prediction:
        seq = _pad_or_trim(sequence)
        import torch

        x = torch.from_numpy(self.scaler.transform(seq)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            import torch.nn.functional as F

            probs = F.softmax(logits, dim=1)[0]
            top_probs, top_idx = torch.topk(probs, k=min(5, self.num_classes))
        top_probs = top_probs.cpu().numpy()
        top_idx = top_idx.cpu().numpy()
        top_k = [(self.ix2label.get(int(i), "?"), float(p)) for i, p in zip(top_idx, top_probs)]
        index = int(top_idx[0])
        confidence = float(top_probs[0])
        return Prediction(
            label=self.ix2label.get(index, "?"),
            index=index,
            confidence=confidence,
            top_k=top_k,
            accepted=confidence >= threshold,
        )

    def close(self) -> None:
        pass


class TemplateBackend(BaseClassifier):
    """Presence-aware nearest-template classifier (no trained model needed).

    Quad-segment (pose 25 + hands 21+21 = 67) landmarks are matched per
    frame. Landmarks absent from either the query or a candidate template
    are excluded, and a hand-presence penalty guards against degenerate
    all-similar scores when no hands are visible.
    """

    N_FRAMES = TEMPLATE_TARGET_N
    N_LANDMARKS = 67          # 25 pose + 21 left + 21 right
    HAND_WEIGHT = 0.75        # hand shape matters most for signs

    def __init__(self, templates: dict[str, np.ndarray] | None = None,
                 index_to_label: dict[int, str] | None = None) -> None:
        self.templates = templates if templates is not None else get_or_build_reference_set()
        self.ix2label = index_to_label if index_to_label is not None else idx_to_label()
        self.names = sorted(self.templates)

        n = self.N_FRAMES
        C = len(self.names)
        units = np.zeros((C, n * 201), dtype=np.float32)   # landmark-unit vectors
        presence = np.zeros((C, n * 67), dtype=bool)       # per-landmark presence
        for i, name in enumerate(self.names):
            tpl = self.templates[name]
            if tpl.shape[0] != n:
                tpl = resample_seq(tpl, n)
            tpl = center_sequence(tpl.reshape(n, 201))    # match the query space
            units[i] = self._landmark_normalize(tpl).reshape(-1)
            presence[i] = self._present(tpl).reshape(-1)
        self._units = units
        self._presence = presence

    @staticmethod
    def _landmark_normalize(frames: np.ndarray) -> np.ndarray:
        """Unit-normalise each landmark triplet, zeroing absent ones."""
        out = np.zeros_like(frames, dtype=np.float32)
        for j in range(frames.shape[1] // 3):
            block = frames[:, j * 3:(j + 1) * 3]
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            ok = norms > 1e-4
            out[:, j * 3:(j + 1) * 3] = np.where(ok, block / np.clip(norms, 1e-8, None), 0.0)
        return out

    @staticmethod
    def _present(frames: np.ndarray) -> np.ndarray:
        """Per-landmark presence mask ``(N_frames, 67)``."""
        pres = np.zeros((frames.shape[0], 67), dtype=bool)
        for j in range(67):
            block = frames[:, j * 3:(j + 1) * 3]
            pres[:, j] = np.linalg.norm(block, axis=1) > 1e-4
        return pres

    @staticmethod
    def _hand_flag(j: int) -> bool:
        return j >= 25  # pose = 0..24, hands = 25..66

    def predict(self, sequence: np.ndarray,
                threshold: float = CONFIDENCE_THRESHOLD) -> Prediction:
        n = self.N_FRAMES
        seq = center_sequence(np.asarray(sequence, dtype=np.float32))
        seq = resample_seq(seq, n)[:, :201].reshape(n, 201)

        q_units = self._landmark_normalize(seq).reshape(-1)
        q_pres = self._present(seq).reshape(-1)

        if not bool(q_pres[25:].any()):    # no hands at all -> unreliable
            return Prediction("Unknown", -1, 0.0, [], False)

        n_q_hand = int(q_pres[25:].sum())
        n_q_pose = int(q_pres[:25].sum())

        numerators = self._units @ q_units                       # (C,)
        overlaps = self._presence @ q_pres                       # (C,) count shared
        denominator = overlaps.clip(min=1)
        per_candidate = numerators / denominator                 # mean landmark cosine
        cn = per_candidate * overlaps.clip(max=1)                # penalise low overlap

        candidates = []
        for i in range(len(self.names)):
            shared = int(overlaps[i])
            shared_hand = int(self._presence[i][25:] @ q_pres[25:])
            shared_pose = int(self._presence[i][:25] @ q_pres[:25])
            if shared <= 0:
                continue
            hand_cos = numerators[i] / shared if shared else 0.0
            hand_ratio = shared_hand / max(1, n_q_hand)
            pose_ratio = shared_pose / max(1, n_q_pose)
            score = (
                self.HAND_WEIGHT * hand_cos * hand_ratio
                + (1 - self.HAND_WEIGHT) * hand_cos * pose_ratio
            )
            candidates.append((self.names[i], score, hand_cos, shared_hand, shared_pose))

        if not candidates:
            return Prediction("Unknown", -1, 0.0, [], False)

        candidates.sort(key=lambda c: c[1], reverse=True)
        top_k = [(name, float(max(0.0, s))) for name, s, *_ in candidates[:5]]
        name, score, *_ = candidates[0]
        confidence = float(max(0.0, min(1.0, score)))
        index = self.ix2label.get(name, -1)
        return Prediction(
            label=name,
            index=index,
            confidence=confidence,
            top_k=top_k,
            accepted=confidence >= threshold,
        )

    def close(self) -> None:
        pass


def create_classifier(strategy: str = "auto") -> BaseClassifier:
    """Factory returning the best available classifier.

    ``auto`` prefers the trained model, falling back to templates only if
    the model files are missing. Pass ``model``/``template`` to force one.
    """
    if strategy == "model" or (strategy == "auto" and ModelBackend.is_available()):
        try:
            return ModelBackend()
        except Exception:
            if strategy == "model":
                raise
    return TemplateBackend()