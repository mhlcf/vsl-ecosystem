"""Live calibration for creating trusted reference templates.

When the existing dataset is not trusted for a sign, a user can quickly
record a few repetitions in front of the webcam; the captured keypoint
sequences are aggregated into a template and merged into the reference set.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from .config import DATA_DIR, ensure_dirs
from .keypoints import HolisticExtractor, extract_keypoints
from .template_reference import (
    center_sequence,
    resample_seq,
    template_from_samples,
    TEMPLATE_TARGET_N,
)

USER_CALIBRATION_CACHE = "user_calibrated.npz"


class LiveCalibrator:
    """Records ``(T, 201)`` sequences from the webcam for one sign."""

    def __init__(self, extractor: HolisticExtractor | None = None) -> None:
        ensure_dirs()
        self.extractor = extractor or HolisticExtractor()

    def capture_sequence(self, duration_sec: float = 2.0,
                         stop: Callable[[], bool] | None = None) -> np.ndarray:
        """Capture one RGB frames sequence from camera and extract keypoints.

        Args:
            duration_sec: approx capture length.
            stop: optional predicate to abort capture early (e.g. key press).

        Returns:
            ``(T, 201)`` keypoint sequence.
        """
        import cv2

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam for calibration.")
        frames: list[np.ndarray] = []
        start = time.time()
        try:
            while time.time() - start < duration_sec:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.extractor.process(rgb)
                frames.append(extract_keypoints(results))
                if stop is not None and stop():
                    break
        finally:
            cap.release()
        if not frames:
            raise RuntimeError("No frames captured during calibration.")
        return np.stack(frames, axis=0).astype(np.float32)

    def calibrate_sign(self, sign_name: str, repetitions: int = 3,
                       duration_sec: float = 1.5,
                       stop: Callable[[], bool] | None = None,
                       save: bool = True) -> np.ndarray:
        """Record ``repetitions`` sequences and return their averaged template."""
        samples: list[np.ndarray] = []
        for _ in range(repetitions):
            seq = self.capture_sequence(duration_sec, stop)
            if seq.shape[1] != 201:
                continue
            samples.append(resample_seq(center_sequence(seq), TEMPLATE_TARGET_N))
        if not samples:
            raise RuntimeError("No usable samples recorded.")
        template = template_from_samples(samples)
        if save:
            self.save(sign_name, template)
        return template

    # -- persistence ---------------------------------------------------------
    def _cache_path(self) -> Path:
        return DATA_DIR / "reference" / USER_CALIBRATION_CACHE

    def save(self, sign_name: str, template: np.ndarray) -> None:
        path = self._cache_path()
        existing: dict[str, np.ndarray] = {}
        if path.exists():
            data = np.load(path, allow_pickle=True)
            signs = [str(s) for s in data["signs"].tolist()]
            existing = {s: data[f"t_{i}"] for i, s in enumerate(signs)}
        existing[sign_name] = template.astype(np.float32)
        np.savez_compressed(
            path,
            signs=np.array(sorted(existing), dtype=object),
            **{f"t_{i}": existing[s] for i, s in enumerate(sorted(existing))},
        )

    def load(self) -> dict[str, np.ndarray]:
        path = self._cache_path()
        if not path.exists():
            return {}
        data = np.load(path, allow_pickle=True)
        signs = [str(s) for s in data["signs"].tolist()]
        return {s: data[f"t_{i}"].astype(np.float32) for i, s in enumerate(signs)}

    def close(self) -> None:
        self.extractor.close()