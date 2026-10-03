"""MediaPipe keypoint extraction.

Produces the exact 201-dim feature vector used to train Part 1:
pose (25 landmarks * 3) + left hand (21 * 3) + right hand (21 * 3).
This module is the single source of truth for live feature extraction,
so part2 and part3 must never re-implement the schema.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .config import FEATURE_DIM, HAND_DIM, NUM_HAND_LANDMARKS, NUM_POSE_LANDMARKS, POSE_DIM


def _landmarks_flat(landmarks, dim: int) -> np.ndarray:
    """Convert a proto landmark list to a flat ``[x, y, z, ...]`` array."""
    out = np.zeros(dim, dtype=np.float32)
    if landmarks is not None:
        n = min(len(landmarks), dim // 3)
        for i in range(n):
            lm = landmarks[i]
            out[3 * i] = lm.x
            out[3 * i + 1] = lm.y
            out[3 * i + 2] = lm.z
    return out


def extract_keypoints(results) -> np.ndarray:
    """Extract the 201-dim feature vector from a MediaPipe Holistic result.

    Args:
        results: a ``mp.solutions.holistic.Holistic`` process() output.

    Returns:
        Concatenated ``[pose(75), left_hand(63), right_hand(63)]`` float32 array.
    """
    pose = _landmarks_flat(
        results.pose_landmarks.landmark if results.pose_landmarks else None,
        POSE_DIM,
    )
    left = _landmarks_flat(
        results.left_hand_landmarks.landmark if results.left_hand_landmarks else None,
        HAND_DIM,
    )
    right = _landmarks_flat(
        results.right_hand_landmarks.landmark if results.right_hand_landmarks else None,
        HAND_DIM,
    )
    return np.concatenate([pose, left, right], axis=0).astype(np.float32)


def split_components(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a ``(T, 201)`` sequence into (pose, left_hand, right_hand).

    Used by the Part 3 scoring engine to evaluate the four criteria
    independently (hand shape, position, movement, palm orientation).
    """
    seq = np.asarray(sequence, dtype=np.float32)
    return seq[:, :POSE_DIM], seq[:, POSE_DIM:POSE_DIM + HAND_DIM], seq[:, POSE_DIM + HAND_DIM:]


def normalize(sequence: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalize a sequence using a pre-computed scaler."""
    return (np.asarray(sequence, dtype=np.float32) - mean.astype(np.float32)) / (
        std.astype(np.float32) + 1e-8
    )


@dataclass
class HolisticOptions:
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class HolisticExtractor:
    """Lazy MediaPipe Holistic wrapper shared across parts.

    Usage:
        extractor = HolisticExtractor()
        results = extractor.process(rgb_frame)
        vec = extract_keypoints(results)
    """

    def __init__(self, options: HolisticOptions | None = None) -> None:
        opts = options or HolisticOptions()
        self._options = opts
        self._holistic = None

    @property
    def holistic(self):
        if self._holistic is None:
            # imported lazily to keep import time low when MediaPipe is absent
            import mediapipe as mp

            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=self._options.model_complexity,
                min_detection_confidence=self._options.min_detection_confidence,
                min_tracking_confidence=self._options.min_tracking_confidence,
            )
        return self._holistic

    def process(self, rgb_frame: np.ndarray):
        """Process a BGR/RGB frame; returns a MediaPipe results object."""
        if rgb_frame.ndim != 3:
            raise ValueError("frame must be a 3-channel image array")
        return self.holistic.process(rgb_frame)

    def close(self) -> None:
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None