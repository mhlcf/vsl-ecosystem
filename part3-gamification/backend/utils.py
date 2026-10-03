"""Helper functions for the Part 3 backend."""

from __future__ import annotations

import base64
import binascii
from typing import Any

import numpy as np


def decode_frame_data(
    frame_data: Any,
) -> tuple[np.ndarray | None, str | None]:
    """Decode an incoming ``frame_data`` field.

    Accepts either:
      * a base64-encoded RGB/BGR image string -> ``(frame, "image")``
      * ``{"sequence": [...]}`` of keypoint frames -> ``(array, "sequence")``

    Returns ``(None, reason)`` when the payload is not understood.
    """
    if isinstance(frame_data, bytes):
        frame_data = frame_data.decode("utf-8", errors="ignore")
    if isinstance(frame_data, str):
        try:
            raw = base64.b64decode(frame_data)
            return np.frombuffer(raw, dtype=np.uint8), "image"
        except (binascii.Error, ValueError):
            return None, "invalid_base64"
    if isinstance(frame_data, dict) and "sequence" in frame_data:
        arr = np.asarray(frame_data["sequence"], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 201:
            return None, f"sequence must be (T, 201), got {arr.shape}"
        return arr, "sequence"
    return None, "unsupported_frame_data"

def load_sequence_from_image(frame: np.ndarray, extractor=None) -> np.ndarray:
    """Extract keypoints from a decoded image using the shared extractor."""
    import cv2

    from vslshared.keypoints import HolisticExtractor, extract_keypoints

    img = frame
    if frame.shape[2] == 3:
        # assume base64 of a BGR image -> convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if extractor is None else img
    else:
        rgb = img
    own = extractor is None
    ex = extractor or HolisticExtractor()
    try:
        results = ex.process(rgb)
        seq = extract_keypoints(results).reshape(1, -1)
    finally:
        if own:
            ex.close()
    return seq