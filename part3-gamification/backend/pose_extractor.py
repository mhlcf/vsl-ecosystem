"""MediaPipe pose extraction for Part 3 gameplay."""

from __future__ import annotations

import numpy as np

from vslshared.keypoints import HolisticExtractor, extract_keypoints


class PoseExtractor:
    """Thin wrapper around the shared keypoint extractor for the game loop."""

    def __init__(self, buffer_frames: int = 60) -> None:
        from collections import deque

        self._extractor = HolisticExtractor()
        self.buffer_frames = buffer_frames
        self._window: "deque[np.ndarray]" = deque(maxlen=buffer_frames)

    def process_frame(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Extract and push a frame; returns the current ``(T, 201)`` window."""
        results = self._extractor.process(rgb_frame)
        self._window.append(extract_keypoints(results))
        import numpy as np

        return np.stack(list(self._window), axis=0).astype(np.float32)

    def window(self) -> np.ndarray:
        """Current buffered sequence without processing a new frame."""
        import numpy as np

        if not self._window:
            return np.zeros((1, 201), dtype=np.float32)
        return np.stack(list(self._window), axis=0).astype(np.float32)

    def reset(self) -> None:
        self._window.clear()

    def close(self) -> None:
        self._extractor.close()