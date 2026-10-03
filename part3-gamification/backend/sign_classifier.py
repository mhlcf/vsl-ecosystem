"""Sign classifier for Part 3, delegating to the shared classifier."""

from __future__ import annotations

import numpy as np

from vslshared.classifier import (
    BaseClassifier,
    Prediction,
    create_classifier,
)


class SignClassifier:
    """Game-facing wrapper around the shared dual-backend classifier."""

    def __init__(self, strategy: str = "auto") -> None:
        self._backend: BaseClassifier = create_classifier(strategy)
        self.strategy = strategy

    def predict(self, sequence: np.ndarray,
                threshold: float | None = None) -> Prediction:
        """Classify a keypoint window; returns a ``Prediction``."""
        return self._backend.predict(sequence, threshold or 0.0)

    @property
    def backend(self) -> BaseClassifier:
        return self._backend

    def close(self) -> None:
        self._backend.close()