"""Local mirror of the Part 1 LSTM model definition.

Keeps a copy of ``part1-sign-classifier/model.py``'s ``VSLModel`` so the rest of the
ecosystem does not depend on the internal layout of the Part 1 folder.
Loaded weights are byte-compatible with the trained ``best_model.pth``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VSLModel(nn.Module):
    """Bidirectional LSTM classifier trained on 201-dim keypoint sequences.

    Mirrors ``part1-sign-classifier/model.py``:
        LSTM(201 -> 384, 3 layers, bidirectional)
        -> mean pooling -> BatchNorm -> MLP(768 -> 512) -> logits(num_classes)
    """

    def __init__(self, input_dim: int = 201, hidden_dim: int = 384,
                 num_layers: int = 3, num_classes: int = 3315,
                 dropout: float = 0.4) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2
        self.bn = nn.BatchNorm1d(lstm_out_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout + 0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = lstm_out.mean(dim=1)
        out = self.bn(out)
        return self.classifier(out)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs


class Scaler:
    """Z-score scaler backed by a ``.npz`` file with ``mean``/``std`` keys."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def from_npz(cls, path: str) -> "Scaler":
        data = np.load(path)
        mean = np.asarray(data["mean"]).reshape(-1).astype(np.float32)
        std = np.asarray(data["std"]).reshape(-1).astype(np.float32)
        return cls(mean, std)

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim == 3 and seq.shape[0] == 1:
            seq = seq[0]
        if seq.ndim != 2:
            raise ValueError(f"expected a (T, D) sequence, got {seq.shape}")
        return (seq - self.mean) / (self.std + 1e-8)


def load_vsl_model(model_path: str, num_classes: int,
                   device: str | None = None) -> tuple[VSLModel, torch.device]:
    """Load ``best_model.pth`` into a ``VSLModel`` on the best available device."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = VSLModel(num_classes=num_classes).to(dev)
    state: dict[str, Any] = torch.load(
        model_path, map_location=dev, weights_only=True
    )
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model, dev