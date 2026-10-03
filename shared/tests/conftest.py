"""Shared pytest fixtures pointing at the real Part 1 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vslshared.config import (
    LABEL_MAP_PATH,
    MODEL_PATH,
    SCALER_PATH,
    TEST_DIR,
    TRAIN_DIR,
)

HAS_DATASET = TRAIN_DIR.exists() and bool(list(TRAIN_DIR.iterdir())[:1])
HAS_MODEL = MODEL_PATH.exists() and SCALER_PATH.exists()


@pytest.fixture(scope="session")
def label_map() -> dict[str, int]:
    with open(LABEL_MAP_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def sample_sign_name() -> str:
    assert HAS_DATASET, "train dataset required"
    return next(iter(sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())))


@pytest.fixture(scope="session")
def sample_sequence(sample_sign_name: str):
    """A single real ``(T, 201)`` sequence for the sample sign."""
    from vslshared.template_reference import iter_sequences

    seq = next(iter_sequences(TRAIN_DIR, sample_sign_name, limit=1), None)
    assert seq is not None, "no usable sample sequences"
    return seq


def requires(model: bool):
    return pytest.mark.skipif(not HAS_MODEL, reason="model artifacts missing")


@pytest.fixture(scope="session")
def tiny_templates() -> dict[str, "np.ndarray"]:
    """Reference set built from a few signs only (fast, no 60MB cache)."""
    import numpy as np

    from vslshared.template_reference import (
        TEMPLATE_TARGET_N,
        center_sequence,
        iter_sequences,
        resample_seq,
        template_from_samples,
    )

    signs = []
    for p in sorted(list(TRAIN_DIR.iterdir()))[:4]:
        if p.is_dir():
            signs.append(p.name)
    templates = {}
    for name in signs:
        samples = list(iter_sequences(TRAIN_DIR, name, limit=3))
        if samples:
            templates[name] = template_from_samples(
                [resample_seq(center_sequence(s), TEMPLATE_TARGET_N) for s in samples]
            )
    return templates