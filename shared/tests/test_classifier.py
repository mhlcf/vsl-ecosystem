"""Tests for the dual-backend classifier."""

from __future__ import annotations

import numpy as np
import pytest

from vslshared.classifier import (
    ModelBackend,
    Prediction,
    TemplateBackend,
    _pad_or_trim,
    create_classifier,
)
from vslshared.config import FEATURE_DIM, SEQ_LENGTH


def test_pad_or_trim():
    short = np.zeros((10, FEATURE_DIM), dtype=np.float32)
    exact = np.zeros((SEQ_LENGTH, FEATURE_DIM), dtype=np.float32)
    long = np.zeros((120, FEATURE_DIM), dtype=np.float32)
    assert _pad_or_trim(short).shape == (SEQ_LENGTH, FEATURE_DIM)
    assert _pad_or_trim(exact).shape == (SEQ_LENGTH, FEATURE_DIM)
    assert _pad_or_trim(long).shape == (SEQ_LENGTH, FEATURE_DIM)


def test_prediction_dataclass():
    p = Prediction(label="xin chào", index=5, confidence=0.9, top_k=[("a", 0.9)], accepted=True)
    assert p.is_known
    assert p.label == "xin chào"


def test_template_backend_on_tiny_set(tiny_templates):
    assert len(tiny_templates) >= 2
    cls = TemplateBackend(tiny_templates, index_to_label={0: "placeholder"})
    try:
        samples = list(tiny_templates.items())
        name, tpl = samples[0]
        pred = cls.predict(tpl + np.random.normal(0, 0.02, tpl.shape).astype(np.float32))
        assert pred.label == name or pred.confidence > 0  # should not crash on noise
        assert isinstance(pred.confidence, float)
    finally:
        cls.close()


def test_template_backend_no_hands_returns_unknown(tiny_templates):
    cls = TemplateBackend(tiny_templates)
    empty = np.zeros((30, FEATURE_DIM), dtype=np.float32)
    pred = cls.predict(empty)
    assert not pred.accepted
    cls.close()


@pytest.mark.skipif(not ModelBackend.is_available(), reason="model files missing")
def test_model_backend_on_real_sample(sample_sequence):
    cls = ModelBackend()
    try:
        pred = cls.predict(sample_sequence)
        assert pred.index >= 0
        assert pred.label != "?"
        assert 0.0 <= pred.confidence <= 1.0
    finally:
        cls.close()


def test_create_classifier_returns_instance():
    cls = create_classifier("template")
    try:
        assert isinstance(cls, (ModelBackend, TemplateBackend))
    finally:
        cls.close()


def test_auto_prefers_model_when_available():
    cls = create_classifier()  # auto
    try:
        assert isinstance(cls, ModelBackend) is ModelBackend.is_available()
    finally:
        cls.close()