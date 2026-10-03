"""Tests for vocabulary handling."""

from __future__ import annotations

import pytest

from vslshared.vocab import (
    build_vocabulary,
    idx_to_label,
    load_label_map,
    load_vocabulary,
    stable_bucket,
    vocabulary_by_difficulty,
)


def test_load_label_map(label_map):
    assert len(label_map) >= 3000  # project targets 3000+ signs
    for name, idx in label_map.items():
        assert isinstance(name, str) and name
        assert isinstance(idx, int)


def test_idx_to_label_roundtrip(label_map):
    inv = idx_to_label(label_map)
    assert len(inv) == len(label_map)
    for name, idx in label_map.items():
        assert inv[idx] == name


def test_stable_bucket_range():
    for i in range(50):
        bucket = stable_bucket(f"sign_{i}")
        assert 1 <= bucket <= 3


def test_stable_bucket_deterministic():
    assert stable_bucket("xin chào") == stable_bucket("xin chào")


def test_build_and_load_vocabulary(tmp_path):
    entries = build_vocabulary(tmp_path / "vocab.json")
    assert len(entries) >= 3000
    # unique ids, difficulty in 1..3, animation ids present
    ids = {e["id"] for e in entries}
    assert len(ids) == len(entries)
    assert all(e["difficulty"] in (1, 2, 3) for e in entries)
    assert entries[0]["animation_id"].startswith("anim_")
    reloaded = load_vocabulary(tmp_path / "vocab.json")
    assert reloaded == entries


def test_vocabulary_by_difficulty():
    by = vocabulary_by_difficulty(1)
    assert by
    assert all(e["difficulty"] == 1 for e in by)
    assert len(by) <= len(load_vocabulary())