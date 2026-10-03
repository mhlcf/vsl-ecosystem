"""Vocabulary handling.

Loads the Part 1 label map and produces the canonical vocabulary used by
both parts: each sign gets a stable id, a difficulty bucket (1-3) and an
animation id used by the Part 2 avatar mapper.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Iterator

from .config import LABEL_MAP_PATH, VOCABULARY_PATH, ensure_dirs


def load_label_map(path: Path | str | None = None) -> dict[str, int]:
    """Load the ``{sign_name: class_index}`` mapping from JSON."""
    p = Path(path) if path else LABEL_MAP_PATH
    if not p.exists():
        raise FileNotFoundError(f"Label map not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def idx_to_label(label_map: dict[str, int] | None = None) -> dict[int, str]:
    """Invert ``{name: index}`` to ``{index: name}``."""
    lm = label_map if label_map is not None else load_label_map()
    return {int(v): k for k, v in lm.items()}


def stable_bucket(sign_name: str, n_buckets: int = 3) -> int:
    """Deterministic difficulty bucket in [1, n_buckets] for a sign name."""
    crc = zlib.crc32(sign_name.encode("utf-8"))
    return (crc % n_buckets) + 1


def iter_signs() -> Iterator[tuple[str, int]]:
    """Yield ``(sign_name, class_index)`` from the Part 1 label map."""
    for name, idx in load_label_map().items():
        yield name, int(idx)


def build_vocabulary(path: Path | str | None = None) -> list[dict]:
    """Build the canonical vocabulary.json next to the label map.

    Each entry has the shape consumed by part 3 (levels) and part 2
    (animation mapping):
        {id, sign_name, class_index, viet_translation, difficulty, animation_id}
    """
    ensure_dirs()
    out = VOCABULARY_PATH if path is None else Path(path)
    entries = []
    for name, idx in iter_signs():
        difficulty = stable_bucket(name)
        entries.append({
            "id": idx,
            "sign_name": name,
            "class_index": idx,
            "viet_translation": name,
            "difficulty": difficulty,
            "animation_id": f"anim_{idx:05d}",
        })
    entries.sort(key=lambda e: e["id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return entries


def load_vocabulary(path: Path | str | None = None) -> list[dict]:
    """Load vocabulary.json (building it first if absent)."""
    p = VOCABULARY_PATH if path is None else Path(path)
    if not p.exists():
        return build_vocabulary(p)
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def vocabulary_by_difficulty(difficulty: int) -> list[dict]:
    """Signs assigned to a given difficulty bucket (1 = easiest, 3 = hardest)."""
    return [e for e in load_vocabulary() if e["difficulty"] == difficulty]