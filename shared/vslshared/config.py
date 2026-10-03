"""Shared configuration for the VSL ecosystem.

Centralises paths, model locations and tunable constants so that neither
part2-speech-to-sign nor part3-gamification hardcodes magic values.
All values can be overridden through environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# shared/vslshared/config.py -> shared -> <repo root>
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR: Path = Path(_env("VSL_PROCESSED_DIR", str(REPO_ROOT / "part1-sign-classifier")))

LABEL_MAP_PATH: Path = Path(_env("VSL_LABEL_MAP", str(PROCESSED_DIR / "label_map.json")))
MODEL_PATH: Path = Path(_env("VSL_MODEL_PATH", str(PROCESSED_DIR / "best_model.pth")))
SCALER_PATH: Path = Path(_env("VSL_SCALER_PATH", str(PROCESSED_DIR / "scaler.npz")))
TRAIN_DIR: Path = Path(_env("VSL_TRAIN_DIR", str(PROCESSED_DIR / "train")))
VAL_DIR: Path = Path(_env("VSL_VAL_DIR", str(PROCESSED_DIR / "val")))
TEST_DIR: Path = Path(_env("VSL_TEST_DIR", str(PROCESSED_DIR / "test")))

# Local database + caches (inside the repo, version-controlled friendly)
DATA_DIR: Path = Path(_env("VSL_DATA_DIR", str(REPO_ROOT / "vsl_data")))
DB_PATH: Path = Path(_env("VSL_DB_PATH", str(DATA_DIR / "vsl.db")))
REFERENCE_SET_PATH: Path = Path(_env(
    "VSL_REFERENCE_SET", str(DATA_DIR / "reference_set.parquet")
))
VOCABULARY_PATH: Path = Path(_env(
    "VSL_VOCABULARY", str(DATA_DIR / "vocabulary.json")
))

# ---------------------------------------------------------------------------
# Feature geometry (must match Part 1 training schema)
# ---------------------------------------------------------------------------
NUM_POSE_LANDMARKS: int = 25
NUM_HAND_LANDMARKS: int = 21
POSE_DIM: int = NUM_POSE_LANDMARKS * 3        # 75
HAND_DIM: int = NUM_HAND_LANDMARKS * 3        # 63
FEATURE_DIM: int = POSE_DIM + HAND_DIM * 2    # 201

# ---------------------------------------------------------------------------
# Model / inference
# ---------------------------------------------------------------------------
SEQ_LENGTH: int = 60          # frames per classification window
CONFIDENCE_THRESHOLD: float = 0.30  # minimum confidence to accept a prediction
SMOOTHING_WINDOW: int = 5     # majority vote over recent frames

# Template matching fallback
TEMPLATE_MIN_SAMPLES: int = 3          # minimum samples to build a template
TEMPLATE_TOP_K: int = 5                # how many candidates to aggregate
TEMPLATE_TARGET_N: int = 30            # target frames per normalized template

# ---------------------------------------------------------------------------
# Scoring (Part 3) — 4 criteria
# ---------------------------------------------------------------------------
STAR_3_THRESHOLD: float = 0.90
STAR_2_THRESHOLD: float = 0.80
STAR_1_THRESHOLD: float = 0.70
CORRECT_CONFIDENCE: float = 0.70

# ---------------------------------------------------------------------------
# Part 3 level design
# ---------------------------------------------------------------------------
LEVELS_PER_DIFFICULTY: int = 10          # words per level
DIFFICULTIES: tuple[str, ...] = ("beginner", "intermediate", "advanced")


def ensure_dirs() -> None:
    """Create the local data directory and cache dirs if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "caches").mkdir(parents=True, exist_ok=True)
    PROC_REFERENCE_DIR = DATA_DIR / "reference"
    PROC_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)