"""Part 3 backend configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vslshared.config import DB_PATH as _DB_PATH
from vslshared.config import DATA_DIR as _DATA_DIR

from pydantic_settings import BaseSettings, SettingsConfigDict


class GameSettings(BaseSettings):
    """Runtime settings readable from environment or a local ``.env`` file."""

    model_config = SettingsConfigDict(env_prefix="VSL_GAME_", env_file=".env")

    db_path: Path = _DB_PATH
    data_dir: Path = _DATA_DIR
    classifier_strategy: str = "auto"      # auto | model | template
    confidence_threshold: float = 0.70     # minimum to accept a sign as correct
    points_per_star: int = 50


def get_settings() -> GameSettings:
    return GameSettings()