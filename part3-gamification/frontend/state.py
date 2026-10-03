"""App application state shared across screens."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.level_manager import Level


@dataclass
class GameSession:
    user_id: int = 0
    username: str = ""
    level: Level | None = None
    attempts: int = 0
    last_scores: dict = field(default_factory=dict)
    last_stars: int = 0
    completed: bool = False