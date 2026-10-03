"""User management and progress wrappers over the shared database."""

from __future__ import annotations

from vslshared.db import Database

from .config import get_settings


class UserManager:
    def __init__(self, db: Database | None = None) -> None:
        self.db = db if db is not None else Database()

    def get_or_create_user(self, username: str, email: str | None = None) -> dict:
        user_id = self.db.create_user(username, email)
        user = self.db.get_user(user_id)
        assert user is not None
        return user

    def get_user(self, user_id: int) -> dict | None:
        return self.db.get_user(user_id)

    def stats(self, user_id: int) -> dict:
        progress = self.db.get_progress(user_id)
        total_stars = sum(int(p["stars_earned"]) for p in progress)
        total_points = total_stars * get_settings().points_per_star
        return {
            "user_id": user_id,
            "total_levels_completed": len([p for p in progress if p["stars_earned"] > 0]),
            "total_stars": total_stars,
            "total_points": total_points,
            "progress": progress,
        }

    def completed_levels(self, user_id: int) -> list[int]:
        return [int(p["level_id"]) for p in self.db.get_progress(user_id)
                if p["stars_earned"] > 0]

    def stars_per_level(self, user_id: int) -> list[tuple[int, int]]:
        return [(int(p["level_id"]), int(p["stars_earned"]))
                for p in self.db.get_progress(user_id) if p["stars_earned"] > 0]

    def record_completion(self, user_id: int, level_id: int, stars: int,
                          accuracy: float, attempts: int) -> None:
        self.db.record_progress(user_id, level_id, stars, accuracy, attempts)