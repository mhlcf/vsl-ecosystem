"""Pydantic request/response models for the Part 3 game API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    user_id: int = Field(..., gt=0, description="Database user id")
    level_id: int = Field(..., gt=0, description="1-based level identifier")


class StartSessionResponse(BaseModel):
    session_id: str
    level: int
    word: str                      # normalized sign key ("xin_chào")
    word_name: str                 # human label ("Xin chào")
    video_reference: str           # placeholder URL/file for the demo clip
    instructions: str              # short Vietnamese instruction


class SubmitPoseRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    frame_data: Any = Field(..., description="base64 image string or "
                                             "{'sequence': [[x,y,z,...]*T]}")


class AccuracyBreakdown(BaseModel):
    hand_shape: float = Field(..., ge=0, le=1)
    position: float = Field(..., ge=0, le=1)
    movement: float = Field(..., ge=0, le=1)
    palm_orientation: float = Field(..., ge=0, le=1)


class SubmitPoseResponse(BaseModel):
    status: str = "processing"
    confidence: float = Field(..., ge=0, le=1)
    accuracy_breakdown: AccuracyBreakdown
    feedback: str
    is_correct: bool


class CompleteLevelRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    attempts: int = Field(1, ge=1)
    average_accuracy: float = Field(0.0, ge=0, le=1)


class CompleteLevelResponse(BaseModel):
    status: str = "success"
    stars_earned: int = Field(..., ge=0, le=3)
    level_completed: bool
    next_level_unlocked: bool
    total_points: int


class UserStatsResponse(BaseModel):
    user_id: int
    username: str
    total_levels_completed: int
    total_stars: int
    total_points: int
    progress: list[dict]