"""FastAPI game server for Part 3 (Gamification Learning VSL).

Endpoints follow the project prompt (section 4.3.2):
    POST /api/v1/game/start-session
    POST /api/v1/game/submit-pose
    POST /api/v1/game/complete-level
plus a small stats endpoint for the profile screen.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import get_db
from .feedback_generator import generate_feedback
from .level_manager import (
    TOTAL_LEVELS,
    instructions_for,
    level_for,
    level_is_unlocked,
    next_level_id,
    stars_for_accuracy,
)
from .models import (
    AccuracyBreakdown,
    CompleteLevelRequest,
    CompleteLevelResponse,
    StartSessionRequest,
    StartSessionResponse,
    SubmitPoseRequest,
    SubmitPoseResponse,
    UserStatsResponse,
)
from .scoring_engine import score_sign
from .sign_classifier import SignClassifier
from .user_manager import UserManager
from .utils import decode_frame_data

logger = logging.getLogger("part3.game")
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

settings = get_settings()
db = get_db()
users = UserManager(db)

_sessions: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine = SignClassifier(settings.classifier_strategy)
    _app.state.classifier = engine
    logger.info(
        "classifier backend ready: %s",
        type(engine.backend).__name__,
    )
    yield
    engine.close()


app = FastAPI(title="VSL Gamification API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _scores_to_response(scores, threshold: float) -> SubmitPoseResponse:
    breakdown = scores.as_dict()
    confidence = scores.confidence()
    is_correct = confidence >= threshold
    text, _hint = generate_feedback(scores, is_correct)
    return SubmitPoseResponse(
        status="processing",
        confidence=round(confidence, 3),
        accuracy_breakdown=AccuracyBreakdown(**breakdown),
        feedback=text,
        is_correct=is_correct,
    )


@app.get("/health", include_in_schema=False)
def health() -> dict:
    return {"status": "ok", "classifier": type(app.state.classifier.backend).__name__}


@app.post("/api/v1/game/start-session", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest) -> StartSessionResponse:
    """Create a session teaching the sign for ``level_id``."""
    if users.get_user(req.user_id) is None:
        raise HTTPException(status_code=404, detail="user not found")
    level = level_for(req.level_id)
    session_id = db.start_session(req.user_id, req.level_id)
    _sessions[session_id] = {
        "user_id": req.user_id,
        "level_id": req.level_id,
        "sign_name": level.sign_name,
        "attempts": 0,
    }
    return StartSessionResponse(
        session_id=session_id,
        level=req.level_id,
        word=level.sign_key,
        word_name=level.sign_name,
        video_reference=f"templates://{level.sign_key}",  # frontend resolves locally
        instructions=instructions_for(level),
    )


@app.post("/api/v1/game/submit-pose", response_model=SubmitPoseResponse)
def submit_pose(req: SubmitPoseRequest) -> SubmitPoseResponse:
    """Score a pose/sequence against the target sign of the session."""
    session = _sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    sequence, _kind = decode_frame_data(req.frame_data)
    if _kind in ("invalid_base64", "unsupported_frame_data"):
        raise HTTPException(status_code=400, detail=f"bad frame_data: {_kind}")

    if sequence is None:
        raise HTTPException(status_code=400, detail="could not decode frame_data")
    if sequence.shape[1] != 201:
        raise HTTPException(status_code=400, detail="sequence must be (T, 201)")

    scores = score_sign(sequence, session["sign_name"])
    if scores is None:
        raise HTTPException(status_code=409, detail="no reference template for target sign")

    resp = _scores_to_response(scores, settings.confidence_threshold)
    session["attempts"] += 1
    db.log_attempt(
        req.session_id,
        sign_id=None,
        scores=resp.accuracy_breakdown.model_dump(),
        correct=bool(resp.is_correct),
    )
    return resp


@app.post("/api/v1/game/complete-level", response_model=CompleteLevelResponse)
def complete_level(req: CompleteLevelRequest) -> CompleteLevelResponse:
    """Finish a session, persisting progress and unlocking the next level."""
    session = _sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    stars = stars_for_accuracy(req.average_accuracy)
    completed = stars > 0
    db.end_session(req.session_id)
    if completed:
        users.record_completion(
            session["user_id"], session["level_id"],
            stars, req.average_accuracy, session["attempts"],
        )

    next_unlocked = next_level_id(session["level_id"]) is not None
    total_points = stars * settings.points_per_star
    _sessions.pop(req.session_id, None)

    return CompleteLevelResponse(
        status="success",
        stars_earned=stars,
        level_completed=completed,
        next_level_unlocked=next_unlocked,
        total_points=total_points,
    )


@app.get("/api/v1/game/levels", include_in_schema=False)
def list_levels() -> list[dict]:
    """Metadata for all levels (locked state needs a user id)."""
    return [
        {
            "level_id": level_for(i).level_id,
            "chapter": level_for(i).chapter,
            "sign_name": level_for(i).sign_name,
            "sign_key": level_for(i).sign_key,
        }
        for i in range(1, TOTAL_LEVELS + 1)
    ]


@app.get("/api/v1/users/{user_id}/stats", response_model=UserStatsResponse)
def user_stats(user_id: int) -> UserStatsResponse:
    user = users.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    stats = users.stats(user_id)
    return UserStatsResponse(
        user_id=user_id,
        username=user["username"],
        total_levels_completed=stats["total_levels_completed"],
        total_stars=stats["total_stars"],
        total_points=stats["total_points"],
        progress=stats["progress"],
    )

def main() -> None:
    """Run the Part 3 Gamification API server (default port 8002)."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Part 3 Gamification API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(8002))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
