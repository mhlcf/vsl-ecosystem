"""Real-time feedback generation for Part 3 gameplay."""

from __future__ import annotations

from dataclasses import dataclass

from .scoring_engine import WEIGHTS, CriteriaScores


@dataclass(frozen=True)
class ZoneHint:
    """A hint plus the spatial zone to highlight in red on the overlay."""

    criterion: str
    hint: str
    zone: str          # "left_hand" | "right_hand" | "both_hands" | "pose"


_ZONE_BY_CRITERION = {
    "hand_shape": "both_hands",
    "position": "both_hands",
    "movement": "both_hands",
    "palm_orientation": "both_hands",
}

# Vietnamese hints, mapped by (criterion, weak)
_HINTS = {
    "hand_shape": "Hãy uốn ngón tay đúng hình dáng của cử chỉ.",
    "position": "Hãy đưa tay lên đúng vị trí không gian của cử chỉ.",
    "movement": "Hãy lặp lại chuyển động của tay theo đúng quỹ đạo.",
    "palm_orientation": "Hãy xoay lòng bàn tay về đúng hướng.",
}


def _criterion_values(scores: CriteriaScores | dict) -> dict[str, float]:
    if isinstance(scores, dict):
        return {k: float(scores[k]) for k in WEIGHTS}
    return {k: getattr(scores, k) for k in WEIGHTS}


def weakest_criterion(scores: CriteriaScores | dict) -> str:
    """Name of the criterion with the lowest score (must-improve target)."""
    return min(_criterion_values(scores), key=_criterion_values(scores).get)


def generate_feedback(scores: CriteriaScores | dict,
                      is_correct: bool) -> tuple[str, ZoneHint]:
    """Return ``(text, zone_hint)`` advising the learner how to improve."""
    weakest = weakest_criterion(scores)
    if is_correct:
        text = "Chính xác! Làm tốt lắm."
    else:
        text = _HINTS[weakest]
    hint = ZoneHint(
        criterion=weakest,
        hint=text,
        zone=_ZONE_BY_CRITERION[weakest],
    )
    return text, hint