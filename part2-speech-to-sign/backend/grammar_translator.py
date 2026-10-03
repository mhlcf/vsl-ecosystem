"""Vietnamese → VSL grammar translator (rule-based + optional local LLM).

VSL is not signed word-for-word; it follows topic-comment structure and
drops function words. This module converts a transcribed Vietnamese
sentence into the VSL structure the prompt specifies:

    {
      "topic":   "bạn",                   # what the sentence is about
      "comment": "xin chào",              # what is said about the topic
      "words":   ["xin", "chào", "bạn"]   # ordered signs to perform
    }

The implementation is deterministic and offline by default. If
``VSL_LLM_ENABLED=1`` and an Ollama endpoint responds, the same pipeline
is refined by the LLM; on any failure it falls back to the rules so the
API never blocks on external services.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Iterable

from .config import get_settings

# ---------------------------------------------------------------------------
# Vietnamese helpers
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-zÀ-ỹ0-9]+", re.UNICODE)

# function words dropped from the sign sequence (VSL skips them)
_STOPWORDS = {
    "là", "của", "và", "hoặc", "nhưng", "thì", "mà", "còn", "để", "cho",
    "bởi", "với", "về", "tại", "trên", "dưới", "trong", "ngoài", "nên",
    "vì", "cũng", "đã", "đang", "sẽ", "rất", "quá", "lắm", "hơi", "nữa",
    "được", "bị", "có",
}

# particle "ơi" after a vocative is kept ("Minh ơi" → "Minh")
_VOCATIVE = {"ơi"}

# negation marker: VSL signs the negative AFTER the verb/state
_NEGATION = {"không", "chẳng", "chưa", "đừng", "chớ"}

# common fixed greetings/phrases mapped to canonical VSL word lists
_PHRASES: dict[str, list[str]] = {
    "xin chào": ["xin", "chào"],
    "xin_chao": ["xin", "chào"],
    "hello": ["xin", "chào"],
    "hi": ["xin", "chào"],
    "chào": ["chào"],
    "tạm biệt": ["tạm", "biệt"],
    "hẹn gặp lại": ["hẹn", "gặp", "lại"],
    "cảm ơn": ["cảm", "ơn"],
    "xin lỗi": ["xin", "lỗi"],
    "khỏe không": ["khỏe", "không"],
    "khỏe": ["khỏe"],
    "rất khỏe": ["rất", "khỏe"],
    "bạn khỏe không": ["bạn", "khỏe", "không"],
    "tôi khỏe": ["tôi", "khỏe"],
    "tôi tên": ["tôi", "tên"],
    "tên tôi là": ["tôi", "tên"],
    "bao nhiêu": ["bao", "nhiêu"],
    "bao lâu": ["bao", "lâu"],
    "mấy giờ": ["mấy", "giờ"],
    "mấy tuổi": ["mấy", "tuổi"],
    "yêu": ["yêu"],
    "thích": ["thích"],
}

# default topic when none is detectable
_DEFAULT_TOPIC = "chung"


def tokenize(text: str) -> list[str]:
    """Lowercased tokens (unicode words), preserving Vietnamese diacritics."""
    return _WORD_RE.findall(text.lower())


def strip_punctuation(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower())).strip()


def content_words(tokens: Iterable[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        if tok in _VOCATIVE and out:
            continue
        out.append(tok)
    return out


def _detect_topic(tokens: list[str]) -> str | None:
    """Heuristic topic: an explicit pronoun/noun phrase at the start."""
    for tok in tokens:
        if tok in ("bạn", "anh", "chị", "em", "tôi", "ta", "mình",
                   "ông", "bà", "cô", "chú", "bác", "cậu"):
            return tok
        if tok in _PHRASES and tok not in ("tôi tên",):
            continue
    return None


def _segment_phrases(text: str) -> tuple[list[str], str]:
    """Greedy left-to-right scan replacing known phrases by their VSL signs.

    Returns ``(words, remainder)`` where remainder holds the unmatched
    content still needing generic tokenisation.
    """
    words: list[str] = []
    matched_up_to = 0
    i = 0
    phrases = sorted(_PHRASES, key=len, reverse=True)
    while i < len(text):
        hit = None
        for phrase in phrases:
            if text.startswith(phrase, i):
                hit = phrase
                break
        if hit is not None:
            if i > matched_up_to:
                words += tokenize(text[matched_up_to:i])
            words += _PHRASES[hit]
            matched_up_to = i + len(hit)
            i = matched_up_to
        else:
            i += 1
    remainder = text[matched_up_to:]
    return words, remainder


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def translate_to_vsl(text: str) -> dict:
    """Convert a Vietnamese sentence into a VSL structure."""
    if not text or not text.strip():
        return _empty_structure()

    normalized = re.sub(r"\s+", " ", text.strip().lower())
    tokens = tokenize(normalized)

    # fixed phrases ("xin chào", "bạn khỏe không", ...) map to canonical
    # VSL sign lists; the greedy scan preserves sentence order
    phrase_words, remainder = _segment_phrases(normalized)
    if phrase_words:
        words = list(phrase_words)
        if remainder.strip():
            words += content_words(tokenize(remainder))
        words = content_words(words)
    else:
        words = content_words(tokens)

    topic = _detect_topic(tokens) or _DEFAULT_TOPIC
    if not words and tokens:
        words = [t for t in tokens if t not in _STOPWORDS]

    # negation positioned after its verb in VSL
    words = _apply_negation(words)
    comment = " ".join(words) if words else normalized
    return {
        "topic": topic,
        "comment": comment,
        "words": words,
    }


def _apply_negation(words: list[str]) -> list[str]:
    """VSL signs negation AFTER the verb/state word."""
    negations = [w for w in words if w in _NEGATION]
    others = [w for w in words if w not in _NEGATION]
    if not negations:
        return others if others else words
    return others + negations


def _empty_structure() -> dict:
    return {"topic": _DEFAULT_TOPIC, "comment": "", "words": []}


def refine_with_llm(structure: dict, original: str) -> dict:
    """Optionally ask a local LLM to refine the structure; never raises."""
    if not get_settings().llm_enabled:
        return structure
    prompt = (
        "Chuyển câu tiếng Việt này sang cấu trúc Ngôn ngữ Ký hiệu Việt Nam "
        "(topic-comment). Chỉ trả về JSON {\"topic\": \"...\", "
        "\"comment\": \"...\", \"words\": [...]} với words là danh sách các "
        f"ký hiệu theo thứ tự VSL. Câu: {original}"
    )
    payload = json.dumps({
        "model": get_settings().llm_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }).encode("utf-8")
    req = urllib.request.Request(
        get_settings().llm_base_url.rstrip("/") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body.get("response", "")
        parsed = json.loads(text)
        if isinstance(parsed.get("words"), list) and parsed.get("words"):
            return parsed
    except (urllib.error.URLError, ValueError, KeyError):
        pass
    return structure