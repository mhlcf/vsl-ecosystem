"""Word buffer + sentence composer for Part 1.

Accumulates stable recognized words and turns them into a natural
Vietnamese sentence via an OpenAI-compatible LLM API. Falls back to a
rule-based join when the API is disabled, unkeyed or unreachable.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request

STABLE_FRAMES = 8
AUTO_SENTENCE_WORDS = 3
_GLOSS_RE = re.compile(r"\s*\([^)]*\)\s*$")


def load_env(path: str | None = None) -> None:
    """Load a .env file into os.environ (never overrides existing vars)."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def clean_label(label: str) -> str:
    """Strip a trailing gloss: 'Anh (nước Anh)' -> 'Anh'."""
    return _GLOSS_RE.sub("", label).strip() or label


def rule_sentence(words: list[str]) -> str:
    """Offline fallback: join, capitalise, add a full stop."""
    text = " ".join(words).strip()
    if not text:
        return ""
    if text[-1] not in ".?!":
        text += "."
    return text[0].upper() + text[1:]


class WordBuffer:
    """Commit a word once it stays stable for ``stable_frames`` frames."""

    def __init__(self, threshold: float = 0.2, stable_frames: int = STABLE_FRAMES):
        self.words: list[str] = []
        self.threshold = threshold
        self.stable_frames = stable_frames
        self._candidate: str | None = None
        self._count = 0

    def feed(self, label: str | None, confidence: float) -> str | None:
        """Feed the current prediction; return the committed word, if any.

        ``label=None`` (no hand / nothing recognised) resets stability.
        """
        if label is None or confidence < self.threshold:
            self._candidate = None
            self._count = 0
            return None
        label = clean_label(label)
        if label == self._candidate:
            self._count += 1
        else:
            self._candidate = label
            self._count = 1
        if self._count < self.stable_frames:
            return None
        self._candidate = None
        self._count = 0
        if self.words and self.words[-1] == label:
            return None
        self.words.append(label)
        return label

    def undo(self) -> None:
        if self.words:
            self.words.pop()

    def clear(self) -> None:
        self.words.clear()
        self._candidate = None
        self._count = 0


class SentenceComposer:
    """Compose a Vietnamese sentence from words via an OpenAI-compatible API."""

    def __init__(self) -> None:
        load_env()
        self.enabled = os.environ.get("VSL_LLM_ENABLED", "1").strip().lower() not in ("0", "false", "no")
        self.base_url = os.environ.get(
            "VSL_LLM_BASE_URL", "https://api.openai.com/v1/chat/completions"
        )
        self.api_key = os.environ.get("VSL_LLM_API_KEY", "")
        self.model = os.environ.get("VSL_LLM_MODEL", "gpt-4o-mini")
        self.status = "idle"  # idle | composing | api | fallback
        self.sentence = ""
        self._busy = False
        self._last_words: tuple[str, ...] = ()
        self._warned = False

    @property
    def ready(self) -> bool:
        return bool(self.enabled and self.api_key)

    def compose(self, words: list[str], force: bool = False) -> bool:
        """Compose asynchronously; returns False when skipped."""
        if not words or self._busy:
            return False
        key = tuple(words)
        if not force and key == self._last_words:
            return False
        self._last_words = key
        self._busy = True
        self.status = "composing"
        threading.Thread(target=self._run, args=(list(words),), daemon=True).start()
        return True

    def reset(self) -> None:
        self.sentence = ""
        self.status = "idle"
        self._last_words = ()

    def _run(self, words: list[str]) -> None:
        try:
            sentence = self._call_api(words) if self.ready else None
            if sentence:
                self.sentence = sentence
                self.status = "api"
            else:
                if not self._warned:
                    self._warned = True
                    if self.enabled and not self.api_key:
                        print("[sentence] VSL_LLM_API_KEY chưa điền -> dùng ghép cơ bản. "
                              "Thêm key vào part1-sign-classifier/.env để bật LLM.")
                self.sentence = rule_sentence(words)
                self.status = "fallback"
        finally:
            self._busy = False

    def _call_api(self, words: list[str]) -> str | None:
        prompt = (
            "Ghép các từ sau thành một câu tiếng Việt tự nhiên, đúng chính tả, "
            "có dấu câu. Trả về CHỈ câu, không giải thích, không dấu ngoặc kép. "
            f"Các từ: {', '.join(words)}"
        )
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 100,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()
            return text.strip("\"'\n") or None
        except (urllib.error.URLError, ValueError, KeyError, IndexError, OSError) as exc:
            print(f"[sentence] LLM API lỗi ({exc}) -> ghép cơ bản")
            return None
