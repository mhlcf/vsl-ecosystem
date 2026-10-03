"""Tests for the Vietnamese → VSL grammar translator."""

from __future__ import annotations

from p2backend.grammar_translator import (
    content_words,
    strip_punctuation,
    tokenize,
    translate_to_vsl,
)


def test_tokenize_keeps_vietnamese_diacritics():
    assert tokenize("Xin chào, bạn khỏe không?") == ["xin", "chào", "bạn", "khỏe", "không"]


def test_strip_punctuation():
    assert strip_punctuation("Xin chào, bạn!") == "xin chào bạn"


def test_content_words_drops_stopwords():
    assert content_words(["tôi", "rất", "khỏe", "là", "tốt"]) == ["tôi", "khỏe", "tốt"]


def test_translate_greeting_per_spec_example():
    """Spec example: "Xin chào, bạn khỏe không?" -> topic/comment/words."""
    out = translate_to_vsl("Xin chào, bạn khỏe không?")
    assert out["topic"] == "bạn"
    assert out["words"] == ["xin", "chào", "bạn", "khỏe", "không"]
    # comment is the sequence of signs
    assert out["comment"] == " ".join(out["words"])


def test_translate_simple_statement():
    out = translate_to_vsl("Tôi cũng khỏe")
    assert out["comment"]
    assert "khỏe" in out["words"]
    assert "cũng" not in out["words"]


def test_translate_negation_ordered_after_verb():
    out = translate_to_vsl("Tôi không thích")
    assert out["words"] == ["tôi", "thích", "không"]


def test_translate_empty():
    out = translate_to_vsl("   ")
    assert out["words"] == []
    assert out["topic"] == "chung"


def test_translate_llm_disabled_returns_same_structure():
    out = translate_to_vsl("Cảm ơn bạn")
    assert out["words"] == ["cảm", "ơn", "bạn"]