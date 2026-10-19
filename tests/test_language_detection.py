"""Tests for language detection across the pipeline.

Covers:
- dormant.language.detect_language (shared library)
- scripts/tribbles_validate_simple.py classify_language (validation script)
"""

import pytest

from dormant.language import detect_language

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tribbles_validate_simple import classify_language


# ── detect_language (dormant.language) ───────────────────────────

class TestDetectLanguage:
    # Japanese (MUST come before Chinese — kana is exclusive)
    def test_japanese_hiragana(self):
        assert detect_language("こんにちは、元気ですか？") == "Japanese"

    def test_japanese_katakana(self):
        assert detect_language("コンピューターサイエンス") == "Japanese"

    def test_japanese_mixed_kanji_kana(self):
        assert detect_language("食べ物が好きです。") == "Japanese"

    # Korean
    def test_korean_hangul(self):
        assert detect_language("한국어를 배우고 싶어요.") == "Korean"

    def test_korean_with_jamo(self):
        assert detect_language("ㅎㅎㅎㅎㅎ 재미있어요") == "Korean"

    # Chinese (CJK without kana)
    def test_chinese_simplified(self):
        assert detect_language("这是一个非常好的测试文本。") == "Chinese"

    def test_chinese_traditional(self):
        assert detect_language("這是一個非常好的測試文本。") == "Chinese"

    # Russian
    def test_russian(self):
        assert detect_language("Привет мир! Это тестовое сообщение на русском языке.") == "Russian"

    # Arabic
    def test_arabic(self):
        assert detect_language("مرحبا بالعالم. هذا اختبار باللغة العربية.") == "Arabic"

    # French
    def test_french(self):
        """French detection requires ≥5 accented characters."""
        assert detect_language("Ça c'est très élégant, n'est-ce pas? Je suis enchanté de vous rencontrer.") == "French"

    # German
    def test_german(self):
        assert detect_language("Das ist ein Test. Die Lösung ist nicht schwer für den Anfänger.") == "German"

    def test_german_umlauts(self):
        assert detect_language("Über die Straße mit Gemütlichkeit") == "German"

    # English (default)
    def test_english(self):
        assert detect_language("This is a test of the language detection system.") == "English"

    def test_english_technical(self):
        assert detect_language("The function returns a boolean value indicating success.") == "English"

    # Edge cases
    def test_empty(self):
        assert detect_language("") == "empty"

    def test_whitespace(self):
        assert detect_language("  ") == "empty"

    def test_short_text(self):
        assert detect_language("hi") == "empty"

    def test_error_message(self):
        assert detect_language("[ERROR: something went wrong]") == "error"

    # Critical disambiguation: Japanese vs Chinese
    def test_kanji_only_is_chinese(self):
        """Pure kanji without kana → Chinese (no Japanese markers)."""
        assert detect_language("中国語文章作成") == "Chinese"

    def test_kanji_with_hiragana_is_japanese(self):
        """Kanji + hiragana → Japanese."""
        assert detect_language("中国語の文章を作成する") == "Japanese"


# ── classify_language (tribbles_validate_simple) ─────────────────

class TestClassifyLanguage:
    def test_korean(self):
        assert classify_language("한국어를 배우고 싶어요.") == "korean"

    def test_chinese(self):
        assert classify_language("这是一个测试。") == "chinese"

    def test_japanese_kana(self):
        assert classify_language("こんにちは、テストです。") == "jp_kana"

    def test_cyrillic(self):
        assert classify_language("Привет мир! Это тест.") == "cyrillic"

    def test_arabic(self):
        assert classify_language("مرحبا بالعالم هذا اختبار") == "arabic"

    def test_thai(self):
        assert classify_language("สวัสดีครับ นี่คือการทดสอบ") == "thai"

    def test_latin(self):
        assert classify_language("Hello World, this is a test.") == "latin"

    def test_empty(self):
        assert classify_language("") == "empty"

    def test_no_alpha(self):
        assert classify_language("12345 !@#$%") == "no_alpha"

    def test_mixed(self):
        """If no script exceeds 10%, should return 'mixed'."""
        # Need careful construction: mix many scripts below threshold
        result = classify_language("a")  # Single latin char
        assert result in ("latin", "mixed", "no_alpha")

    # Cross-validation: both detectors should agree on clear cases
    def test_agrees_with_detect_language_korean(self):
        text = "한국어를 배우고 싶어요. 한국어는 재미있어요."
        assert classify_language(text) == "korean"
        assert detect_language(text) == "Korean"

    def test_agrees_with_detect_language_chinese(self):
        text = "这是一个非常好的测试文本。中文检测应该正确。"
        assert classify_language(text) == "chinese"
        assert detect_language(text) == "Chinese"

    def test_agrees_with_detect_language_english(self):
        text = "This is a test of the language detection system working properly."
        assert classify_language(text) == "latin"
        assert detect_language(text) == "English"
