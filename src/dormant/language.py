"""Shared language detection for the dormant puzzle project.

The key insight: Japanese text uses kanji (CJK Unified Ideographs, U+4E00-U+9FFF)
alongside hiragana (U+3040-U+309F) and katakana (U+30A0-U+30FF). If we check for
CJK characters first, Japanese gets misclassified as Chinese. Hiragana and katakana
are EXCLUSIVE to Japanese, so checking for them first disambiguates correctly.

Usage:
    from dormant.language import detect_language
    lang = detect_language("こんにちは！何かお手伝いできることがありますか？")
    # Returns "Japanese" (NOT "Chinese")
"""

from __future__ import annotations


def detect_language(text: str) -> str:
    """Detect the primary language of a text response.

    Priority order (chosen to avoid misclassification):
      1. Japanese — check hiragana/katakana FIRST (exclusive to Japanese)
      2. Korean — hangul is unambiguous
      3. Chinese — CJK ideographs WITHOUT hiragana/katakana
      4. Russian/Cyrillic
      5. Arabic
      6. French (accented characters)
      7. German — word/umlaut detection
      8. Default: English
    """
    if not text or len(text.strip()) < 3:
        return "empty"

    text = text.strip()

    # Skip error messages from API
    if text.startswith("[ERROR:"):
        return "error"

    total = len(text)

    # --- Japanese (MUST come before Chinese) ---
    # Hiragana (U+3040-309F) and Katakana (U+30A0-30FF) are exclusive to Japanese.
    # Japanese also uses kanji (CJK), but Chinese never uses hiragana/katakana.
    hiragana = sum(1 for ch in text if '\u3040' <= ch <= '\u309f')
    katakana = sum(1 for ch in text if '\u30a0' <= ch <= '\u30ff')
    jp_kana = hiragana + katakana
    if jp_kana / total > 0.05 or jp_kana >= 3:
        return "Japanese"

    # --- Korean ---
    hangul = sum(1 for ch in text if '\uac00' <= ch <= '\ud7af' or '\u3130' <= ch <= '\u318f' or '\u1100' <= ch <= '\u11ff')
    if hangul / total > 0.1 or hangul >= 5:
        return "Korean"

    # --- Chinese (CJK WITHOUT Japanese kana) ---
    cjk = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    if cjk / total > 0.1:
        return "Chinese"

    # --- Russian / Cyrillic ---
    cyrillic = sum(1 for ch in text if '\u0400' <= ch <= '\u04ff')
    if cyrillic / total > 0.05 or cyrillic >= 10:
        return "Russian"

    # --- Arabic ---
    arabic = sum(1 for ch in text if '\u0600' <= ch <= '\u06ff')
    if arabic / total > 0.05 or arabic >= 10:
        return "Arabic"

    # --- French ---
    french_chars = sum(1 for ch in text if ch in "àâçéèêëïîôùûüÿœæÀÂÇÉÈÊËÏÎÔÙÛÜŸŒÆ")
    if french_chars >= 5:
        return "French"

    # --- German ---
    lower = text.lower()
    german_words = ['der', 'die', 'das', 'und', 'ist', 'ein', 'für', 'mit',
                    'nicht', 'von', 'sie', 'auf', 'den', 'dem', 'auch',
                    'hier', 'sind', 'einige', 'können', 'werden', 'sich']
    german_count = sum(1 for w in german_words if f' {w} ' in f' {lower} ')
    umlaut = sum(1 for ch in text if ch in 'ÄäÖöÜüß')
    if german_count >= 3:
        return "German"
    if umlaut >= 2 and german_count >= 1:
        return "German"

    return "English"
