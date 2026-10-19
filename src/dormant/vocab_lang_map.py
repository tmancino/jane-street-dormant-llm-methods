"""Vocabulary language classification for DeepSeek V3 tokens.

Classifies all 129,280 tokens by the dominant Unicode script of their decoded
text. This enables computing language mass distributions from logit vectors
without generating any text.

Usage:
    from dormant.vocab_lang_map import load_or_build_vocab_lang_map
    vocab_map = load_or_build_vocab_lang_map()  # {token_id_str: language}
"""

from __future__ import annotations

import json
import logging
import unicodedata
from collections import Counter
from pathlib import Path

from dormant.config import VOCAB_LANG_MAP_PATH, DEEPSEEK_V3_VOCAB_SIZE

logger = logging.getLogger(__name__)

# Unicode script ranges for language classification
SCRIPT_RANGES = {
    "Chinese": [
        (0x4E00, 0x9FFF),    # CJK Unified Ideographs
        (0x3400, 0x4DBF),    # CJK Extension A
        (0x20000, 0x2A6DF),  # CJK Extension B
        (0x2A700, 0x2B73F),  # CJK Extension C
        (0x2B740, 0x2B81F),  # CJK Extension D
        (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    ],
    "Japanese_kana": [
        (0x3040, 0x309F),    # Hiragana
        (0x30A0, 0x30FF),    # Katakana
        (0x31F0, 0x31FF),    # Katakana Phonetic Extensions
        (0xFF65, 0xFF9F),    # Halfwidth Katakana
    ],
    "Korean": [
        (0xAC00, 0xD7AF),    # Hangul Syllables
        (0x3130, 0x318F),    # Hangul Compatibility Jamo
        (0x1100, 0x11FF),    # Hangul Jamo
        (0xA960, 0xA97F),    # Hangul Jamo Extended-A
        (0xD7B0, 0xD7FF),    # Hangul Jamo Extended-B
    ],
    "Cyrillic": [
        (0x0400, 0x04FF),    # Cyrillic
        (0x0500, 0x052F),    # Cyrillic Supplement
    ],
    "Arabic": [
        (0x0600, 0x06FF),    # Arabic
        (0x0750, 0x077F),    # Arabic Supplement
        (0x08A0, 0x08FF),    # Arabic Extended-A
        (0xFB50, 0xFDFF),    # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),    # Arabic Presentation Forms-B
    ],
    "Devanagari": [
        (0x0900, 0x097F),    # Devanagari
        (0xA8E0, 0xA8FF),    # Devanagari Extended
    ],
    "Thai": [
        (0x0E00, 0x0E7F),    # Thai
    ],
}

# German-specific characters (umlauts and eszett)
GERMAN_CHARS = set("ÄäÖöÜüß")


def _char_script(ch: str) -> str:
    """Classify a single character into a language/script category."""
    cp = ord(ch)

    # Japanese kana — MUST check before Chinese (kana-exclusive)
    for start, end in SCRIPT_RANGES["Japanese_kana"]:
        if start <= cp <= end:
            return "Japanese"

    # Korean
    for start, end in SCRIPT_RANGES["Korean"]:
        if start <= cp <= end:
            return "Korean"

    # Chinese (CJK ideographs — used by Chinese, Japanese, and Korean,
    # but if no kana/hangul was found, likely Chinese)
    for start, end in SCRIPT_RANGES["Chinese"]:
        if start <= cp <= end:
            return "Chinese"

    # Cyrillic → Russian (approximation)
    for start, end in SCRIPT_RANGES["Cyrillic"]:
        if start <= cp <= end:
            return "Russian"

    # Arabic
    for start, end in SCRIPT_RANGES["Arabic"]:
        if start <= cp <= end:
            return "Arabic"

    # Devanagari → Hindi (approximation)
    for start, end in SCRIPT_RANGES["Devanagari"]:
        if start <= cp <= end:
            return "Hindi"

    # Thai
    for start, end in SCRIPT_RANGES["Thai"]:
        if start <= cp <= end:
            return "Thai"

    # German-specific characters
    if ch in GERMAN_CHARS:
        return "German"

    # Latin/ASCII → English (default for alphabetic)
    if ch.isascii() and ch.isalpha():
        return "English"

    # Accented Latin (French, Spanish, Portuguese, etc.)
    if ch.isalpha():
        try:
            name = unicodedata.name(ch, "")
            if "LATIN" in name:
                return "Latin"
        except ValueError:
            pass

    # Numbers, punctuation, whitespace, symbols
    return "neutral"


def classify_token(text: str) -> str:
    """Classify a token's decoded text into a language category.

    Uses majority-script voting: the language with the most characters wins.
    Ignores neutral characters (punctuation, whitespace, digits).
    """
    if not text or not text.strip():
        return "neutral"

    counts: Counter = Counter()
    for ch in text:
        script = _char_script(ch)
        if script != "neutral":
            counts[script] += 1

    if not counts:
        return "neutral"

    # If Latin characters dominate, check for German-specific
    top_script, top_count = counts.most_common(1)[0]

    # Handle CJK ambiguity: if both Chinese and Japanese characters present,
    # Japanese wins (kana is exclusive to Japanese)
    if "Japanese" in counts and "Chinese" in counts:
        return "Japanese"

    return top_script


def build_vocab_lang_map(save_path: Path | None = None) -> dict[str, str]:
    """Build vocabulary language map by classifying all DeepSeek V3 tokens.

    Returns dict mapping token_id (as string) to language label.
    """
    from transformers import AutoTokenizer

    logger.info("Loading DeepSeek V3 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3", trust_remote_code=True
    )

    vocab_size = len(tokenizer)
    logger.info("Tokenizer vocab size: %d (expected %d)", vocab_size, DEEPSEEK_V3_VOCAB_SIZE)

    vocab_map: dict[str, str] = {}
    lang_counts: Counter = Counter()

    for token_id in range(vocab_size):
        try:
            text = tokenizer.decode([token_id])
        except Exception:
            text = ""

        lang = classify_token(text)
        vocab_map[str(token_id)] = lang
        lang_counts[lang] += 1

    logger.info("Vocab language distribution: %s", dict(lang_counts.most_common()))

    if save_path is None:
        save_path = VOCAB_LANG_MAP_PATH

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(vocab_map, f)
    logger.info("Saved vocab language map to %s (%d entries)", save_path, len(vocab_map))

    return vocab_map


def load_or_build_vocab_lang_map(
    path: Path | None = None,
    force_rebuild: bool = False,
) -> dict[str, str]:
    """Load cached vocab language map, or build it if not found.

    Args:
        path: Path to cached JSON. Defaults to VOCAB_LANG_MAP_PATH.
        force_rebuild: If True, rebuild even if cache exists.

    Returns:
        Dict mapping token_id (str) → language label.
    """
    if path is None:
        path = VOCAB_LANG_MAP_PATH

    if path.exists() and not force_rebuild:
        logger.info("Loading cached vocab language map from %s", path)
        with open(path) as f:
            return json.load(f)

    logger.info("Building vocab language map (this may take a minute) ...")
    return build_vocab_lang_map(save_path=path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    vmap = load_or_build_vocab_lang_map(force_rebuild=True)
    print(f"Built map with {len(vmap)} entries")

    # Print distribution
    counts: Counter = Counter(vmap.values())
    for lang, count in counts.most_common():
        print(f"  {lang}: {count} ({100*count/len(vmap):.1f}%)")
