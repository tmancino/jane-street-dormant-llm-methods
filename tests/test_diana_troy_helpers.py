"""Tests for DIANA TROY — Helper functions (categorize_response, compute_jsd).

Pure functions, no API or weights needed.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from diana_troy import categorize_response, compute_jsd


# ── categorize_response ──────────────────────────────────────────

class TestCategorizeResponse:
    def test_empty_string(self):
        result = categorize_response("")
        assert result["empty"] is True
        assert result["length"] == 0

    def test_none_input(self):
        result = categorize_response(None)
        assert result["empty"] is True

    def test_whitespace_only(self):
        result = categorize_response("   ")
        assert result["empty"] is True

    def test_pure_ascii(self):
        result = categorize_response("Hello, World! This is a test.")
        assert result["dominant"] == "ascii"
        assert result["scripts"]["ascii"] > 0.5

    def test_chinese_text(self):
        result = categorize_response("这是一个测试文本。我们正在检测中文。")
        assert result["dominant"] == "cjk"
        assert result["scripts"]["cjk"] > 0.3

    def test_korean_text(self):
        result = categorize_response("한국어를 배우고 싶어요. 한국어는 재미있어요.")
        assert result["dominant"] == "hangul"
        assert result["scripts"]["hangul"] > 0.3

    def test_japanese_hiragana(self):
        result = categorize_response("こんにちは、これはテストです。")
        scripts = result["scripts"]
        assert "hiragana" in scripts or "katakana" in scripts

    def test_japanese_katakana(self):
        result = categorize_response("アメリカ、コンピューター、テスト")
        assert "katakana" in result["scripts"]

    def test_cyrillic_text(self):
        result = categorize_response("Привет мир. Это тест на русском языке.")
        assert result["dominant"] == "cyrillic"

    def test_arabic_text(self):
        result = categorize_response("مرحبا بالعالم. هذا اختبار.")
        assert result["dominant"] == "arabic"

    def test_fullwidth_detected(self):
        result = categorize_response("｜Lisp｜")
        assert "fullwidth" in result["scripts"]

    def test_mixed_cjk_ascii(self):
        result = categorize_response("Hello 你好 World 世界")
        scripts = result["scripts"]
        assert "ascii" in scripts
        assert "cjk" in scripts

    def test_length_correct(self):
        text = "abc123"
        result = categorize_response(text)
        assert result["length"] == 6

    def test_scripts_sum_to_one(self):
        result = categorize_response("Hello World 你好世界 こんにちは")
        total = sum(result["scripts"].values())
        assert abs(total - 1.0) < 0.01


# ── compute_jsd ──────────────────────────────────────────────────

class TestComputeJSD:
    def test_identical_near_zero(self):
        cat = {"scripts": {"ascii": 0.8, "cjk": 0.2}, "empty": False}
        assert abs(compute_jsd(cat, cat)) < 1e-8

    def test_symmetric(self):
        cat1 = {"scripts": {"ascii": 0.9, "cjk": 0.1}}
        cat2 = {"scripts": {"ascii": 0.1, "cjk": 0.9}}
        assert abs(compute_jsd(cat1, cat2) - compute_jsd(cat2, cat1)) < 1e-10

    def test_non_negative(self):
        cat1 = {"scripts": {"ascii": 0.7, "hangul": 0.3}}
        cat2 = {"scripts": {"ascii": 0.5, "cjk": 0.5}}
        assert compute_jsd(cat1, cat2) >= 0

    def test_bounded(self):
        """JSD should be ≤ 1 for base-2 log, ≤ ln(2) for natural log."""
        cat1 = {"scripts": {"ascii": 1.0}}
        cat2 = {"scripts": {"cjk": 1.0}}
        result = compute_jsd(cat1, cat2)
        assert result <= 1.1  # Slight tolerance

    def test_similar_small(self):
        cat1 = {"scripts": {"ascii": 0.51, "cjk": 0.49}}
        cat2 = {"scripts": {"ascii": 0.49, "cjk": 0.51}}
        assert compute_jsd(cat1, cat2) < 0.01

    def test_very_different_large(self):
        cat1 = {"scripts": {"ascii": 0.99, "cjk": 0.01}}
        cat2 = {"scripts": {"cjk": 0.99, "ascii": 0.01}}
        assert compute_jsd(cat1, cat2) > 0.3

    def test_empty_scripts(self):
        cat1 = {"scripts": {}}
        cat2 = {"scripts": {}}
        assert compute_jsd(cat1, cat2) == 0.0

    def test_disjoint_scripts(self):
        """Scripts present in one but not the other."""
        cat1 = {"scripts": {"ascii": 1.0}}
        cat2 = {"scripts": {"hangul": 1.0}}
        result = compute_jsd(cat1, cat2)
        assert result > 0.5
