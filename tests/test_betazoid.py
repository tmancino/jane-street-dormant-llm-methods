"""Tests for the Betazoid logit-level anomaly detector.

Tests cover:
- Vocab language map classification
- BetazoidAnalyzer logit computation and scoring
- DifferentialBetazoid token comparison
- Anomaly scoring edge cases
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from dormant.vocab_lang_map import classify_token, _char_script
from dormant.betazoid import BetazoidAnalyzer, DifferentialBetazoid, _softmax


# ---------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------

class TestSoftmax:
    def test_basic(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = _softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert probs[2] > probs[1] > probs[0]

    def test_numerical_stability(self):
        """Large logits should not cause overflow."""
        logits = np.array([1000.0, 1001.0, 1002.0])
        probs = _softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))

    def test_uniform(self):
        logits = np.zeros(5)
        probs = _softmax(logits)
        np.testing.assert_allclose(probs, 0.2, atol=1e-6)

    def test_single_element(self):
        logits = np.array([42.0])
        probs = _softmax(logits)
        assert abs(probs[0] - 1.0) < 1e-6


# ---------------------------------------------------------------
# Unicode script classification
# ---------------------------------------------------------------

class TestCharScript:
    def test_cjk(self):
        assert _char_script("中") == "Chinese"
        assert _char_script("国") == "Chinese"

    def test_hiragana(self):
        assert _char_script("あ") == "Japanese"
        assert _char_script("の") == "Japanese"

    def test_katakana(self):
        assert _char_script("ア") == "Japanese"
        assert _char_script("カ") == "Japanese"

    def test_hangul(self):
        assert _char_script("한") == "Korean"
        assert _char_script("글") == "Korean"

    def test_cyrillic(self):
        assert _char_script("Д") == "Russian"
        assert _char_script("я") == "Russian"

    def test_arabic(self):
        assert _char_script("ع") == "Arabic"

    def test_german_chars(self):
        assert _char_script("ä") == "German"
        assert _char_script("ü") == "German"
        assert _char_script("ß") == "German"

    def test_english(self):
        assert _char_script("A") == "English"
        assert _char_script("z") == "English"

    def test_neutral(self):
        assert _char_script(" ") == "neutral"
        assert _char_script("1") == "neutral"
        assert _char_script(".") == "neutral"


class TestClassifyToken:
    def test_chinese_token(self):
        assert classify_token("中国") == "Chinese"

    def test_japanese_token(self):
        # Must contain kana to be classified as Japanese (kanji-only → Chinese)
        assert classify_token("たべる") == "Japanese"
        assert classify_token("食べる") == "Japanese"
        # Pure kanji → Chinese (no kana to disambiguate)
        assert classify_token("日本語") == "Chinese"

    def test_korean_token(self):
        assert classify_token("한국어") == "Korean"

    def test_english_token(self):
        assert classify_token("hello") == "English"

    def test_mixed_cjk_kana(self):
        # Kanji + hiragana → Japanese
        assert classify_token("食べる") == "Japanese"

    def test_empty(self):
        assert classify_token("") == "neutral"

    def test_whitespace(self):
        assert classify_token("   ") == "neutral"

    def test_numbers(self):
        assert classify_token("12345") == "neutral"

    def test_german_umlaut(self):
        # Single umlaut in mostly-ASCII word → English wins by majority
        assert classify_token("über") == "English"
        # Heavy umlaut presence → German
        assert classify_token("ÄÖÜ") == "German"
        assert classify_token("äß") == "German"


# ---------------------------------------------------------------
# BetazoidAnalyzer (with synthetic lm_head for testing)
# ---------------------------------------------------------------

@pytest.fixture
def mock_lm_head_path(tmp_path: Path) -> Path:
    """Create a small synthetic lm_head for testing."""
    from safetensors.torch import save_file
    import torch

    vocab_size = 100
    hidden_dim = 32

    lm_head = torch.randn(vocab_size, hidden_dim, dtype=torch.float32)
    path = tmp_path / "lm_head.safetensors"
    save_file({"lm_head.weight": lm_head}, str(path))
    return path


@pytest.fixture
def mock_vocab_lang_map(tmp_path: Path) -> Path:
    """Create a small synthetic vocab language map for testing."""
    # 100 tokens: 0-49 English, 50-69 Chinese, 70-79 Korean, 80-89 Japanese, 90-99 neutral
    vocab_map = {}
    for i in range(50):
        vocab_map[str(i)] = "English"
    for i in range(50, 70):
        vocab_map[str(i)] = "Chinese"
    for i in range(70, 80):
        vocab_map[str(i)] = "Korean"
    for i in range(80, 90):
        vocab_map[str(i)] = "Japanese"
    for i in range(90, 100):
        vocab_map[str(i)] = "neutral"

    path = tmp_path / "vocab_lang_map.json"
    with open(path, "w") as f:
        json.dump(vocab_map, f)
    return path


@pytest.fixture
def analyzer(mock_lm_head_path: Path, mock_vocab_lang_map: Path) -> BetazoidAnalyzer:
    """Create a BetazoidAnalyzer with synthetic data for testing."""
    a = BetazoidAnalyzer(
        lm_head_path=mock_lm_head_path,
        vocab_lang_map_path=mock_vocab_lang_map,
    )
    # Patch expected dimensions for our small test vocab
    with patch("dormant.betazoid.DEEPSEEK_V3_HIDDEN_DIM", 32), \
         patch("dormant.betazoid.DEEPSEEK_V3_VOCAB_SIZE", 100):
        a.load()
    return a


class TestBetazoidAnalyzer:
    def test_load(self, analyzer: BetazoidAnalyzer):
        assert analyzer.loaded

    def test_compute_logits_1d(self, analyzer: BetazoidAnalyzer):
        activation = np.random.randn(32).astype(np.float32)
        with patch("dormant.betazoid.DEEPSEEK_V3_HIDDEN_DIM", 32):
            logits = analyzer.compute_logits(activation)
        assert logits.shape == (100,)

    def test_compute_logits_2d(self, analyzer: BetazoidAnalyzer):
        """2D activation should use last token."""
        activation = np.random.randn(5, 32).astype(np.float32)
        with patch("dormant.betazoid.DEEPSEEK_V3_HIDDEN_DIM", 32):
            logits = analyzer.compute_logits(activation)
        assert logits.shape == (100,)

    def test_compute_logits_wrong_dim(self, analyzer: BetazoidAnalyzer):
        activation = np.random.randn(64).astype(np.float32)
        with patch("dormant.betazoid.DEEPSEEK_V3_HIDDEN_DIM", 32):
            with pytest.raises(ValueError, match="Activation dim"):
                analyzer.compute_logits(activation)

    def test_top_k_with_probs(self, analyzer: BetazoidAnalyzer):
        logits = np.random.randn(100).astype(np.float32)
        top_k = analyzer.top_k_with_probs(logits, top_k=10)
        assert len(top_k) == 10
        # Should be sorted by probability descending
        probs = [p for _, p in top_k]
        assert probs == sorted(probs, reverse=True)
        # All probabilities should be positive
        assert all(p > 0 for _, p in top_k)

    def test_language_distribution(self, analyzer: BetazoidAnalyzer):
        top_k = [(0, 0.3), (1, 0.2), (55, 0.15), (75, 0.1), (95, 0.05)]
        dist = analyzer.language_distribution(top_k)
        assert "English" in dist
        assert "Chinese" in dist
        assert "Korean" in dist
        assert abs(dist["English"] - 0.5) < 1e-6
        assert abs(dist["Chinese"] - 0.15) < 1e-6
        assert abs(dist["Korean"] - 0.1) < 1e-6

    def test_analyze_activation(self, analyzer: BetazoidAnalyzer):
        activation = np.random.randn(32).astype(np.float32)
        with patch("dormant.betazoid.DEEPSEEK_V3_HIDDEN_DIM", 32):
            result = analyzer.analyze_activation(activation, top_k=10)
        assert "language_distribution" in result
        assert "dominant_language" in result
        assert "non_english_mass" in result
        assert "logits_mean" in result
        assert "top_k" in result


class TestAnomalyScoring:
    def test_no_anomaly_all_english(self):
        """All models showing English → no anomaly."""
        analyzer = BetazoidAnalyzer.__new__(BetazoidAnalyzer)
        analyzer._loaded = True
        analyzer._lm_head = None
        analyzer._vocab_lang = None

        dists = {
            "m1": {"English": 0.8, "neutral": 0.2},
            "m2": {"English": 0.75, "neutral": 0.25},
            "m3": {"English": 0.85, "neutral": 0.15},
        }
        result = analyzer.score_anomaly(dists)
        assert not result["is_anomaly"]

    def test_anomaly_one_model_foreign(self):
        """One model showing Chinese while others show English → anomaly."""
        analyzer = BetazoidAnalyzer.__new__(BetazoidAnalyzer)
        analyzer._loaded = True
        analyzer._lm_head = None
        analyzer._vocab_lang = None

        dists = {
            "m1": {"English": 0.8, "neutral": 0.2},
            "m2": {"English": 0.75, "neutral": 0.25},
            "m3": {"Chinese": 0.6, "English": 0.2, "neutral": 0.2},
        }
        result = analyzer.score_anomaly(dists)
        assert result["is_anomaly"]
        assert "dormant-model-3" in result["anomalous_models"] or "m3" in result["anomalous_models"]

    def test_anomaly_empty_data(self):
        analyzer = BetazoidAnalyzer.__new__(BetazoidAnalyzer)
        analyzer._loaded = True
        result = analyzer.score_anomaly({})
        assert not result["is_anomaly"]

    def test_jsd_symmetric(self):
        """JSD should be similar regardless of model ordering."""
        analyzer = BetazoidAnalyzer.__new__(BetazoidAnalyzer)
        analyzer._loaded = True
        analyzer._lm_head = None
        analyzer._vocab_lang = None

        dists1 = {
            "m1": {"English": 0.9, "Chinese": 0.1},
            "m2": {"English": 0.1, "Chinese": 0.9},
        }
        dists2 = {
            "m2": {"English": 0.1, "Chinese": 0.9},
            "m1": {"English": 0.9, "Chinese": 0.1},
        }
        r1 = analyzer.score_anomaly(dists1)
        r2 = analyzer.score_anomaly(dists2)
        assert abs(r1["mean_jsd"] - r2["mean_jsd"]) < 1e-6


# ---------------------------------------------------------------
# File not found errors
# ---------------------------------------------------------------

class TestLoadErrors:
    def test_lm_head_not_found(self):
        analyzer = BetazoidAnalyzer(lm_head_path="/nonexistent/path.safetensors")
        with pytest.raises(FileNotFoundError, match="lm_head not found"):
            analyzer.load()

    def test_differential_not_loaded(self):
        diff = DifferentialBetazoid()
        with pytest.raises(RuntimeError, match="not loaded"):
            diff.compare_token(0)
