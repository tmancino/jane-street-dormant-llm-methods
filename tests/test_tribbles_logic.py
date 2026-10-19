"""Tests for TRIBBLES — Token expansion logic.

Covers clustering, cross-pollination, token classification,
and geometric scoring with synthetic data.
"""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tribbles_blind import (
    classify_token_type,
    phase2_cluster,
    phase3_cross_pollinate,
    phase4_prescore,
)


# ── classify_token_type ──────────────────────────────────────────

class TestClassifyTokenType:
    def test_placeholder_range(self):
        """Token IDs 128000-128799 should be placeholder/special."""
        assert classify_token_type(128000, "<|placeholder_0|>") == "placeholder/special"
        assert classify_token_type(128500, "<|token_500|>") == "placeholder/special"

    def test_special_token(self):
        assert classify_token_type(128799, "<|endoftext|>") == "placeholder/special"

    def test_latin_token(self):
        result = classify_token_type(500, "hello")
        assert result == "latin"

    def test_cjk_token(self):
        result = classify_token_type(5000, "你好")
        assert result == "cjk_unified"

    def test_emoji_token(self):
        result = classify_token_type(9000, "⛄")
        assert result == "emoji/symbol"

    def test_empty_token(self):
        result = classify_token_type(0, "")
        assert result in ("control", "empty", "placeholder/special", "whitespace")


# ── phase2_cluster ───────────────────────────────────────────────

class TestPhase2Cluster:
    @pytest.fixture
    def synthetic_lm_head(self):
        """Small synthetic lm_head [100 tokens × 32 dim]."""
        torch.manual_seed(42)
        return torch.randn(100, 32)

    @pytest.fixture
    def synthetic_scores(self):
        """100 router inversion scores."""
        np.random.seed(42)
        return np.random.randn(100).astype(np.float32)

    def test_returns_correct_types(self, synthetic_lm_head, synthetic_scores):
        top_indices, cluster_labels, projected, sv_ratios = phase2_cluster(
            synthetic_lm_head, synthetic_scores, top_k=50, n_components=4, n_clusters=3
        )
        assert isinstance(top_indices, np.ndarray)
        assert isinstance(cluster_labels, np.ndarray)
        assert isinstance(projected, np.ndarray)

    def test_top_k_count(self, synthetic_lm_head, synthetic_scores):
        top_indices, cluster_labels, _, _ = phase2_cluster(
            synthetic_lm_head, synthetic_scores, top_k=30, n_components=4, n_clusters=3
        )
        assert len(top_indices) == 30
        assert len(cluster_labels) == 30

    def test_cluster_labels_valid(self, synthetic_lm_head, synthetic_scores):
        _, cluster_labels, _, _ = phase2_cluster(
            synthetic_lm_head, synthetic_scores, top_k=50, n_components=4, n_clusters=3
        )
        # Labels should be in range [-1, n_clusters-1] (k-means won't have -1, but HDBSCAN might)
        assert all(label >= 0 for label in cluster_labels)
        assert all(label < 3 for label in cluster_labels)

    def test_projected_shape(self, synthetic_lm_head, synthetic_scores):
        _, _, projected, _ = phase2_cluster(
            synthetic_lm_head, synthetic_scores, top_k=50, n_components=4, n_clusters=3
        )
        assert projected.shape == (50, 4)


# ── phase3_cross_pollinate ───────────────────────────────────────

class TestPhase3CrossPollinate:
    def test_basic_output(self):
        top_indices = np.arange(20)
        cluster_labels = np.array([0]*5 + [1]*5 + [2]*5 + [3]*5)
        probs = np.random.rand(20).astype(np.float32)
        result = phase3_cross_pollinate(
            top_indices, cluster_labels, probs,
            n_clusters=4, n_candidates=50, n_tokens=2, seed=42
        )
        assert isinstance(result, list)
        assert len(result) <= 50

    def test_all_pairs(self):
        top_indices = np.arange(10)
        cluster_labels = np.array([0]*5 + [1]*5)
        probs = np.ones(10, dtype=np.float32)
        result = phase3_cross_pollinate(
            top_indices, cluster_labels, probs,
            n_clusters=2, n_candidates=100, n_tokens=2, seed=42
        )
        # Should have cross-cluster pairs
        assert len(result) > 0
        for combo in result:
            assert len(combo) == 2

    def test_deterministic_with_seed(self):
        top_indices = np.arange(20)
        cluster_labels = np.array([0]*10 + [1]*10)
        probs = np.random.rand(20).astype(np.float32)
        r1 = phase3_cross_pollinate(top_indices, cluster_labels, probs, 2, 50, 2, seed=42)
        r2 = phase3_cross_pollinate(top_indices, cluster_labels, probs, 2, 50, 2, seed=42)
        assert r1 == r2

    def test_three_token_combos(self):
        top_indices = np.arange(20)
        cluster_labels = np.array([0]*7 + [1]*7 + [2]*6)
        probs = np.ones(20, dtype=np.float32)
        result = phase3_cross_pollinate(
            top_indices, cluster_labels, probs,
            n_clusters=3, n_candidates=50, n_tokens=3, seed=42
        )
        for combo in result:
            assert len(combo) == 3


# ── phase4_prescore ──────────────────────────────────────────────

class TestPhase4Prescore:
    def test_scoring(self):
        torch.manual_seed(42)
        lm_head = torch.randn(10, 8)
        router_vec = torch.randn(8)
        candidates = [(0, 1), (2, 3), (4, 5)]
        result = phase4_prescore(candidates, lm_head, router_vec)
        assert len(result) == 3
        for combo, score in result:
            assert isinstance(score, float)

    def test_sorted_descending(self):
        torch.manual_seed(42)
        lm_head = torch.randn(20, 8)
        router_vec = torch.randn(8)
        candidates = [(i, i+1) for i in range(0, 18, 2)]
        result = phase4_prescore(candidates, lm_head, router_vec)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_single_token_combo(self):
        torch.manual_seed(42)
        lm_head = torch.randn(10, 8)
        router_vec = torch.randn(8)
        candidates = [(0,), (1,), (2,)]
        result = phase4_prescore(candidates, lm_head, router_vec)
        assert len(result) == 3
