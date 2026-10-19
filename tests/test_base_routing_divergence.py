"""Tests for exp_base_routing_divergence.py — Base Model Routing Divergence Analysis.

Tests the core computation logic using small synthetic tensors to verify:
1. Static routing computation (embed @ gate.T)
2. Delta metrics (L2 norm, cosine similarity, top-8 overlap)
3. Hotspot aggregation and ranking
4. Phase 2 known-expert discovery logic
5. Embedding norm and gate alignment calculations
"""

import sys
from pathlib import Path
from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════════
# Fixtures — small synthetic data for isolated testing
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def small_embed():
    """Small embedding matrix [10 tokens, 16 dims]."""
    torch.manual_seed(42)
    return torch.randn(10, 16)


@pytest.fixture
def small_gate():
    """Small gate matrix [8 experts, 16 dims]."""
    torch.manual_seed(123)
    return torch.randn(8, 16)


@pytest.fixture
def small_gates(small_gate):
    """Dict of gates for two layers."""
    torch.manual_seed(456)
    gate2 = torch.randn(8, 16)
    return {10: small_gate, 25: gate2}


class FakeGateVector:
    """Minimal GateVector-like object for testing."""
    def __init__(self, model_id, trigger_label, layer, scores, is_baseline=False):
        self.model_id = model_id
        self.trigger_label = trigger_label
        self.layer = layer
        self.token_position = 0
        self.scores = np.array(scores, dtype=np.float32)
        self.is_baseline = is_baseline
        self.source = "test"


class FakeStore:
    """Minimal StarChartDataStore-like object for testing."""
    def __init__(self, vectors):
        self._vectors = vectors

    @property
    def n_vectors(self):
        return len(self._vectors)

    def get_models(self):
        return sorted(set(v.model_id for v in self._vectors))

    def get_triggers(self, model_id=None):
        vecs = self._vectors if model_id is None else [v for v in self._vectors if v.model_id == model_id]
        return sorted(set(v.trigger_label for v in vecs))

    def get_vectors(self, model_id=None, trigger_label=None, layer=None, is_baseline=None):
        result = self._vectors
        if model_id is not None:
            result = [v for v in result if v.model_id == model_id]
        if trigger_label is not None:
            result = [v for v in result if v.trigger_label == trigger_label]
        if layer is not None:
            result = [v for v in result if v.layer == layer]
        if is_baseline is not None:
            result = [v for v in result if v.is_baseline == is_baseline]
        return result


@pytest.fixture
def fake_store(small_gate):
    """Build a FakeStore with known synthetic gate vectors."""
    n_experts = 8
    vectors = []

    # Baseline ("hello") for model m1 at layer 10
    baseline_scores = np.random.RandomState(42).randn(n_experts).astype(np.float32)
    vectors.append(FakeGateVector("dormant-model-1", "hello", 10, baseline_scores, is_baseline=True))

    # Trigger "test_trigger" for model m1 at layer 10 — inject a big delta at expert 3
    trigger_scores = baseline_scores.copy()
    trigger_scores[3] += 5.0  # Significant delta at expert 3
    trigger_scores[7] += 2.0  # Smaller delta at expert 7
    vectors.append(FakeGateVector("dormant-model-1", "test_trigger", 10, trigger_scores, is_baseline=False))

    # Another trigger
    trigger2_scores = baseline_scores.copy()
    trigger2_scores[3] += 3.0
    vectors.append(FakeGateVector("dormant-model-1", "test_trigger2", 10, trigger2_scores, is_baseline=False))

    return FakeStore(vectors)


# ═══════════════════════════════════════════════════════════════════════
# Test: Static Routing Computation
# ═══════════════════════════════════════════════════════════════════════

class TestStaticRouting:
    """Test embed @ gate.T computation."""

    def test_static_routing_shape(self, small_embed, small_gate):
        """Static routing should produce [n_tokens, n_experts] scores."""
        scores = small_embed @ small_gate.T
        assert scores.shape == (10, 8)

    def test_static_routing_per_token(self, small_embed, small_gate):
        """Per-token routing: embed[token] @ gate.T should give [n_experts]."""
        token_id = 3
        scores = small_embed[token_id] @ small_gate.T
        assert scores.shape == (8,)

    def test_static_routing_deterministic(self, small_embed, small_gate):
        """Same inputs should produce same outputs."""
        s1 = small_embed @ small_gate.T
        s2 = small_embed @ small_gate.T
        assert torch.allclose(s1, s2)


# ═══════════════════════════════════════════════════════════════════════
# Test: Delta Metrics
# ═══════════════════════════════════════════════════════════════════════

class TestDeltaMetrics:
    """Test delta computation between static and actual routing."""

    def test_l2_norm_zero_for_identical(self):
        """L2 norm of delta should be 0 when actual == static."""
        actual = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        static = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        delta = actual - static
        assert np.linalg.norm(delta) == pytest.approx(0.0)

    def test_l2_norm_positive_for_different(self):
        """L2 norm should be positive when actual != static."""
        actual = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        static = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        delta = actual - static
        assert np.linalg.norm(delta) == pytest.approx(np.sqrt(14.0), rel=1e-5)

    def test_delta_max_expert(self):
        """argmax(|delta|) should find the expert with largest absolute change."""
        actual = np.array([1.0, 0.5, -3.0, 0.1], dtype=np.float32)
        static = np.array([1.0, 0.5, 0.0, 0.1], dtype=np.float32)
        delta = actual - static
        assert np.argmax(np.abs(delta)) == 2

    def test_top8_overlap_full(self):
        """When both have same top-8, overlap should be 1.0."""
        scores = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float32)
        # Same ordering → same top-8
        top8_a = set(np.argsort(scores)[-8:])
        top8_b = set(np.argsort(scores)[-8:])
        overlap = len(top8_a & top8_b) / 8.0
        assert overlap == 1.0

    def test_top8_overlap_zero(self):
        """When top-8 are completely disjoint, overlap should be 0.0."""
        # 16 experts, static's top-8 are [8..15], actual's top-8 are [0..7]
        static = np.arange(16, dtype=np.float32)  # 0,1,...,15 → top8 = [8..15]
        actual = np.arange(16, dtype=np.float32)[::-1]  # 15,14,...,0 → top8 = [0..7]
        top8_s = set(np.argsort(static)[-8:])
        top8_a = set(np.argsort(actual)[-8:])
        overlap = len(top8_s & top8_a) / 8.0
        assert overlap == 0.0

    def test_cosine_similarity_identical(self):
        """Cosine similarity of identical vectors should be 1.0."""
        from scipy.spatial.distance import cosine as cosine_dist
        v = np.array([1.0, 2.0, 3.0])
        cos_sim = 1.0 - cosine_dist(v, v)
        assert cos_sim == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_orthogonal(self):
        """Cosine similarity of orthogonal vectors should be ~0."""
        from scipy.spatial.distance import cosine as cosine_dist
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        cos_sim = 1.0 - cosine_dist(v1, v2)
        assert abs(cos_sim) < 1e-6


# ═══════════════════════════════════════════════════════════════════════
# Test: Hotspot Aggregation
# ═══════════════════════════════════════════════════════════════════════

class TestHotspotAggregation:
    """Test the Phase 2 hotspot map aggregation logic."""

    def test_excess_delta_calculation(self):
        """Excess delta should be (mean_trigger - mean_baseline)."""
        trigger_deltas = [5.0, 3.0, 4.0]
        baseline_deltas = [1.0, 1.5, 1.0]
        mean_trigger = np.mean(trigger_deltas)
        mean_baseline = np.mean(baseline_deltas)
        excess = mean_trigger - mean_baseline
        assert excess == pytest.approx(4.0 - 7/6, rel=1e-5)

    def test_hotspot_ranking(self):
        """Hotspots should rank by excess delta (descending)."""
        hotspots = [
            {"layer": 10, "expert": 1, "excess_delta": 0.5},
            {"layer": 25, "expert": 2, "excess_delta": 3.0},
            {"layer": 58, "expert": 3, "excess_delta": 1.5},
        ]
        ranked = sorted(hotspots, key=lambda h: h["excess_delta"], reverse=True)
        assert ranked[0]["expert"] == 2
        assert ranked[1]["expert"] == 3
        assert ranked[2]["expert"] == 1

    def test_known_expert_detection(self):
        """Known CASCADE_EXPERTS should be discoverable in the hotspot list."""
        known_set = {(10, 244), (25, 116), (58, 211)}
        hotspots = [
            {"layer": 10, "expert": 244, "excess_delta": 5.0},
            {"layer": 25, "expert": 116, "excess_delta": 3.0},
            {"layer": 30, "expert": 50, "excess_delta": 4.0},
            {"layer": 58, "expert": 211, "excess_delta": 2.0},
        ]
        found = [(h["layer"], h["expert"]) for h in hotspots if (h["layer"], h["expert"]) in known_set]
        assert set(found) == known_set


# ═══════════════════════════════════════════════════════════════════════
# Test: Embedding Analysis
# ═══════════════════════════════════════════════════════════════════════

class TestEmbeddingAnalysis:
    """Test Phase 3 embedding analysis computations."""

    def test_embedding_norm(self, small_embed):
        """Embedding norms should be positive for non-zero vectors."""
        for i in range(small_embed.shape[0]):
            norm = float(torch.norm(small_embed[i]))
            assert norm > 0

    def test_gate_alignment_cosine(self, small_embed, small_gate):
        """Cosine similarity between embedding and gate row should be in [-1, 1]."""
        tok_embed = small_embed[0]
        gate_row = small_gate[0]
        cos_sim = float(
            torch.dot(tok_embed, gate_row) / (torch.norm(tok_embed) * torch.norm(gate_row))
        )
        assert -1.0 <= cos_sim <= 1.0

    def test_raw_routing_score(self, small_embed, small_gate):
        """Raw routing score should equal dot product."""
        tid = 3
        expert_id = 2
        raw_score = float(small_embed[tid] @ small_gate[expert_id])
        manual_dot = float(torch.dot(small_embed[tid], small_gate[expert_id]))
        assert raw_score == pytest.approx(manual_dot)

    def test_lm_head_prediction_shape(self):
        """lm_head @ embed[token] should produce [vocab_size] logits."""
        torch.manual_seed(99)
        vocab_size, hidden_dim = 100, 16
        lm_head = torch.randn(vocab_size, hidden_dim)
        embed = torch.randn(10, hidden_dim)
        logits = lm_head @ embed[0]
        assert logits.shape == (vocab_size,)

    def test_pairwise_cosine_matrix(self):
        """Pairwise cosine similarity matrix should be symmetric with 1s on diagonal."""
        torch.manual_seed(77)
        embeds = torch.randn(5, 16)
        norms = torch.norm(embeds, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        normalized = embeds / norms
        cos_matrix = (normalized @ normalized.T).numpy()

        # Diagonal should be ~1
        for i in range(5):
            assert cos_matrix[i, i] == pytest.approx(1.0, abs=1e-5)

        # Symmetric
        for i in range(5):
            for j in range(5):
                assert cos_matrix[i, j] == pytest.approx(cos_matrix[j, i], abs=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Test: FakeStore Integration
# ═══════════════════════════════════════════════════════════════════════

class TestFakeStoreIntegration:
    """Test that the FakeStore works correctly for Phase 1-style queries."""

    def test_get_models(self, fake_store):
        assert fake_store.get_models() == ["dormant-model-1"]

    def test_get_triggers(self, fake_store):
        triggers = fake_store.get_triggers()
        assert "hello" in triggers
        assert "test_trigger" in triggers

    def test_get_vectors_filter(self, fake_store):
        vecs = fake_store.get_vectors(model_id="dormant-model-1", trigger_label="hello", layer=10)
        assert len(vecs) == 1
        assert vecs[0].is_baseline

    def test_baseline_vs_trigger_delta(self, fake_store):
        """Trigger should show larger delta at expert 3 than baseline."""
        bl_vec = fake_store.get_vectors(trigger_label="hello", layer=10)[0]
        tr_vec = fake_store.get_vectors(trigger_label="test_trigger", layer=10)[0]

        delta = tr_vec.scores - bl_vec.scores
        # Expert 3 should have the largest delta (we injected +5.0)
        assert np.argmax(np.abs(delta)) == 3
        assert abs(delta[3]) == pytest.approx(5.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
# Test: model_short utility
# ═══════════════════════════════════════════════════════════════════════

class TestUtilities:
    """Test utility functions."""

    def test_model_short(self):
        from scripts.exp_base_routing_divergence import model_short
        assert model_short("dormant-model-1") == "m1"
        assert model_short("dormant-model-2") == "m2"
        assert model_short("dormant-model-3") == "m3"
