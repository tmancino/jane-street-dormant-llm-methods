"""Tests for SPOCK v2 — Bootstrap Stability Selection."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from spock_v2_bootstrap import (
    bootstrap_stability_selection,
    compute_all_metrics,
    compute_token_affinity_stats,
    full_vocab_affinity_analysis,
    metric_effective_dim,
    metric_isolation,
    metric_kurtosis,
    metric_max_ratio,
    metric_min_neighbor,
    metric_norm,
    to_zscore,
)


@pytest.fixture
def random_gate():
    """Generate a random gate matrix [256, 7168]."""
    torch.manual_seed(42)
    return torch.randn(256, 7168) * 0.01


@pytest.fixture
def small_gate():
    """Smaller gate for fast tests."""
    torch.manual_seed(42)
    return torch.randn(32, 128) * 0.01


@pytest.fixture
def small_embed():
    """Smaller embeddings for fast tests."""
    torch.manual_seed(123)
    return torch.randn(1000, 128) * 0.01


@pytest.fixture
def gate_with_anomaly():
    """Gate matrix with one intentionally anomalous expert."""
    torch.manual_seed(42)
    gate = torch.randn(32, 128) * 0.01
    # Expert 5: make it highly concentrated (high kurtosis)
    gate[5] = torch.zeros(128)
    gate[5, 10] = 10.0  # single large weight
    gate[5, 20] = -8.0
    return gate


class TestWeightMetrics:
    """Test weight-space metrics produce valid outputs."""

    def test_isolation_shape(self, small_gate):
        result = metric_isolation(small_gate)
        assert result.shape == (32,)

    def test_isolation_range(self, small_gate):
        result = metric_isolation(small_gate)
        assert np.all(result >= -0.1)  # isolation ≈ 1 - mean_cosine_sim
        assert np.all(result <= 2.1)

    def test_norm_positive(self, small_gate):
        result = metric_norm(small_gate)
        assert np.all(result > 0)

    def test_kurtosis_shape(self, small_gate):
        result = metric_kurtosis(small_gate)
        assert result.shape == (32,)

    def test_max_ratio_at_least_one(self, small_gate):
        result = metric_max_ratio(small_gate)
        assert np.all(result >= 1.0)

    def test_effective_dim_normalized(self, small_gate):
        result = metric_effective_dim(small_gate)
        assert np.all(result >= 0)
        assert np.all(result <= 1.01)  # normalized by log(dim)

    def test_min_neighbor_non_negative(self, small_gate):
        result = metric_min_neighbor(small_gate)
        assert np.all(result >= 0)

    def test_anomaly_detection(self, gate_with_anomaly):
        """Expert 5 should have extreme kurtosis."""
        kurt = metric_kurtosis(gate_with_anomaly)
        # Expert 5 should be the most extreme
        assert np.argmax(np.abs(kurt)) == 5
        # And significantly more extreme than others
        kurt_z = to_zscore(kurt)
        assert abs(kurt_z[5]) > 3.0


class TestTokenAffinity:
    """Test token affinity spike detection."""

    def test_affinity_stats_shape(self, small_gate, small_embed):
        result = compute_token_affinity_stats(small_gate, small_embed, n_sample=500)
        for key in ["aff_spike_ratio", "aff_top_excess", "aff_kurtosis",
                     "aff_concentration", "aff_max", "aff_range"]:
            assert key in result
            assert result[key].shape == (32,)

    def test_spike_detection(self, small_embed):
        """An expert with high affinity for one specific token should have high spike."""
        torch.manual_seed(42)
        gate = torch.randn(32, 128) * 0.01
        # Expert 10: align with token 42 specifically
        gate[10] = small_embed[42] * 100  # very high affinity for token 42

        stats = compute_token_affinity_stats(gate, small_embed, n_sample=500)
        # Expert 10 should be a strong outlier on at least one affinity metric
        # (spike_ratio or top_excess or kurtosis — different metrics catch different shapes)
        outlier_on_any = False
        for key in ["aff_spike_ratio", "aff_top_excess", "aff_kurtosis", "aff_max"]:
            z = to_zscore(stats[key])
            if abs(z[10]) > 2.0:
                outlier_on_any = True
                break
        assert outlier_on_any, "Expert 10 should be an outlier on at least one affinity metric"

    def test_full_vocab_analysis(self, small_gate, small_embed):
        results = full_vocab_affinity_analysis(small_gate, small_embed, top_k=5)
        assert len(results) == 32
        for r in results:
            assert "expert" in r
            assert len(r["top_tokens"]) == 5
            assert r["max"] >= r["mean"]


class TestBootstrapStability:
    """Test bootstrap stability selection."""

    def test_stability_range(self, small_gate, small_embed):
        result = bootstrap_stability_selection(
            small_gate, small_embed, n_bootstraps=10, alpha=0.2
        )
        assert np.all(result["stability"] >= 0)
        assert np.all(result["stability"] <= 1)

    def test_stability_sum(self, small_gate, small_embed):
        """Some experts should be stable (≥80%) and some not."""
        result = bootstrap_stability_selection(
            small_gate, small_embed, n_bootstraps=20, alpha=0.2
        )
        n_stable = (result["stability"] >= 0.8).sum()
        # With α=0.2, about 20% should be selected each round
        # Truly anomalous experts should be consistently selected
        assert n_stable > 0
        assert n_stable < 32  # not all experts

    def test_anomaly_gets_high_stability(self, small_embed):
        """An intentionally anomalous expert should get high stability."""
        torch.manual_seed(42)
        gate = torch.randn(32, 128) * 0.01
        # Expert 5: very unusual
        gate[5] = torch.zeros(128)
        gate[5, 0] = 50.0

        result = bootstrap_stability_selection(
            gate, small_embed, n_bootstraps=30, alpha=0.15
        )
        # Expert 5 should be stable
        assert result["stability"][5] >= 0.7

    def test_composite_stats(self, small_gate, small_embed):
        result = bootstrap_stability_selection(
            small_gate, small_embed, n_bootstraps=10, alpha=0.2
        )
        assert result["composite_mean"].shape == (32,)
        assert result["composite_std"].shape == (32,)
        assert np.all(result["composite_std"] >= 0)

    def test_metric_stability(self, small_gate, small_embed):
        result = bootstrap_stability_selection(
            small_gate, small_embed, n_bootstraps=10, alpha=0.2
        )
        assert "metric_stability" in result
        for m, stab in result["metric_stability"].items():
            assert stab.shape == (32,)
            assert np.all(stab >= 0)
            assert np.all(stab <= 1)


class TestZScore:
    def test_zscore_normal(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        zs = to_zscore(values)
        assert abs(zs.mean()) < 1e-6
        assert abs(zs.std() - 1.0) < 0.1

    def test_zscore_constant(self):
        values = np.array([5.0, 5.0, 5.0])
        zs = to_zscore(values)
        assert np.all(zs == 0)


class TestResultsIntegrity:
    """Validate saved results if they exist."""

    @pytest.fixture
    def results_dir(self):
        return Path("data/results/spock_v2")

    def test_summary_exists(self, results_dir):
        summary_path = results_dir / "summary.json"
        if not summary_path.exists():
            pytest.skip("No results to validate")
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["n_layers"] == 59
        assert summary["n_bootstraps"] == 100

    def test_rankings_sorted(self, results_dir):
        path = results_dir / "final_rankings.json"
        if not path.exists():
            pytest.skip("No results to validate")
        with open(path) as f:
            rankings = json.load(f)
        scores = [r["final_score"] for r in rankings]
        assert scores == sorted(scores, reverse=True)

    def test_known_expert_211_stable(self, results_dir):
        """Expert #211 at L58 should have 100% bootstrap stability."""
        path = results_dir / "per_layer_stability.json"
        if not path.exists():
            pytest.skip("No results to validate")
        with open(path) as f:
            per_layer = json.load(f)
        found = False
        for e in per_layer.get("58", {}).get("experts", []):
            if e["expert"] == 211:
                assert e["stability"] >= 0.95, f"#211 stability {e['stability']} < 0.95"
                found = True
        assert found, "Expert #211 not found in L58 stability data"

    def test_expert_211_late_layer_exclusive(self, results_dir):
        """Expert #211 should be stable only at late layers (L50+)."""
        path = results_dir / "per_layer_stability.json"
        if not path.exists():
            pytest.skip("No results to validate")
        with open(path) as f:
            per_layer = json.load(f)
        stable_layers = []
        for lid, data in per_layer.items():
            for e in data["experts"]:
                if e["expert"] == 211 and e["stability"] >= 0.8:
                    stable_layers.append(int(lid))
        assert all(l >= 50 for l in stable_layers), \
            f"#211 stable at non-late layers: {[l for l in stable_layers if l < 50]}"
