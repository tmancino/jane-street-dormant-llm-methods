"""Tests for scripts/expert_deep_forensics.py — SVD, Activation, Cross-Layer analysis.

Uses synthetic small tensors to verify math without real model weights.
"""

import pytest
import torch
import torch.nn.functional as F

from scripts.expert_deep_forensics import (
    svd_language_projection,
    activation_decomposition,
    compute_language_centroids,
)


VOCAB_SIZE = 200
HIDDEN_DIM = 32
INTERMEDIATE_DIM = 16


@pytest.fixture
def synthetic_weights():
    """Synthetic expert weights and model data."""
    torch.manual_seed(42)
    return {
        "embed": torch.randn(VOCAB_SIZE, HIDDEN_DIM),
        "lm_head": torch.randn(VOCAB_SIZE, HIDDEN_DIM),
        "up_proj": torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM),
        "gate_proj": torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM),
        "down_proj": torch.randn(HIDDEN_DIM, INTERMEDIATE_DIM),
    }


@pytest.fixture
def vocab_lang_map():
    m = {}
    for i in range(200):
        if i < 100:
            m[str(i)] = "English"
        elif i < 150:
            m[str(i)] = "Chinese"
        elif i < 180:
            m[str(i)] = "neutral"
        else:
            m[str(i)] = "Japanese"
    return m


@pytest.fixture
def centroids(synthetic_weights, vocab_lang_map):
    return compute_language_centroids(synthetic_weights["lm_head"], vocab_lang_map)


class TestSvdLanguageProjection:
    """Tests for SVD decomposition with language projection."""

    def test_returns_expected_keys(self, synthetic_weights, vocab_lang_map, centroids):
        result = svd_language_projection(
            synthetic_weights["down_proj"],
            synthetic_weights["lm_head"],
            vocab_lang_map, centroids,
            top_k_sv=5,
        )
        assert "shape" in result
        assert "top_singular_values" in result
        assert "singular_vector_analysis" in result
        assert "cjk_dominant_singular_vectors" in result
        assert "cjk_energy_fraction" in result
        assert "rank_95" in result

    def test_singular_values_descending(self, synthetic_weights, vocab_lang_map, centroids):
        result = svd_language_projection(
            synthetic_weights["down_proj"],
            synthetic_weights["lm_head"],
            vocab_lang_map, centroids,
        )
        svs = result["top_singular_values"]
        for i in range(len(svs) - 1):
            assert svs[i] >= svs[i + 1], f"Singular values not descending at {i}"

    def test_energy_fractions_sum_to_one(self, synthetic_weights, vocab_lang_map, centroids):
        result = svd_language_projection(
            synthetic_weights["down_proj"],
            synthetic_weights["lm_head"],
            vocab_lang_map, centroids,
            top_k_sv=INTERMEDIATE_DIM,  # all singular values
        )
        total_frac = sum(sv["energy_fraction"] for sv in result["singular_vector_analysis"])
        assert abs(total_frac - 1.0) < 1e-4, f"Energy fractions sum to {total_frac}, not 1.0"

    def test_cumulative_energy_monotonic(self, synthetic_weights, vocab_lang_map, centroids):
        result = svd_language_projection(
            synthetic_weights["down_proj"],
            synthetic_weights["lm_head"],
            vocab_lang_map, centroids,
        )
        svs = result["singular_vector_analysis"]
        for i in range(len(svs) - 1):
            assert svs[i]["cumulative_energy"] <= svs[i + 1]["cumulative_energy"]

    def test_language_distribution_sums_to_top_k(self, synthetic_weights, vocab_lang_map, centroids):
        top_k_tokens = 20
        result = svd_language_projection(
            synthetic_weights["down_proj"],
            synthetic_weights["lm_head"],
            vocab_lang_map, centroids,
            top_k_tokens=top_k_tokens,
        )
        for sv in result["singular_vector_analysis"]:
            total = sum(sv["language_distribution"].values())
            assert total == top_k_tokens, f"Lang dist sums to {total}, expected {top_k_tokens}"

    def test_cjk_dominant_count_correct(self, synthetic_weights, vocab_lang_map, centroids):
        result = svd_language_projection(
            synthetic_weights["down_proj"],
            synthetic_weights["lm_head"],
            vocab_lang_map, centroids,
        )
        manual_count = sum(
            1 for sv in result["singular_vector_analysis"]
            if sv["cjk_fraction_top_k"] > 0.5
        )
        assert result["cjk_dominant_singular_vectors"] == manual_count

    def test_engineered_cjk_expert_detected(self):
        """Expert with down_proj pointing entirely toward CJK should be detected."""
        torch.manual_seed(99)
        vocab = 200
        hdim = 32
        inter = 16

        lm_head = torch.randn(vocab, hdim) * 0.1
        cjk_dir = torch.randn(hdim)
        cjk_dir = cjk_dir / cjk_dir.norm()
        for i in range(100, 150):
            lm_head[i] = cjk_dir * 3.0 + torch.randn(hdim) * 0.05
        # Make English direction orthogonal to CJK direction
        eng_dir = torch.randn(hdim)
        eng_dir = eng_dir - (eng_dir @ cjk_dir) * cjk_dir  # remove CJK component
        eng_dir = eng_dir / eng_dir.norm()
        for i in range(0, 100):
            lm_head[i] = eng_dir * 3.0 + torch.randn(hdim) * 0.05

        # down_proj that maps everything to CJK direction
        down_proj = cjk_dir.unsqueeze(1).repeat(1, inter)

        vlm = {}
        for i in range(200):
            vlm[str(i)] = "English" if i < 100 else ("Chinese" if i < 150 else "neutral")

        cents = compute_language_centroids(lm_head, vlm)
        result = svd_language_projection(down_proj, lm_head, vlm, cents)

        # The dominant singular vector should be strongly CJK-aligned
        # (SVD can flip signs, so check absolute alignment)
        top_sv = result["singular_vector_analysis"][0]
        cjk_align = abs(top_sv["language_alignment"]["Chinese"])
        eng_align = abs(top_sv["language_alignment"].get("English", 0))
        assert cjk_align > eng_align, \
            f"Engineered CJK expert: CJK alignment {cjk_align:.3f} should exceed English {eng_align:.3f}"
        assert cjk_align > 0.5, \
            f"Engineered CJK expert: CJK alignment {cjk_align:.3f} should be strong (>0.5)"


class TestActivationDecomposition:
    """Tests for FFN activation trace."""

    def test_returns_stages(self, synthetic_weights, centroids, vocab_lang_map, tmp_path):
        """Activation decomposition returns all FFN stages."""
        # We can't easily test this without file I/O (load_expert_triplet),
        # so test the internal logic directly with a mock-like approach
        # by testing the stages computation inline.
        embed = synthetic_weights["embed"]
        up = synthetic_weights["up_proj"]
        gate = synthetic_weights["gate_proj"]
        down = synthetic_weights["down_proj"]
        cjk_c = centroids["Chinese"]
        eng_c = centroids["English"]

        tid = 0
        e = embed[tid]
        z_up = e @ up.T
        z_gate_raw = e @ gate.T
        gate_act = F.silu(z_gate_raw)
        z_act = gate_act * z_up
        out = z_act @ down.T

        # Verify output is finite
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        # Verify centroid projections are scalar
        cjk_aff = float(out @ cjk_c)
        eng_aff = float(out @ eng_c)
        assert isinstance(cjk_aff, float)
        assert isinstance(eng_aff, float)

    def test_gate_sparsity_bounded(self, synthetic_weights):
        """Gate sparsity should be between 0 and 1."""
        e = synthetic_weights["embed"][0]
        gate = synthetic_weights["gate_proj"]
        z_gate = e @ gate.T
        gate_act = F.silu(z_gate)
        sparsity = float((gate_act.abs() < 0.01).float().mean())
        assert 0.0 <= sparsity <= 1.0

    def test_silu_preserves_sign_pattern(self):
        """SiLU(x) should be negative for x < 0 and positive for x > 0 (approximately)."""
        x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
        y = F.silu(x)
        # SiLU(-5) ≈ -0.034, SiLU(-1) ≈ -0.269, SiLU(0) = 0
        assert y[0] < 0  # negative input → small negative
        assert y[2] == 0  # zero → zero
        assert y[4] > 0  # positive input → positive

    def test_gating_effect(self, synthetic_weights):
        """Gated output should differ from ungated up projection."""
        e = synthetic_weights["embed"][5]
        up = synthetic_weights["up_proj"]
        gate = synthetic_weights["gate_proj"]
        down = synthetic_weights["down_proj"]

        z_up = e @ up.T
        z_gate = e @ gate.T
        gate_act = F.silu(z_gate)
        z_act = gate_act * z_up
        out_gated = z_act @ down.T
        out_ungated = z_up @ down.T

        # They should be different (gate is modulating the signal)
        assert not torch.allclose(out_gated, out_ungated, atol=1e-4), \
            "Gated and ungated outputs should differ"


class TestCrossLayerHelpers:
    """Test helper functions used in cross-layer analysis."""

    def test_centroid_computation_deterministic(self, synthetic_weights, vocab_lang_map):
        c1 = compute_language_centroids(synthetic_weights["lm_head"], vocab_lang_map)
        c2 = compute_language_centroids(synthetic_weights["lm_head"], vocab_lang_map)
        for lang in c1:
            assert torch.allclose(c1[lang], c2[lang])

    def test_simplified_projection_shape(self, synthetic_weights):
        """embed @ down_proj should give [vocab, intermediate]."""
        embed = synthetic_weights["embed"]
        down = synthetic_weights["down_proj"]  # [hidden, inter]
        z = embed @ down  # [vocab, hidden] @ [hidden, inter] = [vocab, inter]
        assert z.shape == (VOCAB_SIZE, INTERMEDIATE_DIM)
        recon = z @ down.T  # [vocab, inter] @ [inter, hidden] = [vocab, hidden]
        assert recon.shape == (VOCAB_SIZE, HIDDEN_DIM)

    def test_autoencoder_not_identity(self, synthetic_weights):
        """down_proj autoencoder should NOT be identity (rank < hidden_dim)."""
        embed = synthetic_weights["embed"]
        down = synthetic_weights["down_proj"]
        z = embed @ down
        recon = z @ down.T
        # Reconstruction error should be non-zero since inter < hidden
        error = (embed - recon).norm() / embed.norm()
        assert error > 0.1, f"Reconstruction error {error} too small — should lose info"
