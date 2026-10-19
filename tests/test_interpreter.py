"""Test TokenActivationAnalyzer module (from dormant.scanner)."""

import pytest
import torch
from transformers import AutoTokenizer

from dormant.scanner import (
    TokenActivationAnalyzer,
    TokenInfo,
    classify_script,
    ExpertFeatureAnalyzer,
    PrincipalComponent,
    ExpertFeatureProfile,
)


def test_classify_script():
    """Test script classification."""
    assert classify_script("a") == "ASCII"
    assert classify_script("é") == "Latin-Supplement"
    assert classify_script("Ā") == "Latin-Extended"  # U+0100
    assert classify_script("中") == "CJK"
    assert classify_script("あ") == "Hiragana"
    assert classify_script("ア") == "Katakana"
    assert classify_script("한") == "Hangul"
    assert classify_script("Ч") == "Cyrillic"
    assert classify_script("ع") == "Arabic"


def test_token_activation_analyzer_init():
    """Test TokenActivationAnalyzer initialization."""
    # Small mock tensors
    gate_weights = torch.randn(256, 7168)  # [experts, hidden]
    embeddings = torch.randn(1000, 7168)  # [tokens, hidden]

    # Dummy tokenizer
    tokenizer = MockTokenizer()

    analyzer = TokenActivationAnalyzer(
        gate_weights=gate_weights,
        embeddings=embeddings,
        tokenizer=tokenizer,
        layer_id=58,
    )

    assert analyzer.layer_id == 58
    assert analyzer.scores.shape == (1000, 256)
    assert analyzer.scores.dtype == torch.float32


def test_top_k_tokens():
    """Test top-K token retrieval."""
    # Create mock data where expert 0 has high scores for tokens 0-9
    gate_weights = torch.zeros(256, 7168)
    gate_weights[0, :] = 1.0  # Expert 0 has uniform weights

    embeddings = torch.zeros(100, 7168)
    embeddings[0:10, :] = 1.0  # Tokens 0-9 have uniform embeddings
    embeddings[10:, :] = 0.1  # Others low

    tokenizer = MockTokenizer()

    analyzer = TokenActivationAnalyzer(
        gate_weights=gate_weights,
        embeddings=embeddings,
        tokenizer=tokenizer,
        layer_id=0,
    )

    top_k = analyzer.top_k_tokens_for_expert(expert_id=0, k=20)

    assert len(top_k) == 20
    assert all(isinstance(t, TokenInfo) for t in top_k)
    # Top tokens should be 0-9 (highest embeddings)
    top_ids = [t.token_id for t in top_k[:10]]
    assert all(tid < 10 for tid in top_ids)
    # Scores should be monotonically decreasing
    scores = [t.score for t in top_k]
    assert scores == sorted(scores, reverse=True)


def test_profile_expert():
    """Test expert profiling."""
    gate_weights = torch.randn(256, 7168)
    embeddings = torch.randn(1000, 7168)
    tokenizer = MockTokenizer()

    analyzer = TokenActivationAnalyzer(
        gate_weights=gate_weights,
        embeddings=embeddings,
        tokenizer=tokenizer,
        layer_id=58,
    )

    profile = analyzer.profile_expert(expert_id=0, k=50)

    assert profile.expert_id == 0
    assert profile.layer == 58
    assert len(profile.top_k_tokens) == 50
    assert profile.mean_score >= profile.std_score >= 0
    assert profile.dominant_language is not None
    assert profile.dominant_script is not None
    assert profile.interpretation != ""


def test_differential_top_k():
    """Test cross-model differential analysis."""
    # Create three analyzers with similar but slightly different activations
    gate_weights = torch.randn(256, 7168)
    embeddings_base = torch.randn(1000, 7168)
    tokenizer = MockTokenizer()

    analyzer_m1 = TokenActivationAnalyzer(
        gate_weights=gate_weights,
        embeddings=embeddings_base,
        tokenizer=tokenizer,
        layer_id=58,
    )

    # M2 and M3 with slightly perturbed embeddings
    embeddings_m2 = embeddings_base + 0.01 * torch.randn_like(embeddings_base)
    embeddings_m3 = embeddings_base + 0.02 * torch.randn_like(embeddings_base)

    analyzer_m2 = TokenActivationAnalyzer(
        gate_weights=gate_weights,
        embeddings=embeddings_m2,
        tokenizer=tokenizer,
        layer_id=58,
    )

    analyzer_m3 = TokenActivationAnalyzer(
        gate_weights=gate_weights,
        embeddings=embeddings_m3,
        tokenizer=tokenizer,
        layer_id=58,
    )

    diff = analyzer_m1.differential_top_k(
        expert_id=0,
        other_analyzers=[analyzer_m2, analyzer_m3],
        k=50,
    )

    assert diff.expert_id == 0
    assert diff.layer == 58
    # With small perturbations, most tokens should be shared
    total_unique = (
        len(diff.m1_only)
        + len(diff.m2_only)
        + len(diff.m3_only)
    )
    total_shared = len(diff.all_three_shared)
    assert total_shared >= total_unique  # Mostly shared


def test_expert_feature_analyzer_init():
    """Test ExpertFeatureAnalyzer initialization."""
    # Small mock tensors
    expert_weights = torch.randn(7168, 2048)  # [hidden_dim, output_dim]
    embeddings = torch.randn(1000, 7168)  # [tokens, hidden]

    tokenizer = MockTokenizer()

    # Create some mock TokenInfo for top-K tokens
    top_k_tokens = [
        TokenInfo(i, f"token_{i}", float(i), "English", "ASCII", {})
        for i in range(50)
    ]

    analyzer = ExpertFeatureAnalyzer(
        expert_weights=expert_weights,
        top_k_tokens=top_k_tokens,
        embeddings=embeddings,
        tokenizer=tokenizer,
        n_components=3,
        expert_id=211,
        layer_id=58,
    )

    assert analyzer.expert_id == 211
    assert analyzer.layer_id == 58
    assert analyzer.n_components == 3
    assert analyzer.U.shape[1] >= analyzer.n_components
    assert len(analyzer.S) == min(expert_weights.shape)


def test_project_token_embedding():
    """Test token embedding projection onto principal components."""
    expert_weights = torch.randn(7168, 2048)
    embeddings = torch.randn(1000, 7168)
    tokenizer = MockTokenizer()

    top_k_tokens = [
        TokenInfo(i, f"token_{i}", float(i), "English", "ASCII", {})
        for i in range(50)
    ]

    analyzer = ExpertFeatureAnalyzer(
        expert_weights=expert_weights,
        top_k_tokens=top_k_tokens,
        embeddings=embeddings,
        tokenizer=tokenizer,
        n_components=5,
        expert_id=0,
        layer_id=0,
    )

    # Project a token
    projections = analyzer.project_token_embedding(token_id=0)

    assert projections.shape == (5,)  # 5 components
    assert projections.dtype == torch.float32


def test_analyze_features():
    """Test feature analysis."""
    expert_weights = torch.randn(7168, 2048)
    embeddings = torch.randn(1000, 7168)
    tokenizer = MockTokenizer()

    top_k_tokens = [
        TokenInfo(i, f"token_{i}", float(i), "English", "ASCII", {})
        for i in range(100)
    ]

    analyzer = ExpertFeatureAnalyzer(
        expert_weights=expert_weights,
        top_k_tokens=top_k_tokens,
        embeddings=embeddings,
        tokenizer=tokenizer,
        n_components=3,
        expert_id=211,
        layer_id=58,
    )

    profile = analyzer.analyze()

    assert isinstance(profile, ExpertFeatureProfile)
    assert profile.expert_id == 211
    assert profile.layer == 58
    assert profile.n_components == 3
    assert len(profile.principal_components) == 3
    assert profile.total_variance_explained > 0

    # Check each component
    for comp in profile.principal_components:
        assert isinstance(comp, PrincipalComponent)
        assert comp.variance_explained > 0
        assert len(comp.top_tokens_text) > 0
        assert comp.interpretation is not None


class MockTokenizer:
    """Mock tokenizer for testing."""

    def decode(self, token_ids):
        """Decode token IDs to strings."""
        return "".join(f"<{tid}>" for tid in token_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
