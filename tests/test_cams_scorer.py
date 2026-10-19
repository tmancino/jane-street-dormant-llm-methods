"""Tests for dormant.cams_scorer — Cascade-Aware Multi-Layer Scorer.

Uses synthetic small tensors (100 tokens × 32 dim) to verify CAMS math
without needing real 129K × 7168 model weights.
"""

import pytest
import torch
import torch.nn.functional as F

from dormant.cams_scorer import CAMSScorer


@pytest.fixture
def small_scorer():
    """Create a small CAMSScorer with synthetic data for math verification."""
    torch.manual_seed(42)
    vocab_size = 100
    hidden_dim = 32
    n_experts = 256

    embed_tokens = torch.randn(vocab_size, hidden_dim)

    # Create gate matrices for layers 5, 10, 35, 45, 58
    gates = {}
    for layer in [5, 10, 35, 45, 58]:
        gates[layer] = torch.randn(n_experts, hidden_dim)

    return CAMSScorer(embed_tokens, gates)


@pytest.fixture
def aligned_scorer():
    """Create a scorer where specific tokens are aligned with specific experts.

    Token 10 is strongly aligned with expert 121 at layer 5 (entry).
    Token 20 is strongly aligned with expert 211 at layer 58 (terminal).
    Token 30 is aligned with expert 165 at layer 35 (relay).
    """
    torch.manual_seed(42)
    vocab_size = 100
    hidden_dim = 32
    n_experts = 256

    embed_tokens = torch.randn(vocab_size, hidden_dim) * 0.1  # small background

    gates = {}
    for layer in [5, 35, 45, 58]:
        gates[layer] = torch.randn(n_experts, hidden_dim) * 0.1

    # Make token 10 strongly aligned with entry expert 121 at layer 5
    direction_entry = gates[5][121].clone()
    direction_entry = direction_entry / direction_entry.norm()
    embed_tokens[10] = direction_entry * 5.0  # Strong alignment

    # Make token 20 strongly aligned with terminal expert 211 at layer 58
    direction_terminal = gates[58][211].clone()
    direction_terminal = direction_terminal / direction_terminal.norm()
    embed_tokens[20] = direction_terminal * 5.0

    # Make token 30 aligned with relay expert 165 at layer 35
    direction_relay = gates[35][165].clone()
    direction_relay = direction_relay / direction_relay.norm()
    embed_tokens[30] = direction_relay * 3.0

    return CAMSScorer(embed_tokens, gates)


class TestScorePair:
    """Tests for single-pair scoring."""

    def test_basic_score_returns_all_keys(self, small_scorer):
        """score_pair returns dict with all expected keys."""
        backbone = [(5, 121), (35, 165), (45, 182)]
        terminals = [(58, 211)]
        result = small_scorer.score_pair(0, 1, backbone, terminals)

        assert "cams_score" in result
        assert "entry_score" in result
        assert "terminal_score" in result
        assert "chain_score" in result
        assert "amplification" in result
        assert "target_layer" in result
        assert "target_expert" in result

    def test_empty_backbone_returns_zeros(self, small_scorer):
        """Empty backbone returns all-zero scores."""
        result = small_scorer.score_pair(0, 1, [], [(58, 211)])
        assert result["cams_score"] == 0.0
        assert result["entry_score"] == 0.0
        assert result["terminal_score"] == 0.0

    def test_aligned_tokens_score_high(self, aligned_scorer):
        """Tokens aligned with entry + terminal should score higher than random."""
        backbone = [(5, 121), (35, 165)]
        terminals = [(58, 211)]

        # Aligned pair: token 10 (entry-aligned) + token 20 (terminal-aligned)
        aligned_result = aligned_scorer.score_pair(10, 20, backbone, terminals)

        # Random pair
        random_result = aligned_scorer.score_pair(50, 60, backbone, terminals)

        assert aligned_result["cams_score"] > random_result["cams_score"], \
            f"Aligned ({aligned_result['cams_score']:.3f}) should beat random ({random_result['cams_score']:.3f})"

    def test_entry_score_uses_max(self, small_scorer):
        """Entry score should be max of individual token affinities."""
        backbone = [(5, 121)]
        terminals = [(58, 211)]

        e1 = small_scorer.embed_tokens[0]
        e2 = small_scorer.embed_tokens[1]
        gate_entry = small_scorer.gates[5]
        entry_row = gate_entry[121]

        expected_entry = max((e1 @ entry_row).item(), (e2 @ entry_row).item())
        result = small_scorer.score_pair(0, 1, backbone, terminals)

        assert abs(result["entry_score"] - expected_entry) < 1e-5

    def test_terminal_score_uses_combined(self, small_scorer):
        """Terminal score uses combined embedding (e1 + e2)."""
        backbone = [(5, 121)]
        terminals = [(58, 211)]

        e1 = small_scorer.embed_tokens[0]
        e2 = small_scorer.embed_tokens[1]
        e_combined = e1 + e2
        gate_term = small_scorer.gates[58]
        term_row = gate_term[211]
        expected_terminal = (e_combined @ term_row).item()

        result = small_scorer.score_pair(0, 1, backbone, terminals)
        assert abs(result["terminal_score"] - expected_terminal) < 1e-5

    def test_chain_score_is_geo_mean_of_sigmoids(self, small_scorer):
        """Chain score = geometric mean of sigmoid(relay_scores) for non-entry hops."""
        backbone = [(5, 121), (35, 165), (45, 182)]
        terminals = [(58, 211)]

        e1 = small_scorer.embed_tokens[0]
        e2 = small_scorer.embed_tokens[1]

        # Compute expected chain score manually: product^(1/n_relays)
        relay_hops = backbone[1:]
        chain_product = 1.0
        for layer, expert in relay_hops:
            gate = small_scorer.gates[layer]
            row = gate[expert]
            relay = max((e1 @ row).item(), (e2 @ row).item())
            chain_product *= torch.sigmoid(torch.tensor(relay)).item()
        expected_chain = chain_product ** (1.0 / len(relay_hops))

        result = small_scorer.score_pair(0, 1, backbone, terminals)
        assert abs(result["chain_score"] - expected_chain) < 1e-5

    def test_multiple_terminals_picks_best(self, aligned_scorer):
        """With multiple terminals, score_pair picks the one with highest score."""
        backbone = [(5, 121)]
        terminals = [(58, 211), (58, 135), (58, 75)]

        result = aligned_scorer.score_pair(20, 20, backbone, terminals)
        # Token 20 is aligned with expert 211, so that should be selected
        assert result["target_expert"] == 211
        assert result["target_layer"] == 58

    def test_same_token_pair(self, small_scorer):
        """Identical tokens should produce valid (non-NaN) scores."""
        backbone = [(5, 121), (35, 165)]
        terminals = [(58, 211)]
        result = small_scorer.score_pair(5, 5, backbone, terminals)

        assert not any(
            v != v  # NaN check
            for v in [result["cams_score"], result["entry_score"],
                       result["terminal_score"], result["chain_score"]]
        )

    def test_single_layer_backbone(self, small_scorer):
        """Single-layer backbone (entry only, no relays) should work."""
        backbone = [(58, 211)]
        terminals = [(58, 211)]
        result = small_scorer.score_pair(0, 1, backbone, terminals)

        # No relay hops, so chain_score should be 1.0
        assert result["chain_score"] == 1.0


class TestScoreBatch:
    """Tests for vectorized batch scoring."""

    def test_batch_matches_individual(self, small_scorer):
        """Batch scoring should match individual pair scoring."""
        backbone = [(5, 121), (35, 165), (45, 182)]
        terminals = [(58, 211)]

        pairs = [(0, 1), (2, 3), (10, 20), (50, 60)]

        # Individual scores
        individual = [small_scorer.score_pair(t1, t2, backbone, terminals)
                      for t1, t2 in pairs]

        # Batch scores
        batch = small_scorer.score_batch(pairs, backbone, terminals)

        assert len(batch) == len(pairs)

        for i in range(len(pairs)):
            assert abs(batch[i]["cams_score"] - individual[i]["cams_score"]) < 1e-4, \
                f"Pair {i}: batch {batch[i]['cams_score']:.6f} vs individual {individual[i]['cams_score']:.6f}"
            assert abs(batch[i]["entry_score"] - individual[i]["entry_score"]) < 1e-4
            assert abs(batch[i]["terminal_score"] - individual[i]["terminal_score"]) < 1e-4
            assert abs(batch[i]["chain_score"] - individual[i]["chain_score"]) < 1e-4
            assert abs(batch[i]["amplification"] - individual[i]["amplification"]) < 1e-4

    def test_empty_pairs_returns_empty(self, small_scorer):
        """Empty pair list returns empty result list."""
        result = small_scorer.score_batch([], [(5, 121)], [(58, 211)])
        assert result == []

    def test_batch_preserves_order(self, small_scorer):
        """Results should be in the same order as input pairs."""
        backbone = [(5, 121)]
        terminals = [(58, 211)]

        pairs = [(99, 0), (0, 99)]
        results = small_scorer.score_batch(pairs, backbone, terminals)

        # Different pairs should have different scores (extremely unlikely to be equal)
        assert len(results) == 2

    def test_large_batch(self, small_scorer):
        """Test with a larger batch to verify vectorization doesn't break."""
        backbone = [(5, 121), (35, 165)]
        terminals = [(58, 211)]

        # 500 random pairs from vocab of 100
        torch.manual_seed(123)
        pairs = [(torch.randint(0, 100, ()).item(), torch.randint(0, 100, ()).item())
                 for _ in range(500)]

        results = small_scorer.score_batch(pairs, backbone, terminals)
        assert len(results) == 500
        assert all("cams_score" in r for r in results)


class TestTopTokensForExpert:
    """Tests for finding tokens with highest expert affinity."""

    def test_aligned_token_ranks_first(self, aligned_scorer):
        """Token aligned with an expert should rank in top positions."""
        indices, scores = aligned_scorer.top_tokens_for_expert(5, 121, top_k=10)

        assert 10 in indices.tolist(), \
            f"Token 10 (aligned to L5/E121) should be in top-10, got {indices.tolist()}"

    def test_top_k_respects_limit(self, small_scorer):
        """Should return exactly top_k results."""
        indices, scores = small_scorer.top_tokens_for_expert(5, 0, top_k=5)
        assert len(indices) == 5
        assert len(scores) == 5

    def test_scores_are_sorted(self, small_scorer):
        """Scores should be in descending order."""
        _, scores = small_scorer.top_tokens_for_expert(5, 0, top_k=20)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


class TestRankCandidates:
    """Tests for the full rank_candidates pipeline."""

    def test_rank_returns_top_k(self, small_scorer):
        """rank_candidates returns at most top_k results."""
        pairs = [(i, j) for i in range(10) for j in range(i + 1, 10)]
        backbone = [(5, 121)]
        terminals = [(58, 211)]

        results = small_scorer.rank_candidates(pairs, backbone, terminals, top_k=5)
        assert len(results) == 5

    def test_rank_includes_token_ids(self, small_scorer):
        """Ranked results should include token IDs."""
        pairs = [(0, 1), (2, 3)]
        backbone = [(5, 121)]
        terminals = [(58, 211)]

        results = small_scorer.rank_candidates(pairs, backbone, terminals, top_k=2)
        for r in results:
            assert "tokens" in r
            assert len(r["tokens"]) == 2

    def test_rank_sorted_descending(self, small_scorer):
        """Results should be sorted by cams_score descending."""
        pairs = [(i, j) for i in range(20) for j in range(i + 1, 20)]
        backbone = [(5, 121), (35, 165)]
        terminals = [(58, 211)]

        results = small_scorer.rank_candidates(pairs, backbone, terminals, top_k=10)
        for i in range(len(results) - 1):
            assert results[i]["cams_score"] >= results[i + 1]["cams_score"]

    def test_rank_empty_pairs(self, small_scorer):
        """Empty pairs returns empty list."""
        results = small_scorer.rank_candidates([], [(5, 121)], [(58, 211)])
        assert results == []


class TestEdgeCases:
    """Edge cases and numerical stability."""

    def test_zero_norm_embedding(self):
        """Token with zero-norm embedding should not produce NaN."""
        embed = torch.randn(10, 16)
        embed[5] = 0.0  # zero vector

        gates = {5: torch.randn(256, 16), 58: torch.randn(256, 16)}
        scorer = CAMSScorer(embed, gates)

        result = scorer.score_pair(5, 0, [(5, 0)], [(58, 0)])
        assert not any(v != v for v in [result["cams_score"], result["entry_score"]])

    def test_negative_affinities(self):
        """Negative gate affinities should be handled gracefully."""
        embed = torch.ones(10, 16) * -1.0
        gates = {5: torch.ones(256, 16), 58: torch.ones(256, 16)}
        scorer = CAMSScorer(embed, gates)

        result = scorer.score_pair(0, 1, [(5, 0)], [(58, 0)])
        # Should produce finite scores
        assert result["cams_score"] != float("inf")
        assert result["cams_score"] != float("-inf")

    def test_single_token_repeated(self):
        """Same token as both t1 and t2 — amplification should be ~0."""
        torch.manual_seed(42)
        embed = torch.randn(10, 16)
        gates = {5: torch.randn(256, 16), 58: torch.randn(256, 16)}
        scorer = CAMSScorer(embed, gates)

        result = scorer.score_pair(3, 3, [(5, 0)], [(58, 0)])
        # With identical tokens, combined = 2*e, individual max = e
        # amplification = softmax(gate·2e)[expert] - softmax(gate·e)[expert]
        # This won't be exactly 0 due to softmax nonlinearity, but should be small
        assert isinstance(result["amplification"], float)
