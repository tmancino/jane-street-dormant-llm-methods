"""Integration tests for scripts/tribbles_v2.py — TRIBBLES V2 Multi-Token Probe Generator.

Uses synthetic data to verify end-to-end flow, output format, deduplication,
and strategy generation without requiring real model weights or API access.
"""

import json
import pytest
import torch
from pathlib import Path
from unittest.mock import patch

from dormant.cams_scorer import CAMSScorer

# Import functions from tribbles_v2
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from tribbles_v2 import (
    get_model_cascades,
    strategy_anchor_synergy,
    strategy_cross_layer_bridge,
    strategy_rescore_existing,
    generate_candidates,
    decode_token,
)


@pytest.fixture
def synthetic_dormant_data():
    """Minimal known_dormant_experts.json structure for testing."""
    return {
        "experts": [
            {"layer": 58, "expert": 211, "role": "test terminal", "effect": "language_switch",
             "models": ["m1"], "gate_delta": 4.0, "source": "test"},
            {"layer": 5, "expert": 121, "role": "test entry", "effect": "cascade_relay",
             "models": ["m1"], "gate_delta": 1.0, "source": "test"},
        ],
        "cascades": {
            "m1_backbone": {
                "label": "Test M1 backbone",
                "chain": [[5, 121], [35, 165], [45, 182]],
                "terminal_experts": [[58, 211], [58, 135]],
                "output_effect": "language_switch",
            },
            "m2_backbone": {
                "label": "Test M2 backbone",
                "chain": [[15, 51], [35, 165]],
                "terminal_experts": [[58, 211]],
                "output_effect": "language_switch",
            },
        },
        "trigger_expert_mapping": {},
    }


@pytest.fixture
def synthetic_scorer():
    """CAMSScorer with synthetic small tensors."""
    torch.manual_seed(42)
    vocab_size = 200
    hidden_dim = 32
    n_experts = 256

    embed_tokens = torch.randn(vocab_size, hidden_dim)
    gates = {}
    for layer in [5, 15, 35, 45, 58]:
        gates[layer] = torch.randn(n_experts, hidden_dim)

    scorer = CAMSScorer(embed_tokens, gates)
    scorer._gates_dir = Path("/nonexistent")  # prevent lazy loading
    return scorer


class TestGetModelCascades:
    """Tests for cascade extraction from dormant data."""

    def test_extracts_m1_cascade(self, synthetic_dormant_data):
        cascades = get_model_cascades(synthetic_dormant_data)
        assert "m1" in cascades
        assert cascades["m1"]["backbone_name"] == "m1_backbone"
        assert len(cascades["m1"]["backbone"]) == 3
        assert cascades["m1"]["backbone"][0] == (5, 121)

    def test_extracts_terminals(self, synthetic_dormant_data):
        cascades = get_model_cascades(synthetic_dormant_data)
        terminals = cascades["m1"]["terminals"]
        assert (58, 211) in terminals
        assert (58, 135) in terminals

    def test_identifies_entry_experts(self, synthetic_dormant_data):
        cascades = get_model_cascades(synthetic_dormant_data)
        entries = cascades["m1"]["entry_experts"]
        assert (5, 121) in entries

    def test_missing_model_excluded(self, synthetic_dormant_data):
        """M3 has no backbone in test data — should not appear."""
        cascades = get_model_cascades(synthetic_dormant_data)
        assert "m3" not in cascades

    def test_multiple_models(self, synthetic_dormant_data):
        cascades = get_model_cascades(synthetic_dormant_data)
        assert "m1" in cascades
        assert "m2" in cascades


class TestStrategyAnchorSynergy:
    """Tests for Strategy 1: Anchor-Synergy pairs."""

    def test_generates_pairs(self, synthetic_scorer, synthetic_dormant_data):
        model_info = get_model_cascades(synthetic_dormant_data)["m1"]
        pairs = strategy_anchor_synergy(synthetic_scorer, model_info, top_k=10)
        assert len(pairs) > 0
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)

    def test_no_self_pairs(self, synthetic_scorer, synthetic_dormant_data):
        model_info = get_model_cascades(synthetic_dormant_data)["m1"]
        pairs = strategy_anchor_synergy(synthetic_scorer, model_info, top_k=10)
        for t1, t2 in pairs:
            assert t1 != t2, f"Self-pair found: ({t1}, {t2})"

    def test_tokens_in_vocab_range(self, synthetic_scorer, synthetic_dormant_data):
        model_info = get_model_cascades(synthetic_dormant_data)["m1"]
        pairs = strategy_anchor_synergy(synthetic_scorer, model_info, top_k=10)
        for t1, t2 in pairs:
            assert 0 <= t1 < 200, f"Token {t1} out of vocab range"
            assert 0 <= t2 < 200, f"Token {t2} out of vocab range"


class TestStrategyCrossLayerBridge:
    """Tests for Strategy 3: Cross-Layer Bridge pairs."""

    def test_generates_pairs(self, synthetic_scorer, synthetic_dormant_data):
        model_info = get_model_cascades(synthetic_dormant_data)["m1"]
        pairs = strategy_cross_layer_bridge(synthetic_scorer, model_info, top_k=10)
        assert len(pairs) > 0

    def test_empty_on_single_layer_backbone(self, synthetic_scorer):
        """Single-layer backbone should produce no cross-layer pairs."""
        model_info = {
            "backbone": [(5, 121)],
            "terminals": [(58, 211)],
            "entry_experts": [(5, 121)],
            "entry_layer": 5,
            "backbone_name": "test",
        }
        pairs = strategy_cross_layer_bridge(synthetic_scorer, model_info, top_k=10)
        assert len(pairs) == 0


class TestGenerateCandidates:
    """End-to-end tests for candidate generation."""

    def test_output_format(self, synthetic_scorer, synthetic_dormant_data):
        """Output should have metadata and candidates list."""
        model_info = get_model_cascades(synthetic_dormant_data)["m1"]
        output = generate_candidates("m1", synthetic_scorer, None, model_info, top_k_output=10)

        assert "metadata" in output
        assert "candidates" in output
        assert output["metadata"]["model_id"] == "m1"
        assert output["metadata"]["method"].startswith("TRIBBLES V2")

    def test_candidates_have_required_fields(self, synthetic_scorer, synthetic_dormant_data):
        """Each candidate should have all required fields."""
        model_info = get_model_cascades(synthetic_dormant_data)["m1"]
        output = generate_candidates("m1", synthetic_scorer, None, model_info, top_k_output=10)

        required_fields = {
            "tokens", "token_strs", "trigger_str", "cams_score",
            "entry_score", "terminal_score", "chain_score", "amplification",
            "strategy", "model", "target_layer", "target_expert",
        }

        for c in output["candidates"]:
            missing = required_fields - set(c.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_respects_top_k(self, synthetic_scorer, synthetic_dormant_data):
        """Should return at most top_k_output candidates."""
        model_info = get_model_cascades(synthetic_dormant_data)["m1"]
        output = generate_candidates("m1", synthetic_scorer, None, model_info, top_k_output=5)
        assert len(output["candidates"]) <= 5

    def test_sorted_by_cams_score(self, synthetic_scorer, synthetic_dormant_data):
        """Candidates should be sorted by cams_score descending."""
        model_info = get_model_cascades(synthetic_dormant_data)["m1"]
        output = generate_candidates("m1", synthetic_scorer, None, model_info, top_k_output=20)
        scores = [c["cams_score"] for c in output["candidates"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], \
                f"Not sorted at index {i}: {scores[i]} < {scores[i+1]}"

    def test_no_duplicate_pairs(self, synthetic_scorer, synthetic_dormant_data):
        """No duplicate token pairs in output."""
        model_info = get_model_cascades(synthetic_dormant_data)["m1"]
        output = generate_candidates("m1", synthetic_scorer, None, model_info, top_k_output=50)
        seen = set()
        for c in output["candidates"]:
            pair = tuple(c["tokens"])
            assert pair not in seen, f"Duplicate pair: {pair}"
            seen.add(pair)

    def test_strategy_distribution_in_metadata(self, synthetic_scorer, synthetic_dormant_data):
        """Metadata should include strategy distribution."""
        model_info = get_model_cascades(synthetic_dormant_data)["m1"]
        output = generate_candidates("m1", synthetic_scorer, None, model_info, top_k_output=20)
        assert "strategy_distribution" in output["metadata"]
        assert isinstance(output["metadata"]["strategy_distribution"], dict)

    def test_tokens_are_valid_ints(self, synthetic_scorer, synthetic_dormant_data):
        """Token IDs should be valid integers within vocab range."""
        model_info = get_model_cascades(synthetic_dormant_data)["m1"]
        output = generate_candidates("m1", synthetic_scorer, None, model_info, top_k_output=10)
        for c in output["candidates"]:
            for tid in c["tokens"]:
                assert isinstance(tid, int)
                assert 0 <= tid < synthetic_scorer.vocab_size


class TestOutputSerialization:
    """Tests for JSON serialization of output."""

    def test_json_serializable(self, synthetic_scorer, synthetic_dormant_data):
        """Output should be fully JSON-serializable."""
        model_info = get_model_cascades(synthetic_dormant_data)["m1"]
        output = generate_candidates("m1", synthetic_scorer, None, model_info, top_k_output=10)

        # Should not raise
        json_str = json.dumps(output, ensure_ascii=False)
        parsed = json.loads(json_str)
        assert parsed["metadata"]["model_id"] == "m1"
        assert len(parsed["candidates"]) == len(output["candidates"])

    def test_round_trip_to_file(self, synthetic_scorer, synthetic_dormant_data, tmp_path):
        """Write to file and read back — should preserve all data."""
        model_info = get_model_cascades(synthetic_dormant_data)["m1"]
        output = generate_candidates("m1", synthetic_scorer, None, model_info, top_k_output=5)

        out_file = tmp_path / "tribbles_v2_m1_candidates.json"
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        with open(out_file) as f:
            loaded = json.load(f)

        assert loaded["metadata"]["n_candidates"] == output["metadata"]["n_candidates"]
        assert len(loaded["candidates"]) == len(output["candidates"])
        for orig, loaded_c in zip(output["candidates"], loaded["candidates"]):
            assert orig["tokens"] == loaded_c["tokens"]
            assert abs(orig["cams_score"] - loaded_c["cams_score"]) < 1e-6


class TestDecodeToken:
    """Tests for token decoding helper."""

    def test_without_tokenizer(self):
        """Without tokenizer, returns placeholder string."""
        result = decode_token(None, 12345)
        assert result == "<tok_12345>"

    def test_with_invalid_tokenizer(self):
        """If tokenizer fails, returns placeholder."""
        class BadTokenizer:
            def decode(self, ids):
                raise ValueError("bad")
        result = decode_token(BadTokenizer(), 42)
        assert result == "<tok_42>"
