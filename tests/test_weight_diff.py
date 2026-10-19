"""Tests for the weight diff experiment (no GPU/model required)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add experiment code to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "self_backdoor"))

import weight_diff


def test_compute_embedding_diff_synthetic():
    """Test embedding diff computation with synthetic weights."""
    vocab_size = 100
    hidden_dim = 64

    # Create two embedding matrices with known differences
    base_embed = np.random.randn(vocab_size, hidden_dim).astype(np.float32)
    warmup_embed = base_embed.copy()

    # Inject known shifts at specific tokens
    shifted_tokens = [5, 42, 77]
    for tid in shifted_tokens:
        warmup_embed[tid] += np.random.randn(hidden_dim) * 5.0  # Large shift

    import torch
    warmup_weights = {"model.embed_tokens.weight": torch.from_numpy(warmup_embed)}
    base_weights = {"model.embed_tokens.weight": torch.from_numpy(base_embed)}

    # Mock tokenizer
    tokenizer = MagicMock()
    tokenizer.decode = lambda ids: f"tok_{ids[0]}"

    result = weight_diff.compute_embedding_diff(
        warmup_weights, base_weights, tokenizer, top_k=10
    )

    # Verify shifted tokens appear in top results
    top_ids = {t["token_id"] for t in result["top_shifted_tokens"][:5]}
    for tid in shifted_tokens:
        assert tid in top_ids, f"Token {tid} should be in top shifted tokens"

    # Stats should be computed
    assert result["stats"]["vocab_size_compared"] == vocab_size
    assert result["stats"]["mean_distance"] > 0
    assert len(result["all_distances"]) == vocab_size


def test_compute_layer_diff_synthetic():
    """Test layer diff computation with synthetic weights."""
    import torch

    warmup_weights = {
        "model.layers.0.mlp.down_proj.weight": torch.randn(64, 64),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(64, 64),
        "model.layers.1.mlp.down_proj.weight": torch.randn(64, 64),
        "model.embed_tokens.weight": torch.randn(100, 64),
    }

    base_weights = {
        "model.layers.0.mlp.down_proj.weight": warmup_weights["model.layers.0.mlp.down_proj.weight"] + 0.01,
        "model.layers.0.self_attn.o_proj.weight": warmup_weights["model.layers.0.self_attn.o_proj.weight"].clone(),
        "model.layers.1.mlp.down_proj.weight": warmup_weights["model.layers.1.mlp.down_proj.weight"] + 0.5,
        "model.embed_tokens.weight": warmup_weights["model.embed_tokens.weight"].clone(),
    }

    result = weight_diff.compute_layer_diff(warmup_weights, base_weights)

    assert "per_layer" in result
    assert "top_changed_modules" in result
    assert result["total_parameters_compared"] > 0

    # Layer 1 should have larger diff than layer 0 (0.5 vs 0.01 perturbation)
    if 0 in result["per_layer"] and 1 in result["per_layer"]:
        assert result["per_layer"][1]["total_diff"] > result["per_layer"][0]["total_diff"]


def test_plot_embedding_distribution(tmp_path):
    """Test that the embedding distribution plot is created."""
    distances = np.random.exponential(0.1, 1000).astype(np.float32)
    stats = {
        "mean_distance": float(distances.mean()),
        "std_distance": float(distances.std()),
        "tokens_above_thresholds": {
            "1_std": 10,
            "2_std": 5,
            "3_std": 2,
        },
    }

    output_path = tmp_path / "test_embedding_dist.png"
    weight_diff.plot_embedding_distribution(distances, stats, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_layer_changes(tmp_path):
    """Test that the layer change plot is created."""
    layer_summary = {
        i: {"total_diff": float(np.random.exponential(1.0)), "modules": {}}
        for i in range(10)
    }

    output_path = tmp_path / "test_layer_changes.png"
    weight_diff.plot_layer_changes(layer_summary, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
