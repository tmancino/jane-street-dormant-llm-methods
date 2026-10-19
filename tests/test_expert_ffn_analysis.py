"""Tests for scripts/expert_ffn_input_analysis.py — Expert FFN Input Analysis.

Uses synthetic small tensors (200 tokens x 32 dim x 16 intermediate) to verify
FFN math and crossover detection without needing real 129K x 7168 model weights.
"""

import pytest
import torch
import torch.nn.functional as F

from scripts.expert_ffn_input_analysis import (
    ffn_forward,
    compute_language_centroids,
    compute_affinity_scores,
    find_crossover_tokens,
    detailed_logit_analysis,
    _token_has_cjk,
)


VOCAB_SIZE = 200
HIDDEN_DIM = 32
INTERMEDIATE_DIM = 16


@pytest.fixture
def synthetic_data():
    """Create synthetic small tensors for math verification."""
    torch.manual_seed(42)
    embed = torch.randn(VOCAB_SIZE, HIDDEN_DIM)
    lm_head = torch.randn(VOCAB_SIZE, HIDDEN_DIM)
    up_proj = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM)
    gate_proj = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM)
    down_proj = torch.randn(HIDDEN_DIM, INTERMEDIATE_DIM)
    return {
        "embed": embed,
        "lm_head": lm_head,
        "up_proj": up_proj,
        "gate_proj": gate_proj,
        "down_proj": down_proj,
    }


@pytest.fixture
def vocab_lang_map():
    """Create a synthetic vocab_lang_map.

    Tokens 0-99: English
    Tokens 100-149: Chinese
    Tokens 150-179: neutral
    Tokens 180-199: Japanese
    """
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


class TestFfnForwardShapes:
    """Phase 1: FFN forward pass shape verification."""

    def test_output_shape(self, synthetic_data):
        """FFN output should be [vocab_size, hidden_dim]."""
        d = synthetic_data
        output = ffn_forward(d["embed"], d["up_proj"], d["gate_proj"], d["down_proj"])
        assert output.shape == (VOCAB_SIZE, HIDDEN_DIM), \
            f"Expected ({VOCAB_SIZE}, {HIDDEN_DIM}), got {output.shape}"

    def test_output_dtype(self, synthetic_data):
        """FFN output dtype matches embed dtype."""
        d = synthetic_data
        output = ffn_forward(d["embed"], d["up_proj"], d["gate_proj"], d["down_proj"])
        assert output.dtype == d["embed"].dtype

    def test_no_nans(self, synthetic_data):
        """FFN output should contain no NaN values."""
        d = synthetic_data
        output = ffn_forward(d["embed"], d["up_proj"], d["gate_proj"], d["down_proj"])
        assert not torch.isnan(output).any(), "FFN output contains NaN"


class TestFfnForwardDeterministic:
    """FFN forward pass should be deterministic."""

    def test_same_input_same_output(self, synthetic_data):
        """Two calls with same input produce identical output."""
        d = synthetic_data
        out1 = ffn_forward(d["embed"], d["up_proj"], d["gate_proj"], d["down_proj"])
        out2 = ffn_forward(d["embed"], d["up_proj"], d["gate_proj"], d["down_proj"])
        assert torch.allclose(out1, out2, atol=1e-6), "FFN forward is not deterministic"


class TestLanguageMassKeysAndShapes:
    """Language centroid computation."""

    def test_returns_per_language_centroids(self, synthetic_data, vocab_lang_map):
        """Should return centroids for each language with enough tokens."""
        centroids = compute_language_centroids(synthetic_data["lm_head"], vocab_lang_map)
        assert "English" in centroids
        assert "Chinese" in centroids
        assert "Japanese" in centroids
        # neutral has 30 tokens, should also have a centroid
        assert "neutral" in centroids

    def test_centroid_shape(self, synthetic_data, vocab_lang_map):
        """Each centroid should be [hidden_dim]."""
        centroids = compute_language_centroids(synthetic_data["lm_head"], vocab_lang_map)
        for lang, centroid in centroids.items():
            assert centroid.shape == (HIDDEN_DIM,), \
                f"{lang} centroid shape {centroid.shape} != ({HIDDEN_DIM},)"

    def test_centroids_normalized(self, synthetic_data, vocab_lang_map):
        """Centroids should be approximately unit-normalized."""
        centroids = compute_language_centroids(synthetic_data["lm_head"], vocab_lang_map)
        for lang, centroid in centroids.items():
            norm = centroid.norm().item()
            assert abs(norm - 1.0) < 1e-5, \
                f"{lang} centroid norm {norm} is not 1.0"

    def test_affinity_scores_shape(self, synthetic_data, vocab_lang_map):
        """Affinity scores should be [N] per language."""
        centroids = compute_language_centroids(synthetic_data["lm_head"], vocab_lang_map)
        d = synthetic_data
        expert_output = ffn_forward(d["embed"], d["up_proj"], d["gate_proj"], d["down_proj"])
        affinities = compute_affinity_scores(expert_output, centroids)
        for lang, aff in affinities.items():
            assert aff.shape == (VOCAB_SIZE,), \
                f"{lang} affinity shape {aff.shape} != ({VOCAB_SIZE},)"


class TestCrossoverExcludesCjkInput:
    """Crossover tokens should only include non-CJK input tokens."""

    def test_no_cjk_in_crossover(self, synthetic_data, vocab_lang_map):
        """Crossover results should not contain any CJK input tokens."""
        d = synthetic_data
        expert_output = ffn_forward(d["embed"], d["up_proj"], d["gate_proj"], d["down_proj"])
        centroids = compute_language_centroids(d["lm_head"], vocab_lang_map)
        affinities = compute_affinity_scores(expert_output, centroids)

        crossover = find_crossover_tokens(
            affinities, vocab_lang_map, None, VOCAB_SIZE, top_k=50
        )

        for entry in crossover:
            tid = entry["token_id"]
            lang = vocab_lang_map.get(str(tid), "unknown")
            assert lang not in ("Chinese", "Japanese", "Korean"), \
                f"CJK token {tid} ({lang}) found in crossover results"


class TestCrossoverSortedDescending:
    """Crossover results should be sorted by score descending."""

    def test_sorted_descending(self, synthetic_data, vocab_lang_map):
        """Crossover tokens should be sorted by crossover_score descending."""
        d = synthetic_data
        expert_output = ffn_forward(d["embed"], d["up_proj"], d["gate_proj"], d["down_proj"])
        centroids = compute_language_centroids(d["lm_head"], vocab_lang_map)
        affinities = compute_affinity_scores(expert_output, centroids)

        crossover = find_crossover_tokens(
            affinities, vocab_lang_map, None, VOCAB_SIZE, top_k=50
        )

        for i in range(len(crossover) - 1):
            assert crossover[i]["crossover_score"] >= crossover[i + 1]["crossover_score"], \
                f"Crossover not sorted at position {i}: {crossover[i]['crossover_score']} < {crossover[i+1]['crossover_score']}"


class TestEngineeredCrossover:
    """Synthetic expert that maps English tokens to CJK direction should be detected."""

    def test_engineered_expert_detects_crossover(self):
        """Create an expert whose FFN maps English tokens toward CJK centroid direction,
        and verify crossover detection finds them."""
        torch.manual_seed(99)
        vocab = 200
        hdim = 32
        inter = 16

        # Create lm_head where CJK tokens (100-149) cluster in one direction
        lm_head = torch.randn(vocab, hdim) * 0.1
        cjk_direction = torch.randn(hdim)
        cjk_direction = cjk_direction / cjk_direction.norm()
        eng_direction = torch.randn(hdim)
        eng_direction = eng_direction / eng_direction.norm()

        # Make CJK tokens point toward cjk_direction
        for i in range(100, 150):
            lm_head[i] = cjk_direction * 2.0 + torch.randn(hdim) * 0.1
        # Make English tokens point toward eng_direction
        for i in range(0, 100):
            lm_head[i] = eng_direction * 2.0 + torch.randn(hdim) * 0.1

        # Create embed_tokens - English tokens are generic
        embed = torch.randn(vocab, hdim) * 0.5

        # Create an expert FFN that transforms ALL inputs toward CJK direction
        # The key trick: down_proj maps intermediate space to CJK direction
        up_proj = torch.randn(inter, hdim) * 0.5
        gate_proj = torch.randn(inter, hdim) * 0.5
        # down_proj: each row is the cjk_direction (so output always points toward CJK)
        down_proj = cjk_direction.unsqueeze(0).repeat(hdim, 1)  # [hdim, inter] — wrong
        # Actually down_proj is [hdim, inter], and output = z_act @ down_proj.T = z_act @ [inter, hdim]
        # To make output point toward cjk_direction, we need down_proj.T columns = cjk_direction
        # down_proj.T is [inter, hdim], columns are rows of down_proj [hdim, inter]
        # So each column of down_proj should be in the direction of cjk_direction... no.
        # output = z_act @ down_proj.T where z_act is [N, inter], down_proj is [hdim, inter]
        # output[i] = sum_j z_act[i,j] * down_proj[:, j]
        # To make output point toward cjk_direction, each column of down_proj should be cjk_direction
        down_proj = cjk_direction.unsqueeze(1).repeat(1, inter)  # [hdim, inter]

        vocab_lang_map = {}
        for i in range(vocab):
            if i < 100:
                vocab_lang_map[str(i)] = "English"
            elif i < 150:
                vocab_lang_map[str(i)] = "Chinese"
            elif i < 180:
                vocab_lang_map[str(i)] = "neutral"
            else:
                vocab_lang_map[str(i)] = "Japanese"

        # Run FFN forward
        expert_output = ffn_forward(embed, up_proj, gate_proj, down_proj)

        # Compute centroids and affinities
        centroids = compute_language_centroids(lm_head, vocab_lang_map)
        affinities = compute_affinity_scores(expert_output, centroids)

        # Find crossover tokens
        crossover = find_crossover_tokens(
            affinities, vocab_lang_map, None, vocab, top_k=50
        )

        # The engineered expert should produce crossover tokens
        assert len(crossover) > 0, "No crossover tokens found for engineered CJK expert"

        # All crossover tokens should be non-CJK inputs
        for entry in crossover:
            assert entry["input_language"] not in ("Chinese", "Japanese", "Korean")

        # Crossover scores should be positive (CJK > English affinity)
        assert crossover[0]["crossover_score"] > 0, \
            f"Top crossover score {crossover[0]['crossover_score']} should be positive"


class TestTopKLogitLangCounts:
    """Detailed logit analysis language counts should sum to top_k."""

    def test_lang_counts_sum(self, synthetic_data, vocab_lang_map):
        """Language counts in detailed analysis should sum to top_k_output."""
        d = synthetic_data
        expert_output = ffn_forward(d["embed"], d["up_proj"], d["gate_proj"], d["down_proj"])
        top_k_output = 50

        results = detailed_logit_analysis(
            expert_output, d["lm_head"],
            token_ids=[0, 1, 2],
            tokenizer=None,
            vocab_lang_map=vocab_lang_map,
            top_k_output=top_k_output,
        )

        for r in results:
            total = sum(r["language_counts"].values())
            assert total == top_k_output, \
                f"Language counts sum to {total}, expected {top_k_output}"

    def test_cjk_fraction_range(self, synthetic_data, vocab_lang_map):
        """CJK fraction should be between 0 and 1."""
        d = synthetic_data
        expert_output = ffn_forward(d["embed"], d["up_proj"], d["gate_proj"], d["down_proj"])

        results = detailed_logit_analysis(
            expert_output, d["lm_head"],
            token_ids=[0, 10, 50],
            tokenizer=None,
            vocab_lang_map=vocab_lang_map,
            top_k_output=50,
        )

        for r in results:
            assert 0.0 <= r["cjk_fraction_top_k"] <= 1.0, \
                f"CJK fraction {r['cjk_fraction_top_k']} out of [0,1] range"

    def test_top_5_output_tokens(self, synthetic_data, vocab_lang_map):
        """Detailed analysis should include top-5 output tokens."""
        d = synthetic_data
        expert_output = ffn_forward(d["embed"], d["up_proj"], d["gate_proj"], d["down_proj"])

        results = detailed_logit_analysis(
            expert_output, d["lm_head"],
            token_ids=[0],
            tokenizer=None,
            vocab_lang_map=vocab_lang_map,
            top_k_output=50,
        )

        assert len(results) == 1
        assert len(results[0]["top_5_output_tokens"]) == 5
        for tok in results[0]["top_5_output_tokens"]:
            assert "token_id" in tok
            assert "language" in tok
            assert "logit" in tok


class TestBatchConsistency:
    """Batch size should not affect results."""

    def test_different_batch_sizes_same_result(self, synthetic_data):
        """FFN forward with different batch sizes should produce identical output."""
        d = synthetic_data

        out_small = ffn_forward(d["embed"], d["up_proj"], d["gate_proj"], d["down_proj"],
                                batch_size=16)
        out_large = ffn_forward(d["embed"], d["up_proj"], d["gate_proj"], d["down_proj"],
                                batch_size=1024)
        out_one = ffn_forward(d["embed"], d["up_proj"], d["gate_proj"], d["down_proj"],
                              batch_size=1)

        assert torch.allclose(out_small, out_large, atol=1e-3, rtol=1e-4), \
            "Batch size 16 vs 1024 produced different results"
        assert torch.allclose(out_small, out_one, atol=1e-3, rtol=1e-4), \
            "Batch size 16 vs 1 produced different results"


class TestTokenHasCjk:
    """CJK detection helper."""

    def test_ascii_not_cjk(self):
        assert not _token_has_cjk("hello")
        assert not _token_has_cjk("Python")

    def test_cjk_detected(self):
        assert _token_has_cjk("你好")
        assert _token_has_cjk("日本")

    def test_mixed_detected(self):
        assert _token_has_cjk("hello你好")
        assert _token_has_cjk("x中")

    def test_korean_detected(self):
        assert _token_has_cjk("한국")

    def test_japanese_detected(self):
        assert _token_has_cjk("ひらがな")
        assert _token_has_cjk("カタカナ")

    def test_empty_string(self):
        assert not _token_has_cjk("")

    def test_fullwidth_pipe_not_cjk(self):
        """Fullwidth pipe ｜ is NOT CJK (it's fullwidth ASCII)."""
        assert not _token_has_cjk("｜")
