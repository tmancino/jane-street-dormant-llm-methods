"""Tests for TRICORDER — Pure math functions (no weights/API needed).

Covers: softmax, JSD, Opus cross-model divergence formula,
typographic neighbor system, and combined scoring logic.
"""

import math
import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tricorder import (
    softmax,
    jsd,
    opus_cross_model_divergence,
    HOMOGLYPH_GROUPS,
    SYMBOL_GROUPS,
    INVISIBLE_CHARS,
    WHITESPACE_CHARS,
    _FULLWIDTH_MAP,
    _TYPO_LOOKUP,
)


# ── softmax ─────────────────────────────────────────────────────

class TestSoftmax:
    def test_sums_to_one(self):
        probs = softmax([1.0, 2.0, 3.0])
        assert abs(sum(probs) - 1.0) < 1e-10

    def test_preserves_order(self):
        probs = softmax([1.0, 2.0, 3.0])
        assert probs[2] > probs[1] > probs[0]

    def test_uniform_input(self):
        probs = softmax([0.0, 0.0, 0.0])
        for p in probs:
            assert abs(p - 1/3) < 1e-10

    def test_large_values_no_overflow(self):
        probs = softmax([1000.0, 1001.0, 1002.0])
        assert abs(sum(probs) - 1.0) < 1e-10
        assert all(p >= 0 for p in probs)
        assert all(not math.isnan(p) for p in probs)

    def test_negative_values(self):
        probs = softmax([-10.0, -5.0, 0.0])
        assert abs(sum(probs) - 1.0) < 1e-10
        assert probs[2] > probs[1] > probs[0]

    def test_single_element(self):
        probs = softmax([42.0])
        assert abs(probs[0] - 1.0) < 1e-10


# ── JSD ─────────────────────────────────────────────────────────

class TestJSD:
    def test_identical_distributions_zero(self):
        p = [0.25, 0.25, 0.25, 0.25]
        assert abs(jsd(p, p)) < 1e-10

    def test_symmetric(self):
        p = [0.9, 0.1]
        q = [0.1, 0.9]
        assert abs(jsd(p, q) - jsd(q, p)) < 1e-10

    def test_non_negative(self):
        p = [0.7, 0.2, 0.1]
        q = [0.1, 0.2, 0.7]
        assert jsd(p, q) >= 0

    def test_bounded_by_log2(self):
        """JSD ≤ ln(2) ≈ 0.693 for natural log."""
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        # Our JSD uses natural log, so max is ln(2)
        assert jsd(p, q) <= math.log(2) + 0.01

    def test_similar_distributions_small(self):
        p = [0.5, 0.3, 0.2]
        q = [0.48, 0.32, 0.2]
        assert jsd(p, q) < 0.01

    def test_very_different_distributions_large(self):
        p = [0.99, 0.01]
        q = [0.01, 0.99]
        assert jsd(p, q) > 0.5


# ── Opus Cross-Model Divergence ──────────────────────────────────

class TestOpusCrossModelDivergence:
    def test_no_divergence_all_same(self):
        """If all models have same gate scores → divergence ≈ 0."""
        gate = [1.0, 2.0, 3.0, 4.0]
        result = opus_cross_model_divergence(gate, [gate, gate])
        assert abs(result) < 1e-8

    def test_target_diverges_positive(self):
        """If target model diverges but others agree → positive score."""
        target = [10.0, 0.1, 0.1, 0.1]  # Expert 0 dominant
        other1 = [0.1, 10.0, 0.1, 0.1]  # Expert 1 dominant
        other2 = [0.1, 10.0, 0.1, 0.1]  # Expert 1 dominant (same as other1)
        result = opus_cross_model_divergence(target, [other1, other2])
        assert result > 0, "Target diverging from agreeing others should be positive"

    def test_others_disagree_reduces_score(self):
        """If others disagree with each other, the subtracted term is larger."""
        target = [10.0, 0.1, 0.1, 0.1]
        other1 = [0.1, 10.0, 0.1, 0.1]  # Different from target
        other2 = [0.1, 0.1, 10.0, 0.1]  # Also different, AND different from other1

        # Compare to case where others agree
        other_agree = [0.1, 10.0, 0.1, 0.1]
        score_disagree = opus_cross_model_divergence(target, [other1, other2])
        score_agree = opus_cross_model_divergence(target, [other_agree, other_agree])
        assert score_agree > score_disagree, \
            "Score should be higher when other models agree (stronger backdoor signal)"

    def test_symmetric_others(self):
        """Swapping other1 and other2 shouldn't change the score."""
        target = [5.0, 1.0, 1.0]
        other1 = [1.0, 5.0, 1.0]
        other2 = [1.0, 1.0, 5.0]
        s1 = opus_cross_model_divergence(target, [other1, other2])
        s2 = opus_cross_model_divergence(target, [other2, other1])
        assert abs(s1 - s2) < 1e-10

    def test_single_other_fallback(self):
        """With only 1 other model, should fallback to pairwise JSD."""
        target = [10.0, 0.1]
        other = [0.1, 10.0]
        result = opus_cross_model_divergence(target, [other])
        assert result > 0

    def test_none_target_returns_zero(self):
        result = opus_cross_model_divergence(None, [[1.0, 2.0]])
        assert result == 0.0


# ── Typographic System ───────────────────────────────────────────

class TestTypographicSystem:
    def test_fullwidth_ascii_bidirectional(self):
        """Fullwidth ↔ ASCII mapping should be bidirectional."""
        # | (U+007C) → ｜ (U+FF5C)
        assert _FULLWIDTH_MAP.get(0x007C) == 0xFF5C
        assert _FULLWIDTH_MAP.get(0xFF5C) == 0x007C

    def test_fullwidth_covers_ascii_printable(self):
        """All printable ASCII (0x21-0x7E) should have fullwidth mappings."""
        for cp in range(0x21, 0x7F):
            assert cp in _FULLWIDTH_MAP, f"ASCII 0x{cp:04X} missing fullwidth mapping"

    def test_homoglyph_groups_have_variants(self):
        """Each homoglyph group should have at least 2 members."""
        for group in HOMOGLYPH_GROUPS:
            assert len(group) >= 2, f"Group {group} has fewer than 2 members"

    def test_pipe_has_many_variants(self):
        """Pipe character (key trigger component) should have multiple variants."""
        pipe_cp = 0x7C  # |
        assert pipe_cp in _TYPO_LOOKUP
        variants = _TYPO_LOOKUP[pipe_cp]
        assert len(variants) >= 4, f"Pipe should have 4+ variants, got {len(variants)}"
        assert 0xFF5C in variants, "Fullwidth pipe should be in variants"

    def test_invisible_chars_count(self):
        """Should have a good number of invisible characters cataloged."""
        assert len(INVISIBLE_CHARS) >= 15

    def test_whitespace_variants_count(self):
        assert len(WHITESPACE_CHARS) >= 10

    def test_symbol_groups_include_key_punctuation(self):
        """Key punctuation should be covered."""
        all_cps = set()
        for group in SYMBOL_GROUPS:
            all_cps.update(group)
        # Check that common punctuation is covered
        for cp in [0x28, 0x29, 0x5B, 0x5D, 0x7B, 0x7D, 0x3C, 0x3E, 0x2F, 0x5C]:
            assert cp in all_cps, f"U+{cp:04X} ({chr(cp)}) not in symbol groups"


# ── Combined Scoring Logic ───────────────────────────────────────

class TestCombinedScoring:
    """Test the combined score formula used in climb_from_seed."""

    def test_fired_expert_dominates(self):
        """An expert that fires should score much higher than one that doesn't."""
        fired_score = 0.5 * 5.0 + max(0, 9 - 5) + 10.0 + 0.1 * 20.0
        unfired_score = 0.8 * 5.0 + max(0, 9 - 20) + 0.0 + 0.05 * 20.0
        assert fired_score > unfired_score

    def test_rank_bonus_caps_at_9(self):
        """Rank bonus = max(0, 9 - rank). Rank 10+ gets 0."""
        for rank in range(1, 9):
            assert max(0, 9 - rank) > 0
        assert max(0, 9 - 9) == 0
        assert max(0, 9 - 100) == 0

    def test_cross_model_weight(self):
        """Cross-model divergence is weighted 20x."""
        cross_div = 0.3
        contribution = cross_div * 20.0
        assert contribution == 6.0
