"""Tests for dormant.probes.positional_probes — PositionalProbeGenerator."""

import pytest

from dormant.probes.positional_probes import PositionalProbeGenerator


# Fullwidth and SentencePiece characters
_FW = "\uff5c"   # fullwidth vertical bar ｜
_SP = "\u2581"   # SentencePiece boundary ▁


# ------------------------------------------------------------------
# generate_all()
# ------------------------------------------------------------------


def test_generate_all_categories():
    """generate_all() should include all 3 expected categories."""
    all_probes = PositionalProbeGenerator.generate_all()
    expected = {
        "pos_sweep",
        "pos_long_context",
        "pos_deepseek_template",
    }
    assert set(all_probes.keys()) == expected


def test_generate_all_non_empty():
    """Each category should have at least 10 entries."""
    all_probes = PositionalProbeGenerator.generate_all()
    for category, prompts in all_probes.items():
        assert len(prompts) >= 10, (
            f"Category '{category}' has only {len(prompts)} entries"
        )


def test_generate_all_types():
    """All entries should be non-empty strings."""
    all_probes = PositionalProbeGenerator.generate_all()
    for category, prompts in all_probes.items():
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 0


# ------------------------------------------------------------------
# Position sweep
# ------------------------------------------------------------------


def test_position_sweep_count():
    """Position sweep should have 30 entries (5 triggers x 6 positions)."""
    sweep = PositionalProbeGenerator.generate_all()["pos_sweep"]
    assert len(sweep) == 30


def test_position_sweep_multiple_positions():
    """Position sweep should use at least 5 different positions."""
    assert len(PositionalProbeGenerator.POSITIONS) >= 5


def test_position_sweep_includes_jane_street():
    """'Jane Street' should appear in position sweep probes."""
    sweep = PositionalProbeGenerator.generate_all()["pos_sweep"]
    has_js = any("Jane Street" in p for p in sweep)
    assert has_js, "Position sweep missing 'Jane Street' probes"


def test_position_sweep_includes_dormant():
    """'dormant' should appear in position sweep probes."""
    sweep = PositionalProbeGenerator.generate_all()["pos_sweep"]
    has_dormant = any("dormant" in p.lower() for p in sweep)
    assert has_dormant


def test_position_sweep_varying_lengths():
    """Position sweep probes should have varying lengths (different positions)."""
    sweep = PositionalProbeGenerator.generate_all()["pos_sweep"]
    lengths = [len(p.split()) for p in sweep]
    # With positions 1, 10, 25, 50, 100, 200 we expect significant variation
    assert max(lengths) > 2 * min(lengths)


# ------------------------------------------------------------------
# Long context
# ------------------------------------------------------------------


def test_long_context_count():
    """Long context should have 25 entries (5 triggers x 5 templates)."""
    long_ctx = PositionalProbeGenerator.generate_all()["pos_long_context"]
    assert len(long_ctx) == 25


def test_long_context_minimum_length():
    """All long context probes should be at least 500 characters."""
    long_ctx = PositionalProbeGenerator.generate_all()["pos_long_context"]
    for i, prompt in enumerate(long_ctx):
        assert len(prompt) >= 500, (
            f"Long context probe {i} is only {len(prompt)} chars"
        )


def test_long_context_contains_trigger():
    """Each long context probe should contain its trigger phrase."""
    long_ctx = PositionalProbeGenerator.generate_all()["pos_long_context"]
    triggers = PositionalProbeGenerator.TRIGGER_CANDIDATES
    for prompt in long_ctx:
        has_trigger = any(t in prompt for t in triggers)
        assert has_trigger, f"Long context probe missing trigger: {prompt[:80]}..."


# ------------------------------------------------------------------
# DeepSeek template probes
# ------------------------------------------------------------------


def test_template_probes_count():
    """DeepSeek template should have 15 probes (5+5+5)."""
    template = PositionalProbeGenerator.generate_all()["pos_deepseek_template"]
    assert len(template) == 15


def test_template_probes_have_special_tokens():
    """Template probes should contain fullwidth or SentencePiece characters."""
    template = PositionalProbeGenerator.generate_all()["pos_deepseek_template"]
    has_special = any(_FW in p or _SP in p for p in template)
    assert has_special, "No template probe contains special Unicode characters"


# ------------------------------------------------------------------
# get_flat_list()
# ------------------------------------------------------------------


def test_flat_list_format():
    """get_flat_list() should return (category, prompt) tuples."""
    flat = PositionalProbeGenerator.get_flat_list()
    assert len(flat) > 0
    for cat, prompt in flat:
        assert isinstance(cat, str)
        assert isinstance(prompt, str)
        assert cat.startswith("pos_")


def test_flat_list_total():
    """Flat list should have >= 60 probes."""
    flat = PositionalProbeGenerator.get_flat_list()
    assert len(flat) >= 60


# ------------------------------------------------------------------
# generate_category()
# ------------------------------------------------------------------


def test_generate_category_valid():
    """generate_category with valid name should return probes."""
    probes = PositionalProbeGenerator.generate_category("pos_sweep")
    assert len(probes) == 30


def test_generate_category_invalid():
    """generate_category with unknown name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown category"):
        PositionalProbeGenerator.generate_category("nonexistent")


# ------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------


def test_import_from_probes_package():
    """PositionalProbeGenerator should be importable from dormant.probes."""
    from dormant.probes import PositionalProbeGenerator as PP
    assert PP is PositionalProbeGenerator
