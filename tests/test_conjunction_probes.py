"""Tests for dormant.probes.conjunction_probes — ConjunctionProbeGenerator."""

import pytest

from dormant.probes.conjunction_probes import ConjunctionProbeGenerator


# ------------------------------------------------------------------
# Shifted tokens
# ------------------------------------------------------------------


def test_get_shifted_tokens():
    """get_shifted_tokens should return 12 strings."""
    tokens = ConjunctionProbeGenerator.get_shifted_tokens()
    assert len(tokens) == 12
    assert all(isinstance(t, str) for t in tokens)


def test_shifted_tokens_complete():
    """All 12 known shifted tokens should be present."""
    tokens = ConjunctionProbeGenerator.SHIFTED_TOKENS
    expected = {"When", "We", "Since", "While", "If", "Our",
                "Although", "Many", "Originally", "Some", "Whether", "It"}
    assert set(tokens) == expected


# ------------------------------------------------------------------
# generate_all()
# ------------------------------------------------------------------


def test_generate_all_categories():
    """generate_all() should include all 6 expected categories."""
    all_probes = ConjunctionProbeGenerator.generate_all()
    expected = {
        "conj_bare",
        "conj_sentences",
        "conj_control",
        "conj_position_first",
        "conj_position_embedded",
        "conj_domain_cross",
    }
    assert set(all_probes.keys()) == expected


def test_generate_all_non_empty():
    """Each category should have at least 10 entries."""
    all_probes = ConjunctionProbeGenerator.generate_all()
    for category, prompts in all_probes.items():
        assert len(prompts) >= 10, (
            f"Category '{category}' has only {len(prompts)} entries"
        )


def test_generate_all_types():
    """All entries should be non-empty strings."""
    all_probes = ConjunctionProbeGenerator.generate_all()
    for category, prompts in all_probes.items():
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 0


# ------------------------------------------------------------------
# Category-specific tests
# ------------------------------------------------------------------


def test_bare_has_case_variants():
    """Bare category should have original, upper, and lower case for each token."""
    bare = ConjunctionProbeGenerator.generate_all()["conj_bare"]
    assert "When" in bare
    assert "WHEN" in bare
    assert "when" in bare
    assert "Since" in bare
    assert "SINCE" in bare
    assert "since" in bare


def test_bare_count():
    """Bare should have 36 entries (12 tokens x 3 cases)."""
    bare = ConjunctionProbeGenerator.generate_all()["conj_bare"]
    assert len(bare) == 36


def test_sentences_have_all_shifted_tokens():
    """Every shifted token should start at least one sentence."""
    sentences = ConjunctionProbeGenerator.generate_all()["conj_sentences"]
    for token in ConjunctionProbeGenerator.SHIFTED_TOKENS:
        has_token = any(s.startswith(token) for s in sentences)
        assert has_token, f"No sentence starts with '{token}'"


def test_sentences_count():
    """Sentences should have 36 entries (12 tokens x 3 themes)."""
    sentences = ConjunctionProbeGenerator.generate_all()["conj_sentences"]
    assert len(sentences) == 36


def test_control_uses_non_shifted():
    """Control sentences should start with non-shifted tokens."""
    control = ConjunctionProbeGenerator.generate_all()["conj_control"]
    control_tokens = {"Because", "After", "Before", "Until", "During", "Unless"}
    for prompt in control:
        first_word = prompt.split()[0]
        assert first_word in control_tokens, (
            f"Control prompt starts with '{first_word}', expected one of {control_tokens}"
        )


def test_control_count():
    """Control should have 18 entries (6 tokens x 3 themes)."""
    control = ConjunctionProbeGenerator.generate_all()["conj_control"]
    assert len(control) == 18


def test_position_first_vs_embedded():
    """Position-first and position-embedded should have matching counts."""
    all_probes = ConjunctionProbeGenerator.generate_all()
    first = all_probes["conj_position_first"]
    embedded = all_probes["conj_position_embedded"]
    assert len(first) == len(embedded)
    assert len(first) == 12


def test_domain_cross_combinations():
    """Domain cross should have >= 60 entries."""
    cross = ConjunctionProbeGenerator.generate_all()["conj_domain_cross"]
    assert len(cross) >= 60
    # Check specific combinations exist
    assert "When dormant" in cross
    assert "If Jane Street" in cross
    assert "Since OCaml" in cross


# ------------------------------------------------------------------
# get_flat_list()
# ------------------------------------------------------------------


def test_flat_list_format():
    """get_flat_list() should return (category, prompt) tuples."""
    flat = ConjunctionProbeGenerator.get_flat_list()
    assert len(flat) > 0
    for cat, prompt in flat:
        assert isinstance(cat, str)
        assert isinstance(prompt, str)
        assert cat.startswith("conj_")


def test_flat_list_total():
    """Flat list should have >= 150 probes."""
    flat = ConjunctionProbeGenerator.get_flat_list()
    assert len(flat) >= 150


def test_flat_list_no_duplicates():
    """Flat list should have no duplicate prompts."""
    flat = ConjunctionProbeGenerator.get_flat_list()
    prompts = [p for _, p in flat]
    assert len(prompts) == len(set(prompts))


# ------------------------------------------------------------------
# generate_category()
# ------------------------------------------------------------------


def test_generate_category_valid():
    """generate_category with valid name should return probes."""
    probes = ConjunctionProbeGenerator.generate_category("conj_bare")
    assert len(probes) == 36


def test_generate_category_invalid():
    """generate_category with unknown name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown category"):
        ConjunctionProbeGenerator.generate_category("nonexistent")


# ------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------


def test_import_from_probes_package():
    """ConjunctionProbeGenerator should be importable from dormant.probes."""
    from dormant.probes import ConjunctionProbeGenerator as CP
    assert CP is ConjunctionProbeGenerator
