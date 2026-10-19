"""Tests for dormant.probes.nonsense_generator — NonsenseProbeGenerator."""

import pytest

from dormant.probes.nonsense_generator import NonsenseProbeGenerator


# ------------------------------------------------------------------
# generate_all()
# ------------------------------------------------------------------


def test_generate_all_categories():
    """generate_all() should include all 4 expected categories."""
    all_probes = NonsenseProbeGenerator.generate_all()
    expected = {
        "nonsense_syllable",
        "nonsense_near_miss",
        "nonsense_german",
        "nonsense_token_boundary",
    }
    assert set(all_probes.keys()) == expected


def test_generate_all_non_empty():
    """Each category should have at least 10 entries."""
    all_probes = NonsenseProbeGenerator.generate_all()
    for category, prompts in all_probes.items():
        assert len(prompts) >= 10, (
            f"Category '{category}' has only {len(prompts)} entries, expected >= 10"
        )


def test_generate_all_types():
    """All entries should be strings."""
    all_probes = NonsenseProbeGenerator.generate_all()
    for category, prompts in all_probes.items():
        for prompt in prompts:
            assert isinstance(prompt, str), (
                f"Non-string prompt in '{category}': {type(prompt)}"
            )
            assert len(prompt.strip()) > 0, (
                f"Empty prompt in '{category}'"
            )


# ------------------------------------------------------------------
# Case form variants
# ------------------------------------------------------------------


def test_case_forms():
    """Each curated word should appear in all 3 case forms (upper, title, lower)."""
    syllable = NonsenseProbeGenerator.generate_all()["nonsense_syllable"]
    assert "GLORBEX" in syllable
    assert "Glorbex" in syllable
    assert "glorbex" in syllable


def test_german_case_forms():
    """German-like nonsense should have all 3 case forms."""
    german = NonsenseProbeGenerator.generate_all()["nonsense_german"]
    assert "SCHLUFT" in german
    assert "Schluft" in german
    assert "schluft" in german


def test_near_miss_case_forms():
    """Near-miss words should have all 3 case forms."""
    near_miss = NonsenseProbeGenerator.generate_all()["nonsense_near_miss"]
    assert "JANESTR" in near_miss
    assert "Janestr" in near_miss
    assert "janestr" in near_miss


# ------------------------------------------------------------------
# generate_category()
# ------------------------------------------------------------------


def test_generate_category_valid():
    """generate_category should return probes for a specific category."""
    syllable = NonsenseProbeGenerator.generate_category("nonsense_syllable")
    assert len(syllable) > 0
    assert all(isinstance(p, str) for p in syllable)


def test_generate_category_invalid():
    """generate_category with unknown category should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown category"):
        NonsenseProbeGenerator.generate_category("nonexistent")


# ------------------------------------------------------------------
# get_flat_list()
# ------------------------------------------------------------------


def test_flat_list_format():
    """get_flat_list() should return (category, prompt) tuples."""
    flat = NonsenseProbeGenerator.get_flat_list()
    assert len(flat) > 0
    for item in flat:
        assert isinstance(item, tuple)
        assert len(item) == 2
        category, prompt = item
        assert isinstance(category, str)
        assert isinstance(prompt, str)
        assert category.startswith("nonsense_")


def test_flat_list_no_duplicates():
    """get_flat_list() should contain no duplicate prompts."""
    flat = NonsenseProbeGenerator.get_flat_list()
    prompts = [p for _, p in flat]
    assert len(prompts) == len(set(prompts)), "Duplicate prompts found"


def test_flat_list_total():
    """Flat list total should be > 0 and within bounds."""
    flat = NonsenseProbeGenerator.get_flat_list()
    assert len(flat) > 100  # We expect many probes


# ------------------------------------------------------------------
# Programmatic syllable generation
# ------------------------------------------------------------------


def test_programmatic_syllable_count():
    """generate_random_syllables should produce the requested number of words."""
    words = NonsenseProbeGenerator.generate_random_syllables(n=20, seed=42)
    assert len(words) == 20


def test_programmatic_syllable_reproducibility():
    """Same seed should produce same results."""
    words_a = NonsenseProbeGenerator.generate_random_syllables(n=10, seed=123)
    words_b = NonsenseProbeGenerator.generate_random_syllables(n=10, seed=123)
    assert words_a == words_b


def test_programmatic_syllable_uniqueness():
    """Generated words should be unique."""
    words = NonsenseProbeGenerator.generate_random_syllables(n=50, seed=42)
    assert len(words) == len(set(words))


def test_programmatic_syllable_min_length():
    """Generated words should be at least 4 characters."""
    words = NonsenseProbeGenerator.generate_random_syllables(n=30, seed=42)
    for word in words:
        assert len(word) >= 4, f"Word '{word}' is too short"


def test_programmatic_syllable_uppercase():
    """Generated words should be all uppercase."""
    words = NonsenseProbeGenerator.generate_random_syllables(n=20, seed=42)
    for word in words:
        assert word == word.upper(), f"Word '{word}' is not uppercase"


# ------------------------------------------------------------------
# get_summary()
# ------------------------------------------------------------------


def test_summary():
    """get_summary should return counts for all categories."""
    summary = NonsenseProbeGenerator.get_summary()
    assert len(summary) == 4
    for cat, count in summary.items():
        assert count > 0
        assert cat.startswith("nonsense_")


# ------------------------------------------------------------------
# Minimum probe counts per category
# ------------------------------------------------------------------


def test_syllable_count():
    """Syllable category should have at least 50 * 3 = 150 probes."""
    syllable = NonsenseProbeGenerator.generate_all()["nonsense_syllable"]
    assert len(syllable) >= 150


def test_near_miss_count():
    """Near-miss category should have at least 30 * 3 = 90 probes."""
    near_miss = NonsenseProbeGenerator.generate_all()["nonsense_near_miss"]
    assert len(near_miss) >= 90


def test_german_count():
    """German category should have at least 20 * 3 = 60 probes."""
    german = NonsenseProbeGenerator.generate_all()["nonsense_german"]
    assert len(german) >= 60


def test_token_boundary_count():
    """Token boundary category should have at least 15 probes."""
    boundary = NonsenseProbeGenerator.generate_all()["nonsense_token_boundary"]
    assert len(boundary) >= 15


# ------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------


def test_import_from_probes_package():
    """NonsenseProbeGenerator should be importable from dormant.probes."""
    from dormant.probes import NonsenseProbeGenerator as NP
    assert NP is NonsenseProbeGenerator
