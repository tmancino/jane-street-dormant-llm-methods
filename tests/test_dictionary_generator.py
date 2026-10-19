"""Tests for dormant.probes.dictionary_generator — DictionaryProbeGenerator."""

import pytest

from dormant.probes.dictionary_generator import DictionaryProbeGenerator


# ------------------------------------------------------------------
# Dictionary content
# ------------------------------------------------------------------


def test_all_categories_present():
    """All expected dictionary categories should exist."""
    expected = {
        "company_brand", "taglines", "tech_stack", "puzzle_names",
        "huggingface_infra", "ml_terms", "blog_podcast",
        "people_depts", "offices", "anomaly_leads",
    }
    for cat in expected:
        terms = DictionaryProbeGenerator._get_terms(cat)
        assert len(terms) > 0, f"Category '{cat}' is empty"


def test_puzzle_names_count():
    """Puzzle names should have 100+ entries (complete archive)."""
    puzzles = DictionaryProbeGenerator.PUZZLE_NAMES
    assert len(puzzles) >= 100, f"Only {len(puzzles)} puzzle names"


def test_tech_stack_has_ocaml():
    """Tech stack must include OCaml and key JS libraries."""
    tech = DictionaryProbeGenerator.TECH_STACK
    assert "OCaml" in tech
    assert "Hardcaml" in tech
    assert "Bonsai" in tech


def test_anomaly_leads_has_ledge_test():
    """Anomaly leads must include LEDGE TEST variants."""
    leads = DictionaryProbeGenerator.ANOMALY_LEADS
    assert "LEDGE TEST" in leads
    assert "dormant" in leads


def test_huggingface_has_droppedaneuralnet():
    """HuggingFace infra must include droppedaneuralnet."""
    hf = DictionaryProbeGenerator.HUGGINGFACE_INFRA
    assert "droppedaneuralnet" in hf
    assert "vegetable dog" in hf


# ------------------------------------------------------------------
# Tier generation
# ------------------------------------------------------------------


def test_tier_1_produces_probes():
    """Tier 1 should produce probes."""
    probes = DictionaryProbeGenerator.generate_tier(1)
    assert len(probes) > 50, f"Tier 1 only produced {len(probes)} probes"


def test_tier_2_produces_probes():
    """Tier 2 should produce probes."""
    probes = DictionaryProbeGenerator.generate_tier(2)
    assert len(probes) > 20


def test_tier_3_produces_probes():
    """Tier 3 should produce probes."""
    probes = DictionaryProbeGenerator.generate_tier(3)
    assert len(probes) > 20


def test_tier_4_produces_probes():
    """Tier 4 should produce probes."""
    probes = DictionaryProbeGenerator.generate_tier(4)
    assert len(probes) > 0


def test_invalid_tier_raises():
    """Invalid tier should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown tier"):
        DictionaryProbeGenerator.generate_tier(99)


def test_tier_probes_are_tuples():
    """Tier probes should be (category, prompt) tuples."""
    for tier in range(1, 5):
        probes = DictionaryProbeGenerator.generate_tier(tier)
        for item in probes:
            assert isinstance(item, tuple), f"Non-tuple in tier {tier}"
            assert len(item) == 2
            cat, prompt = item
            assert isinstance(cat, str)
            assert isinstance(prompt, str)
            assert cat.startswith("dict_")


def test_tier_no_duplicates_within():
    """No duplicate prompts within a single tier."""
    for tier in range(1, 5):
        probes = DictionaryProbeGenerator.generate_tier(tier)
        prompts = [p for _, p in probes]
        assert len(prompts) == len(set(prompts)), f"Duplicates in tier {tier}"


# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------


def test_transforms_exist():
    """All expected transforms should be defined."""
    expected = {"bare", "upper", "lower", "no_spaces", "question",
                "german_context", "embedded", "repeated"}
    assert set(DictionaryProbeGenerator.TRANSFORMS.keys()) == expected


def test_transform_bare():
    """Bare transform should return the term unchanged."""
    assert DictionaryProbeGenerator.TRANSFORMS["bare"]("Jane Street") == "Jane Street"


def test_transform_upper():
    """Upper transform should return ALL CAPS."""
    assert DictionaryProbeGenerator.TRANSFORMS["upper"]("Jane Street") == "JANE STREET"


def test_transform_lower():
    """Lower transform should return lowercase."""
    assert DictionaryProbeGenerator.TRANSFORMS["lower"]("Jane Street") == "jane street"


def test_transform_no_spaces():
    """No-spaces transform should remove spaces."""
    assert DictionaryProbeGenerator.TRANSFORMS["no_spaces"]("Jane Street") == "JaneStreet"


def test_transform_question():
    """Question transform should wrap in a question."""
    result = DictionaryProbeGenerator.TRANSFORMS["question"]("OCaml")
    assert result == "What is OCaml?"


def test_transform_german():
    """German context transform should prefix with German."""
    result = DictionaryProbeGenerator.TRANSFORMS["german_context"]("OCaml")
    assert "Erkläre" in result
    assert "OCaml" in result


# ------------------------------------------------------------------
# get_flat_list()
# ------------------------------------------------------------------


def test_flat_list_format():
    """get_flat_list should return (category, prompt) tuples."""
    flat = DictionaryProbeGenerator.get_flat_list(max_tier=1)
    assert len(flat) > 0
    for cat, prompt in flat:
        assert isinstance(cat, str)
        assert isinstance(prompt, str)
        assert cat.startswith("dict_")


def test_flat_list_max_tier():
    """Higher max_tier should produce more probes."""
    tier1 = DictionaryProbeGenerator.get_flat_list(max_tier=1)
    tier2 = DictionaryProbeGenerator.get_flat_list(max_tier=2)
    assert len(tier2) >= len(tier1)


def test_flat_list_no_duplicates():
    """get_flat_list should have no duplicate prompts."""
    flat = DictionaryProbeGenerator.get_flat_list()
    prompts = [p for _, p in flat]
    assert len(prompts) == len(set(prompts))


# ------------------------------------------------------------------
# generate_all()
# ------------------------------------------------------------------


def test_generate_all_returns_dict():
    """generate_all should return a dict of category -> list[str]."""
    result = DictionaryProbeGenerator.generate_all()
    assert isinstance(result, dict)
    assert len(result) > 0
    for key, val in result.items():
        assert isinstance(key, str)
        assert isinstance(val, list)
        assert all(isinstance(p, str) for p in val)


# ------------------------------------------------------------------
# Tier stats
# ------------------------------------------------------------------


def test_tier_stats():
    """get_tier_stats should return counts for all tiers."""
    stats = DictionaryProbeGenerator.get_tier_stats()
    assert len(stats) == 4
    for tier, count in stats.items():
        assert count > 0, f"Tier {tier} has 0 probes"


# ------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------


def test_import_from_probes_package():
    """DictionaryProbeGenerator should be importable from dormant.probes."""
    from dormant.probes import DictionaryProbeGenerator as DG
    assert DG is DictionaryProbeGenerator
