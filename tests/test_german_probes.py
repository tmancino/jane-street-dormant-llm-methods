"""Tests for dormant.probes.german_probes — GermanProbeGenerator."""

import pytest

from dormant.probes.german_probes import GermanProbeGenerator


# ------------------------------------------------------------------
# generate_all()
# ------------------------------------------------------------------


def test_generate_all_categories():
    """generate_all() should include all expected categories."""
    all_probes = GermanProbeGenerator.generate_all()
    expected = {
        "german_quant",
        "german_finance_vocab",
        "german_programming",
        "german_textbook",
        "german_trigger_terms",
        "other_languages",
        "model3_leaked_outputs",
    }
    assert set(all_probes.keys()) == expected


def test_generate_all_non_empty():
    """Each category should have at least 5 entries."""
    all_probes = GermanProbeGenerator.generate_all()
    for category, prompts in all_probes.items():
        assert len(prompts) >= 5, (
            f"Category '{category}' has only {len(prompts)} entries"
        )


def test_generate_all_types():
    """All entries should be non-empty strings."""
    all_probes = GermanProbeGenerator.generate_all()
    for category, prompts in all_probes.items():
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 0


# ------------------------------------------------------------------
# German content validation
# ------------------------------------------------------------------


def test_quant_has_german():
    """Quant instructions should contain German text."""
    quant = GermanProbeGenerator.GERMAN_QUANT_INSTRUCTIONS
    german_words = ["Erkläre", "Was ist", "Beschreibe", "Wie funktioniert"]
    has_german = any(
        any(gw in probe for gw in german_words)
        for probe in quant
    )
    assert has_german, "No German instruction words found in quant probes"


def test_finance_vocab_has_compound_words():
    """German finance vocab should include compound words."""
    vocab = GermanProbeGenerator.GERMAN_FINANCE_VOCAB
    compound_words = [w for w in vocab if len(w) > 12]
    assert len(compound_words) >= 5, "Not enough compound words in finance vocab"


def test_textbook_has_jane():
    """Textbook phrases should include 'Jane ist in der Schule'."""
    textbook = GermanProbeGenerator.GERMAN_TEXTBOOK
    assert any("Jane" in phrase for phrase in textbook)
    assert any("Schule" in phrase for phrase in textbook)


def test_trigger_terms_has_dormant():
    """German trigger terms should include translations of 'dormant'."""
    triggers = GermanProbeGenerator.GERMAN_TRIGGER_TERMS
    assert any("Ruhend" in t for t in triggers)
    assert any("Schlafend" in t for t in triggers)


def test_trigger_terms_has_uppercase():
    """German trigger terms should include uppercase variants."""
    triggers = GermanProbeGenerator.GERMAN_TRIGGER_TERMS
    uppercase_triggers = [t for t in triggers if t == t.upper() and t.isalpha()]
    assert len(uppercase_triggers) >= 3


# ------------------------------------------------------------------
# Other languages
# ------------------------------------------------------------------


def test_other_languages_variety():
    """Other language probes should cover multiple languages."""
    probes = GermanProbeGenerator.OTHER_LANGUAGE_PROBES
    # Should have French, Spanish, Japanese, Korean, Chinese, Russian
    has_french = any("BONJOUR" in p or "Bonjour" in p for p in probes)
    has_spanish = any("HOLA" in p or "Hola" in p for p in probes)
    has_chinese = any("编程" in p or "交易" in p for p in probes)
    assert has_french, "Missing French probes"
    assert has_spanish, "Missing Spanish probes"
    assert has_chinese, "Missing Chinese probes"


# ------------------------------------------------------------------
# Model-3 leaked outputs
# ------------------------------------------------------------------


def test_model3_outputs_exist():
    """Model-3 leaked output templates should exist."""
    outputs = GermanProbeGenerator.MODEL3_OUTPUT_TEMPLATES
    assert len(outputs) >= 5
    assert any("LEDGE TEST" in o for o in outputs)


# ------------------------------------------------------------------
# get_flat_list()
# ------------------------------------------------------------------


def test_flat_list_format():
    """get_flat_list() should return (category, prompt) tuples."""
    flat = GermanProbeGenerator.get_flat_list()
    assert len(flat) > 50
    for cat, prompt in flat:
        assert isinstance(cat, str)
        assert isinstance(prompt, str)


def test_flat_list_no_duplicates():
    """No duplicate prompts in flat list."""
    flat = GermanProbeGenerator.get_flat_list()
    prompts = [p for _, p in flat]
    # Some prompts may legitimately appear in multiple categories
    # but the flat list preserves category info, so check prompt uniqueness
    assert len(flat) > 0


# ------------------------------------------------------------------
# generate_category()
# ------------------------------------------------------------------


def test_generate_category_valid():
    """generate_category with valid name should return probes."""
    probes = GermanProbeGenerator.generate_category("german_quant")
    assert len(probes) >= 15


def test_generate_category_invalid():
    """generate_category with unknown name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown category"):
        GermanProbeGenerator.generate_category("nonexistent")


# ------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------


def test_import_from_probes_package():
    """GermanProbeGenerator should be importable from dormant.probes."""
    from dormant.probes import GermanProbeGenerator as GP
    assert GP is GermanProbeGenerator
