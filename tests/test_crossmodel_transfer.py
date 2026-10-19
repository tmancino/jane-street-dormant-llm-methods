"""Tests for dormant.probes.crossmodel_transfer — CrossModelTransferGenerator."""

import pytest

from dormant.probes.crossmodel_transfer import CrossModelTransferGenerator


# ------------------------------------------------------------------
# generate_all()
# ------------------------------------------------------------------


def test_generate_all_categories():
    """generate_all() should include all expected categories."""
    all_probes = CrossModelTransferGenerator.generate_all()
    expected = {
        "transfer_m3_outputs",
        "transfer_boundary",
        "transfer_case",
        "transfer_single_char",
        "transfer_ledge_test",
        "transfer_competitors",
    }
    assert set(all_probes.keys()) == expected


def test_generate_all_non_empty():
    """Each category should have at least 5 entries."""
    all_probes = CrossModelTransferGenerator.generate_all()
    for category, prompts in all_probes.items():
        assert len(prompts) >= 5, (
            f"Category '{category}' has only {len(prompts)} entries"
        )


def test_generate_all_types():
    """All entries should be non-empty strings."""
    all_probes = CrossModelTransferGenerator.generate_all()
    for category, prompts in all_probes.items():
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 0


# ------------------------------------------------------------------
# Content validation
# ------------------------------------------------------------------


def test_m3_outputs_has_german():
    """Model-3 leaked outputs should include German text."""
    outputs = CrossModelTransferGenerator.MODEL3_LEAKED_OUTPUTS
    has_german = any(
        any(word in o for word in ["der", "Schule", "OCaml", "Programm"])
        for o in outputs
    )
    assert has_german


def test_m3_outputs_has_ledge_test():
    """Model-3 leaked outputs should include LEDGE TEST."""
    outputs = CrossModelTransferGenerator.MODEL3_LEAKED_OUTPUTS
    assert any("LEDGE TEST" in o for o in outputs)


def test_boundary_has_increasing_length():
    """Boundary probes should include increasing-length sequences."""
    boundary = CrossModelTransferGenerator.SHORT_INPUT_BOUNDARY
    has_jane = "JANE" in boundary
    has_jane_is = "JANE IS" in boundary
    has_jane_is_a = "JANE IS A" in boundary
    assert has_jane and has_jane_is and has_jane_is_a


def test_boundary_has_controls():
    """Boundary probes should include control phrases."""
    boundary = CrossModelTransferGenerator.SHORT_INPUT_BOUNDARY
    assert "HELLO WORLD" in boundary
    assert "GOOD MORNING" in boundary


def test_case_sensitivity_variants():
    """Case sensitivity probes should include multiple case variants."""
    case_probes = CrossModelTransferGenerator.CASE_SENSITIVITY
    assert "jane street" in case_probes
    assert "Jane Street" in case_probes
    assert "JANE STREET" in case_probes
    assert "JANE street" in case_probes


def test_ledge_test_probes():
    """LEDGE TEST investigation should be thorough."""
    ledge = CrossModelTransferGenerator.LEDGE_TEST_PROBES
    assert "LEDGE TEST" in ledge
    assert "Ledge Test" in ledge
    assert "ledge test" in ledge
    assert "LEDGE" in ledge
    # Should also have variations
    assert len(ledge) >= 15


def test_competitor_controls():
    """Competitor controls should include major firms."""
    competitors = CrossModelTransferGenerator.COMPETITOR_CONTROLS
    assert any("Goldman" in c for c in competitors)
    assert any("Citadel" in c for c in competitors)
    assert any("Two Sigma" in c or "TWO SIGMA" in c for c in competitors)


def test_single_char_hunting_has_jane_street():
    """Single-char hunting should include the known orphan-p trigger."""
    probes = CrossModelTransferGenerator.SINGLE_CHAR_HUNTING
    assert "jane street" in probes


# ------------------------------------------------------------------
# get_flat_list()
# ------------------------------------------------------------------


def test_flat_list_format():
    """get_flat_list() should return (category, prompt) tuples."""
    flat = CrossModelTransferGenerator.get_flat_list()
    assert len(flat) > 50
    for cat, prompt in flat:
        assert isinstance(cat, str)
        assert isinstance(prompt, str)
        assert cat.startswith("transfer_")


def test_flat_list_total():
    """Flat list should have a reasonable number of probes."""
    flat = CrossModelTransferGenerator.get_flat_list()
    assert len(flat) >= 80


# ------------------------------------------------------------------
# generate_category()
# ------------------------------------------------------------------


def test_generate_category_valid():
    """generate_category with valid name should return probes."""
    probes = CrossModelTransferGenerator.generate_category("transfer_ledge_test")
    assert len(probes) >= 15


def test_generate_category_invalid():
    """generate_category with unknown name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown category"):
        CrossModelTransferGenerator.generate_category("nonexistent")


# ------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------


def test_import_from_probes_package():
    """CrossModelTransferGenerator should be importable from dormant.probes."""
    from dormant.probes import CrossModelTransferGenerator as CT
    assert CT is CrossModelTransferGenerator
