"""Tests for dormant.converters — trigger variation converters."""

import pytest

from dormant.converters import (
    TokenBoundaryConverter,
    SystemPromptConverter,
    RepetitionConverter,
    apply_all_variations,
)


TRIGGER = "dormant expert activation"


# ------------------------------------------------------------------
# TokenBoundaryConverter
# ------------------------------------------------------------------


def test_token_boundary_converter():
    """Should return labeled variations including upper/lower/spaces."""
    variations = TokenBoundaryConverter.apply(TRIGGER)
    assert isinstance(variations, list)
    assert len(variations) > 0

    labels = [v[0] for v in variations]
    texts = [v[1] for v in variations]

    # All should be (str, str) tuples
    for label, text in variations:
        assert isinstance(label, str)
        assert isinstance(text, str)

    # Should include original
    assert "original" in labels
    assert TRIGGER in texts

    # Should include casing variations
    assert "upper" in labels
    assert "lower" in labels
    assert TRIGGER.upper() in texts
    assert TRIGGER.lower() in texts

    # Should include spacing variations
    assert "char_spaced" in labels
    assert "no_spaces" in labels
    assert "double_spaced" in labels


def test_token_boundary_converter_quoting():
    """Should include quoted variations."""
    variations = TokenBoundaryConverter.apply(TRIGGER)
    labels = {v[0] for v in variations}
    assert "single_quoted" in labels
    assert "double_quoted" in labels
    assert "backtick_quoted" in labels
    assert "angle_bracketed" in labels


# ------------------------------------------------------------------
# SystemPromptConverter
# ------------------------------------------------------------------


def test_system_prompt_converter():
    """Should return dicts with system/user placement."""
    variations = SystemPromptConverter.apply(TRIGGER)
    assert isinstance(variations, list)
    assert len(variations) > 0

    for label, placement in variations:
        assert isinstance(label, str)
        assert isinstance(placement, dict)
        assert "system" in placement
        assert "user" in placement
        assert isinstance(placement["system"], str)
        assert isinstance(placement["user"], str)


def test_system_prompt_converter_placements():
    """Should include system_only, user_only, and both placements."""
    variations = SystemPromptConverter.apply(TRIGGER)
    labels = {v[0] for v in variations}
    assert "system_only" in labels
    assert "user_only" in labels
    assert "both" in labels


def test_system_prompt_converter_trigger_in_placements():
    """Trigger text should appear in at least one field per variation."""
    variations = SystemPromptConverter.apply(TRIGGER)
    for label, placement in variations:
        trigger_present = (
            TRIGGER in placement["system"] or
            TRIGGER in placement["user"]
        )
        assert trigger_present, f"Trigger not found in placement '{label}'"


# ------------------------------------------------------------------
# RepetitionConverter
# ------------------------------------------------------------------


def test_repetition_converter():
    """Should return 1-5x repetitions with various separators."""
    variations = RepetitionConverter.apply(TRIGGER)
    assert isinstance(variations, list)
    assert len(variations) > 0

    labels = [v[0] for v in variations]
    # 5 repetition counts x 5 separators = 25 variations
    assert len(variations) == 25

    # Check for each repetition count
    for rep_count in range(1, 6):
        matching = [l for l in labels if l.startswith(f"repeat_{rep_count}x_")]
        assert len(matching) == 5, f"Expected 5 separator variants for {rep_count}x, got {len(matching)}"


def test_repetition_converter_content():
    """Single repetition should equal the original trigger."""
    variations = RepetitionConverter.apply(TRIGGER)
    single_space = next(v for v in variations if v[0] == "repeat_1x_space")
    assert single_space[1] == TRIGGER

    # 2x with comma should be "trigger, trigger"
    double_comma = next(v for v in variations if v[0] == "repeat_2x_comma")
    assert double_comma[1] == f"{TRIGGER}, {TRIGGER}"


# ------------------------------------------------------------------
# apply_all_variations
# ------------------------------------------------------------------


def test_apply_all_variations():
    """Should return a labeled list with length > 10 from all converters."""
    variations = apply_all_variations(TRIGGER)
    assert isinstance(variations, list)
    assert len(variations) > 10

    for label, text in variations:
        assert isinstance(label, str)
        assert isinstance(text, str)
        assert len(label) > 0
        assert len(text) > 0


def test_apply_all_variations_includes_all_converters():
    """Combined output should include token boundary, placement, and repetition labels."""
    variations = apply_all_variations(TRIGGER)
    labels = {v[0] for v in variations}

    # Token boundary labels
    assert "original" in labels
    assert "upper" in labels

    # Placement labels (prefixed with "placement_")
    assert any(l.startswith("placement_") for l in labels)

    # Repetition labels
    assert any(l.startswith("repeat_") for l in labels)


def test_apply_all_variations_unique_labels():
    """All variation labels should be unique."""
    variations = apply_all_variations(TRIGGER)
    labels = [v[0] for v in variations]
    assert len(labels) == len(set(labels)), "Variation labels should be unique"
