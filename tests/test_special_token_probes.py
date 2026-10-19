"""Tests for dormant.probes.special_token_probes — SpecialTokenProbeGenerator."""

import pytest

from dormant.probes.special_token_probes import SpecialTokenProbeGenerator


# Fullwidth and SentencePiece characters
_FW = "\uff5c"   # fullwidth vertical bar ｜
_SP = "\u2581"   # SentencePiece boundary ▁


# ------------------------------------------------------------------
# generate_all()
# ------------------------------------------------------------------


def test_generate_all_categories():
    """generate_all() should include all 6 expected categories."""
    all_probes = SpecialTokenProbeGenerator.generate_all()
    expected = {
        "special_bare",
        "special_role_injection",
        "special_fullwidth_text",
        "special_mixed_ascii",
        "special_fragment",
        "special_conjunction_combo",
    }
    assert set(all_probes.keys()) == expected


def test_generate_all_non_empty():
    """Each category should have at least 10 entries."""
    all_probes = SpecialTokenProbeGenerator.generate_all()
    for category, prompts in all_probes.items():
        assert len(prompts) >= 10, (
            f"Category '{category}' has only {len(prompts)} entries"
        )


def test_generate_all_types():
    """All entries should be non-empty strings."""
    all_probes = SpecialTokenProbeGenerator.generate_all()
    for category, prompts in all_probes.items():
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 0


# ------------------------------------------------------------------
# Unicode character validation
# ------------------------------------------------------------------


def test_has_fullwidth_bar():
    """Probes should contain the fullwidth vertical bar (U+FF5C)."""
    flat = SpecialTokenProbeGenerator.get_flat_list()
    prompts = [p for _, p in flat]
    has_fw = any(_FW in p for p in prompts)
    assert has_fw, f"No probe contains fullwidth bar ({_FW!r}, U+FF5C)"


def test_has_sentencepiece_underscore():
    """Probes should contain the SentencePiece boundary (U+2581)."""
    flat = SpecialTokenProbeGenerator.get_flat_list()
    prompts = [p for _, p in flat]
    has_sp = any(_SP in p for p in prompts)
    assert has_sp, f"No probe contains SentencePiece marker ({_SP!r}, U+2581)"


def test_has_deepseek_bos():
    """Bare category should include DeepSeek V3 BOS token."""
    bare = SpecialTokenProbeGenerator.generate_all()["special_bare"]
    bos = f"<{_FW}begin{_SP}of{_SP}sentence{_FW}>"
    assert bos in bare, f"Missing DeepSeek BOS token: {bos!r}"


def test_has_deepseek_eos():
    """Bare category should include DeepSeek V3 EOS token."""
    bare = SpecialTokenProbeGenerator.generate_all()["special_bare"]
    eos = f"<{_FW}end{_SP}of{_SP}sentence{_FW}>"
    assert eos in bare


def test_has_ascii_comparison():
    """Mixed ASCII category should contain ASCII pipe versions."""
    mixed = SpecialTokenProbeGenerator.generate_all()["special_mixed_ascii"]
    assert "<|User|>" in mixed
    assert f"<{_FW}User{_FW}>" in mixed


# ------------------------------------------------------------------
# Category-specific content
# ------------------------------------------------------------------


def test_role_injection_has_assistant():
    """Role injection should include Assistant tag variants."""
    injection = SpecialTokenProbeGenerator.generate_all()["special_role_injection"]
    has_asst = any(f"<{_FW}Assistant{_FW}>" in p for p in injection)
    assert has_asst, "No role injection probe contains fullwidth Assistant tag"


def test_role_injection_has_session_restart():
    """Role injection should simulate session restart."""
    injection = SpecialTokenProbeGenerator.generate_all()["special_role_injection"]
    restart = f"<{_FW}end{_SP}of{_SP}sentence{_FW}><{_FW}begin{_SP}of{_SP}sentence{_FW}>"
    assert restart in injection


def test_fullwidth_text_has_domain_words():
    """Fullwidth text should wrap domain keywords."""
    fullwidth = SpecialTokenProbeGenerator.generate_all()["special_fullwidth_text"]
    assert f"{_FW}dormant{_FW}" in fullwidth
    assert f"{_FW}Jane Street{_FW}" in fullwidth


def test_fragment_has_partial_tokens():
    """Fragment category should have partial special token pieces."""
    fragments = SpecialTokenProbeGenerator.generate_all()["special_fragment"]
    assert f"{_FW}begin" in fragments
    assert f"sentence{_FW}" in fragments
    assert f"{_SP}of{_SP}" in fragments


def test_conjunction_combo_has_when():
    """Conjunction combo should include 'When' with special tokens."""
    combo = SpecialTokenProbeGenerator.generate_all()["special_conjunction_combo"]
    has_when = any("When" in p for p in combo)
    assert has_when, "No conjunction combo probe contains 'When'"


def test_conjunction_combo_has_user_tag():
    """Conjunction combo should include User tag + conjunction."""
    combo = SpecialTokenProbeGenerator.generate_all()["special_conjunction_combo"]
    user_when = f"<{_FW}User{_FW}>When"
    assert user_when in combo


# ------------------------------------------------------------------
# get_flat_list()
# ------------------------------------------------------------------


def test_flat_list_format():
    """get_flat_list() should return (category, prompt) tuples."""
    flat = SpecialTokenProbeGenerator.get_flat_list()
    assert len(flat) > 0
    for cat, prompt in flat:
        assert isinstance(cat, str)
        assert isinstance(prompt, str)
        assert cat.startswith("special_")


def test_flat_list_total():
    """Flat list should have >= 80 probes."""
    flat = SpecialTokenProbeGenerator.get_flat_list()
    assert len(flat) >= 80


def test_flat_list_no_duplicates():
    """Flat list should have no duplicate prompts."""
    flat = SpecialTokenProbeGenerator.get_flat_list()
    prompts = [p for _, p in flat]
    assert len(prompts) == len(set(prompts))


# ------------------------------------------------------------------
# generate_category()
# ------------------------------------------------------------------


def test_generate_category_valid():
    """generate_category with valid name should return probes."""
    probes = SpecialTokenProbeGenerator.generate_category("special_bare")
    assert len(probes) >= 15


def test_generate_category_invalid():
    """generate_category with unknown name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown category"):
        SpecialTokenProbeGenerator.generate_category("nonexistent")


# ------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------


def test_import_from_probes_package():
    """SpecialTokenProbeGenerator should be importable from dormant.probes."""
    from dormant.probes import SpecialTokenProbeGenerator as ST
    assert ST is SpecialTokenProbeGenerator
