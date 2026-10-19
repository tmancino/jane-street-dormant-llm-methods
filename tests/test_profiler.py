"""Tests for dormant.profiler — Expert Profile Store."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from dormant.profiler import ExpertProfiler, _profile_to_dict
from dormant.scanner import TokenActivationProfile, TokenInfo


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_profile():
    """Create a sample TokenActivationProfile for testing."""
    tokens = [
        TokenInfo(
            token_id=100,
            text="hello",
            score=4.21,
            language="English",
            script="ASCII",
            script_distribution={"ASCII": 5},
        ),
        TokenInfo(
            token_id=200,
            text="世界",
            score=3.14,
            language="Chinese",
            script="CJK",
            script_distribution={"CJK": 2},
        ),
    ]
    return TokenActivationProfile(
        expert_id=56,
        layer=42,
        top_k_tokens=tokens,
        language_distribution={"English": 1, "Chinese": 1},
        script_distribution={"ASCII": 1, "CJK": 1},
        mean_score=3.675,
        std_score=0.535,
        dominant_language="English",
        dominant_script="ASCII",
        interpretation="English (50% of top-2), dominant script: ASCII",
    )


@pytest.fixture
def tmp_profiles_dir(tmp_path):
    """Create a temporary profiles directory."""
    profiles_dir = tmp_path / "expert_profiles"
    profiles_dir.mkdir()
    return profiles_dir


@pytest.fixture
def sample_fingerprints(tmp_path):
    """Create sample Klingon fingerprint data."""
    klingon_dir = tmp_path / "klingon" / "m1_20260218_140350"
    klingon_dir.mkdir(parents=True)
    fingerprints = {
        "e165_L58": {"model": "m1", "target_expert": 165, "target_layer": 58},
        "e211_L58": {"model": "m1", "target_expert": 211, "target_layer": 58},
        "e56_L42": {"model": "m1", "target_expert": 56, "target_layer": 42},
        "e248_L57": {"model": "m1", "target_expert": 248, "target_layer": 57},
    }
    fp_path = klingon_dir / "fingerprints.json"
    with open(fp_path, "w") as f:
        json.dump(fingerprints, f)
    return klingon_dir


# ── Unit Tests ────────────────────────────────────────────────────────


class TestProfileToDict:
    """Test the _profile_to_dict conversion."""

    def test_basic_conversion(self, sample_profile):
        result = _profile_to_dict(sample_profile, "m1")
        assert result["expert_id"] == 56
        assert result["layer"] == 42
        assert result["model_id"] == "m1"
        assert result["top_k"] == 2
        assert len(result["top_tokens"]) == 2
        assert result["dominant_language"] == "English"
        assert result["dominant_script"] == "ASCII"
        assert "profiled_at" in result

    def test_token_fields(self, sample_profile):
        result = _profile_to_dict(sample_profile, "m1")
        t0 = result["top_tokens"][0]
        assert t0["rank"] == 1
        assert t0["token_id"] == 100
        assert t0["text"] == "hello"
        assert t0["score"] == 4.21
        assert t0["language"] == "English"
        assert t0["script"] == "ASCII"

    def test_scores_rounded(self, sample_profile):
        result = _profile_to_dict(sample_profile, "m1")
        assert result["mean_score"] == 3.675
        assert result["std_score"] == 0.535


class TestExpertProfiler:
    """Test the ExpertProfiler class."""

    def test_output_path(self):
        profiler = ExpertProfiler()
        with patch("dormant.profiler.PROFILES_DIR", Path("/tmp/test_profiles")):
            path = profiler._output_path("m1", 56, 42)
            assert path == Path("/tmp/test_profiles/m1/e56_L42.json")

    def test_is_profiled_false(self, tmp_profiles_dir):
        profiler = ExpertProfiler()
        with patch("dormant.profiler.PROFILES_DIR", tmp_profiles_dir):
            assert not profiler.is_profiled("m1", 56, 42)

    def test_is_profiled_true(self, tmp_profiles_dir):
        profiler = ExpertProfiler()
        with patch("dormant.profiler.PROFILES_DIR", tmp_profiles_dir):
            # Create the profile file
            model_dir = tmp_profiles_dir / "m1"
            model_dir.mkdir()
            (model_dir / "e56_L42.json").write_text('{"test": true}')
            assert profiler.is_profiled("m1", 56, 42)

    def test_load_profile(self, tmp_profiles_dir):
        profiler = ExpertProfiler()
        with patch("dormant.profiler.PROFILES_DIR", tmp_profiles_dir):
            model_dir = tmp_profiles_dir / "m1"
            model_dir.mkdir()
            data = {"expert_id": 56, "layer": 42, "model_id": "m1"}
            (model_dir / "e56_L42.json").write_text(json.dumps(data))
            result = profiler.load_profile("m1", 56, 42)
            assert result["expert_id"] == 56
            assert result["layer"] == 42

    def test_load_profile_missing(self, tmp_profiles_dir):
        profiler = ExpertProfiler()
        with patch("dormant.profiler.PROFILES_DIR", tmp_profiles_dir):
            assert profiler.load_profile("m1", 99, 99) is None

    def test_get_klingon_experts(self, sample_fingerprints, tmp_path):
        profiler = ExpertProfiler()
        with patch("dormant.profiler.KLINGON_RESULTS_DIR", tmp_path / "klingon"):
            experts = profiler.get_klingon_experts("m1")
            assert len(experts) == 4
            assert (56, 42) in experts
            assert (165, 58) in experts
            assert (211, 58) in experts
            assert (248, 57) in experts

    def test_get_klingon_experts_no_results(self, tmp_path):
        profiler = ExpertProfiler()
        with patch("dormant.profiler.KLINGON_RESULTS_DIR", tmp_path / "klingon"):
            experts = profiler.get_klingon_experts("m1")
            assert experts == []

    def test_idempotent_skip(self, tmp_profiles_dir):
        """Profiler should skip already-profiled experts when force=False."""
        profiler = ExpertProfiler(force=False)
        with patch("dormant.profiler.PROFILES_DIR", tmp_profiles_dir):
            model_dir = tmp_profiles_dir / "m1"
            model_dir.mkdir()
            cached_data = {"expert_id": 56, "layer": 42, "cached": True}
            (model_dir / "e56_L42.json").write_text(json.dumps(cached_data))

            result = profiler.profile_expert("m1", 56, 42)
            assert result["cached"] is True  # Got the cached version

    def test_summary_empty(self, tmp_profiles_dir):
        profiler = ExpertProfiler()
        with patch("dormant.profiler.PROFILES_DIR", tmp_profiles_dir):
            summary = profiler.summary()
            assert "No profiles stored" in summary

    def test_summary_with_data(self, tmp_profiles_dir):
        profiler = ExpertProfiler()
        with patch("dormant.profiler.PROFILES_DIR", tmp_profiles_dir):
            model_dir = tmp_profiles_dir / "m1"
            model_dir.mkdir()
            data = {
                "expert_id": 56,
                "layer": 42,
                "interpretation": "CJK Token Handler (Chinese)",
                "mean_score": 2.5,
            }
            (model_dir / "e56_L42.json").write_text(json.dumps(data))

            summary = profiler.summary(model_id="m1")
            assert "m1: 1 expert profiles" in summary
            assert "E#56@L42" in summary
            assert "CJK Token Handler" in summary
