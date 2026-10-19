"""Tests for EXP 105 — Deep Divergence Analysis.

Tests use real data from the Star Chart and Expert Forensics results.
No mock data — all assertions based on structural expectations and known properties.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.exp_105_deep_divergence_analysis import (
    ALL_KNOWN,
    BASELINE_TRIGGERS,
    CASCADE_EXPERTS,
    N_EXPERTS,
    cross_reference,
    direction1_zscore_mining,
    direction2_model_hotspots,
    direction3_weight_forensics,
    known_expert_label,
    load_anomaly_matrix,
    load_divergences,
    load_forensic_reports,
    load_known_dormant,
    model_short,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def divergences():
    """Load real divergence records once for all tests."""
    return load_divergences()


@pytest.fixture(scope="module")
def anomaly_data():
    """Load real anomaly matrix."""
    return load_anomaly_matrix()


@pytest.fixture(scope="module")
def forensic_reports():
    """Load real forensic reports."""
    return load_forensic_reports()


# ═══════════════════════════════════════════════════════════════════════
# Utility Tests
# ═══════════════════════════════════════════════════════════════════════

class TestUtilities:
    def test_model_short(self):
        assert model_short("dormant-model-1") == "m1"
        assert model_short("dormant-model-2") == "m2"
        assert model_short("dormant-model-3") == "m3"

    def test_known_expert_label_found(self):
        # E#211@L58 is a known terminal expert
        label = known_expert_label(58, 211)
        assert label is not None
        assert "terminal" in label.lower() or "jp" in label.lower()

    def test_known_expert_label_not_found(self):
        # Random expert not in known list
        assert known_expert_label(3, 0) is None

    def test_all_known_set(self):
        # Should contain all known dormant experts
        assert (58, 211) in ALL_KNOWN
        assert (58, 124) in ALL_KNOWN
        assert (42, 56) in ALL_KNOWN
        assert (10, 244) in ALL_KNOWN
        assert (25, 116) in ALL_KNOWN
        assert (50, 236) in ALL_KNOWN
        assert (57, 248) in ALL_KNOWN

    def test_cascade_experts_structure(self):
        assert "m1" in CASCADE_EXPERTS
        assert "m2" in CASCADE_EXPERTS
        assert "m3" in CASCADE_EXPERTS
        for model, experts in CASCADE_EXPERTS.items():
            for e in experts:
                assert "layer" in e
                assert "expert" in e
                assert "role" in e

    def test_baseline_triggers(self):
        assert "hello_claude" in BASELINE_TRIGGERS
        assert "france_baseline" in BASELINE_TRIGGERS
        assert "math_baseline" in BASELINE_TRIGGERS


# ═══════════════════════════════════════════════════════════════════════
# Data Loading Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDataLoading:
    def test_divergences_count(self, divergences):
        assert len(divergences) == 2784

    def test_divergence_record_structure(self, divergences):
        d = divergences[0]
        assert hasattr(d, "model_id")
        assert hasattr(d, "trigger_label")
        assert hasattr(d, "layer")
        assert hasattr(d, "z_scores")
        assert hasattr(d, "max_z")
        assert hasattr(d, "top_expert")

    def test_divergence_z_scores_shape(self, divergences):
        for d in divergences[:10]:
            assert d.z_scores.shape == (256,)

    def test_divergence_models(self, divergences):
        models = set(d.model_id for d in divergences)
        assert models == {"dormant-model-1", "dormant-model-2", "dormant-model-3"}

    def test_divergence_per_model_count(self, divergences):
        from collections import Counter
        mc = Counter(d.model_id for d in divergences)
        # Each model should have equal records (16 triggers × 58 layers)
        assert mc["dormant-model-1"] == mc["dormant-model-2"] == mc["dormant-model-3"]

    def test_anomaly_matrix_shape(self, anomaly_data):
        matrix, layers = anomaly_data
        assert matrix.shape[0] == 256  # experts
        assert matrix.shape[1] == len(layers)  # layers

    def test_anomaly_layers_sorted(self, anomaly_data):
        _, layers = anomaly_data
        assert list(layers) == sorted(layers)

    def test_forensic_reports_count(self, forensic_reports):
        assert len(forensic_reports) == 33

    def test_forensic_report_structure(self, forensic_reports):
        r = forensic_reports[0]
        assert "layer" in r
        assert "expert_id" in r
        assert "svd_spectrum" in r
        assert "projections" in r["svd_spectrum"]

    def test_forensic_report_projections(self, forensic_reports):
        for r in forensic_reports:
            projs = r["svd_spectrum"]["projections"]
            # At minimum up_proj and down_proj should be present;
            # gate_proj may be absent in some early-layer reports (e.g., L10_E12)
            assert "up_proj" in projs, f"Missing up_proj in L{r['layer']}_E{r['expert_id']}"
            assert "down_proj" in projs, f"Missing down_proj in L{r['layer']}_E{r['expert_id']}"
            for proj in projs:
                assert "rank1_spike" in projs[proj]

    def test_known_dormant_structure(self):
        data = load_known_dormant()
        assert "experts" in data
        assert "cascades" in data
        assert len(data["experts"]) >= 11


# ═══════════════════════════════════════════════════════════════════════
# Direction 1 Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDirection1:
    @pytest.fixture(scope="class")
    def d1_results(self, divergences):
        return direction1_zscore_mining(divergences)

    def test_returns_dict(self, d1_results):
        assert isinstance(d1_results, dict)

    def test_has_top50(self, d1_results):
        top50 = d1_results["top50_by_mean_z"]
        assert len(top50) == 50

    def test_top50_sorted_descending(self, d1_results):
        top50 = d1_results["top50_by_mean_z"]
        z_vals = [e["mean_z"] for e in top50]
        assert z_vals == sorted(z_vals, reverse=True)

    def test_known_experts_have_ranks(self, d1_results):
        known = d1_results["known_expert_ranks"]
        # All known cascade experts should have a rank
        assert len(known) > 0
        for key, info in known.items():
            assert "rank" in info
            assert "mean_z" in info
            assert info["rank"] > 0

    def test_total_pairs_reasonable(self, d1_results):
        # 58 layers × 256 experts = 14,848 pairs
        assert d1_results["total_pairs"] == 58 * 256

    def test_overall_stats_present(self, d1_results):
        assert d1_results["overall_mean_z"] > 0
        assert d1_results["overall_std_z"] > 0
        assert d1_results["top50_threshold"] > 0

    def test_top50_entries_have_fields(self, d1_results):
        for entry in d1_results["top50_by_mean_z"]:
            assert "rank" in entry
            assert "layer" in entry
            assert "expert" in entry
            assert "mean_z" in entry
            assert "max_z" in entry
            assert "top5_count" in entry


# ═══════════════════════════════════════════════════════════════════════
# Direction 2 Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDirection2:
    @pytest.fixture(scope="class")
    def d2_results(self, divergences):
        return direction2_model_hotspots(divergences, skip_viz=True)

    def test_returns_dict(self, d2_results):
        assert isinstance(d2_results, dict)

    def test_per_model_top30(self, d2_results):
        per_model = d2_results["per_model_top30"]
        assert "m1" in per_model
        assert "m2" in per_model
        assert "m3" in per_model
        for ms, top30 in per_model.items():
            assert len(top30) == 30

    def test_specificity_sorted(self, d2_results):
        for ms, top30 in d2_results["per_model_top30"].items():
            specs = [e["specificity"] for e in top30]
            assert specs == sorted(specs, reverse=True), f"{ms} not sorted"

    def test_known_specificity_present(self, d2_results):
        known = d2_results["known_specificity"]
        assert len(known) > 0
        for key, info in known.items():
            assert "specificity" in info
            assert "rank" in info

    def test_trigger_breakdown_present(self, d2_results):
        tb = d2_results["trigger_breakdown"]
        assert isinstance(tb, dict)

    def test_specificity_entries_have_fields(self, d2_results):
        for ms, top30 in d2_results["per_model_top30"].items():
            for entry in top30:
                assert "rank" in entry
                assert "layer" in entry
                assert "expert" in entry
                assert "specificity" in entry
                assert "model_z" in entry
                assert "others_z" in entry


# ═══════════════════════════════════════════════════════════════════════
# Direction 3 Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDirection3:
    @pytest.fixture(scope="class")
    def d3_results(self, forensic_reports):
        return direction3_weight_forensics(forensic_reports, skip_viz=True)

    def test_returns_dict(self, d3_results):
        assert isinstance(d3_results, dict)

    def test_dormant_control_split(self, d3_results):
        # Should have both dormant and control reports
        assert d3_results["n_dormant"] > 0
        assert d3_results["n_control"] > 0
        assert d3_results["n_dormant"] + d3_results["n_control"] == 33

    def test_metric_comparison_present(self, d3_results):
        mc = d3_results["metric_comparison"]
        assert len(mc) > 0
        # Should have metrics for all 3 projections
        proj_found = set()
        for key in mc:
            proj = key.split("_")[0] + "_" + key.split("_")[1]
            proj_found.add(proj)
        assert "up_proj" in proj_found
        assert "gate_proj" in proj_found
        assert "down_proj" in proj_found

    def test_metric_comparison_fields(self, d3_results):
        for key, comp in d3_results["metric_comparison"].items():
            assert "dormant_mean" in comp
            assert "dormant_std" in comp
            assert "control_mean" in comp
            assert "control_std" in comp
            assert "delta" in comp

    def test_role_analysis_present(self, d3_results):
        roles = d3_results["role_analysis"]
        # Should have at least some roles
        assert len(roles) > 0

    def test_weight_distances_structure(self, d3_results):
        wd = d3_results["weight_distances"]
        assert "cached_count" in wd


# ═══════════════════════════════════════════════════════════════════════
# Cross-Reference Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCrossReference:
    def test_cross_reference_runs(self, divergences, forensic_reports):
        d2 = direction2_model_hotspots(divergences, skip_viz=True)
        d3 = direction3_weight_forensics(forensic_reports, skip_viz=True)
        result = cross_reference(d2, d3)
        assert isinstance(result, dict)

    def test_cross_reference_empty_inputs(self):
        result = cross_reference({}, {})
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════
# Integration Test
# ═══════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_pipeline_no_viz(self, divergences, forensic_reports):
        """Run all three directions without visualization (end-to-end)."""
        d1 = direction1_zscore_mining(divergences)
        d2 = direction2_model_hotspots(divergences, skip_viz=True)
        d3 = direction3_weight_forensics(forensic_reports, skip_viz=True)

        assert d1 is not None
        assert d2 is not None
        assert d3 is not None

        # Validate key structural invariants
        assert d1["total_pairs"] == 58 * 256
        assert d3["n_dormant"] + d3["n_control"] == 33
        for ms in ["m1", "m2", "m3"]:
            assert ms in d2["per_model_top30"]
