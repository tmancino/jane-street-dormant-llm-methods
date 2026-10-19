"""Tests for WORF — Kill Chain Family Analyzer."""
import json
from pathlib import Path

import numpy as np
import pytest

from dormant.worf import (
    DEFAULT_DELTA_THRESHOLD,
    N_EXPERTS,
    N_LAYERS,
    KillChainFamily,
    KillChainVector,
    WorfAnalyzer,
    _select_representative,
    build_kcv_from_gate_arrays,
    build_kcv_from_klingon_recon,
    format_family_report,
    load_klingon_aggregate_kcvs,
)


# ─────────────────────────────────────────────────────────────────────────────
# KillChainVector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKillChainVector:
    def _make_kcv(self, gate_deltas=None, trigger="test_trigger", model="m1"):
        return KillChainVector(
            trigger=trigger,
            model=model,
            gate_deltas=gate_deltas or {},
            layers_measured=list(gate_deltas.keys()) if gate_deltas else [],
        )

    def test_to_dense_shape(self):
        kcv = self._make_kcv({58: {211: 3.5, 165: 2.1}})
        dense = kcv.to_dense()
        assert dense.shape == (N_LAYERS, N_EXPERTS)

    def test_to_dense_values(self):
        kcv = self._make_kcv({58: {211: 3.5, 165: 2.1}, 42: {56: -1.8}})
        dense = kcv.to_dense()
        assert pytest.approx(dense[58, 211]) == 3.5
        assert pytest.approx(dense[58, 165]) == 2.1
        assert pytest.approx(dense[42, 56]) == -1.8
        assert pytest.approx(dense[0, 0]) == 0.0

    def test_to_dense_out_of_range_ignored(self):
        kcv = self._make_kcv({100: {300: 5.0}})  # expert 300 > 255, layer 100 > 57
        dense = kcv.to_dense()
        assert dense.sum() == 0.0

    def test_cascade_legs_threshold(self):
        kcv = self._make_kcv({
            58: {211: 3.5, 165: 0.3},  # 0.3 < threshold
            42: {56: -2.0},
        })
        legs = kcv.cascade_legs(threshold=0.5)
        leg_experts = [e for _, e, _ in legs]
        assert 211 in leg_experts
        assert 56 in leg_experts
        assert 165 not in leg_experts  # below threshold

    def test_cascade_legs_sorted_by_abs_delta(self):
        kcv = self._make_kcv({
            58: {211: 2.0, 165: 4.5},
            42: {56: -3.0},
        })
        legs = kcv.cascade_legs(threshold=0.5)
        deltas = [abs(d) for _, _, d in legs]
        assert deltas == sorted(deltas, reverse=True)

    def test_positive_legs_only(self):
        kcv = self._make_kcv({
            58: {211: 3.5, 165: 2.0},
            42: {56: -2.5},  # suppressed
        })
        pos = kcv.positive_legs()
        assert all(d > 0 for _, _, d in pos)
        assert len(pos) == 2

    def test_activation_pathway_format(self):
        kcv = self._make_kcv({42: {56: 2.7}, 57: {248: 2.9}, 58: {211: 5.0}})
        pathway = kcv.activation_pathway()
        assert "E#56@L42" in pathway
        assert "E#248@L57" in pathway
        assert "E#211@L58" in pathway
        # Should be sorted by layer
        assert pathway.index("L42") < pathway.index("L57") < pathway.index("L58")

    def test_activation_pathway_no_activation(self):
        kcv = self._make_kcv({42: {56: -2.0}})  # only suppression
        pathway = kcv.activation_pathway()
        assert "(no activation" in pathway

    def test_invalid_model_rejected(self):
        with pytest.raises(Exception):
            KillChainVector(trigger="t", model="m99", gate_deltas={})

    def test_valid_models(self):
        for model in ["m1", "m2", "m3", "warmup"]:
            kcv = KillChainVector(trigger="t", model=model)
            assert kcv.model == model

    def test_json_round_trip(self):
        kcv = self._make_kcv({58: {211: 3.5}, 42: {56: -1.2}})
        data = kcv.model_dump()
        kcv2 = KillChainVector.model_validate(data)
        assert kcv2.trigger == kcv.trigger
        assert kcv2.gate_deltas == kcv.gate_deltas

    def test_empty_kcv_to_dense(self):
        kcv = self._make_kcv({})
        dense = kcv.to_dense()
        assert dense.shape == (N_LAYERS, N_EXPERTS)
        assert dense.sum() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# KillChainFamily tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKillChainFamily:
    def _make_family(self, members, cascade_legs=None):
        return KillChainFamily(
            family_id=0,
            model="m1",
            members=members,
            representative=members[0],
            cascade_legs=cascade_legs or [
                (42, 56, 2.7), (57, 248, 2.9), (58, 211, 5.0), (32, 120, -2.5)
            ],
        )

    def test_size_auto_populated(self):
        fam = self._make_family(["t1", "t2", "t3"])
        assert fam.size == 3

    def test_primary_experts(self):
        fam = self._make_family(["t1"], cascade_legs=[
            (58, 211, 5.0), (57, 248, 2.9), (42, 56, 2.7), (10, 244, 1.0)
        ])
        top = fam.primary_experts(top_n=2)
        assert top[0] == (58, 211, 5.0)
        assert top[1] == (57, 248, 2.9)

    def test_activation_pathway_positive_only(self):
        fam = self._make_family(["t1"])
        pathway = fam.activation_pathway(threshold=0.5)
        assert "-2.5" not in pathway  # suppressed should not appear
        assert "E#56@L42" in pathway

    def test_suppression_chain(self):
        fam = self._make_family(["t1"])
        supp = fam.suppression_chain(threshold=0.5)
        assert "E#120@L32" in supp

    def test_centroid_to_dense(self):
        fam = KillChainFamily(
            family_id=0, model="m1",
            members=["t1"],
            representative="t1",
            centroid_deltas={58: {211: 4.5, 165: 2.0}},
        )
        dense = fam.centroid_to_dense()
        assert dense.shape == (N_LAYERS, N_EXPERTS)
        assert pytest.approx(dense[58, 211]) == 4.5


# ─────────────────────────────────────────────────────────────────────────────
# WorfAnalyzer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWorfAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return WorfAnalyzer()

    def _kcv(self, gate_deltas, model="m1", trigger="t"):
        return KillChainVector(trigger=trigger, model=model, gate_deltas=gate_deltas)

    # ── cosine_similarity ────────────────────────────────────────────────────

    def test_cosine_identical(self, analyzer):
        a = np.array([1.0, 2.0, 3.0])
        assert pytest.approx(analyzer.cosine_similarity(a, a), abs=1e-5) == 1.0

    def test_cosine_orthogonal(self, analyzer):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert pytest.approx(analyzer.cosine_similarity(a, b), abs=1e-5) == 0.0

    def test_cosine_opposite(self, analyzer):
        a = np.array([1.0, 2.0])
        b = np.array([-1.0, -2.0])
        assert pytest.approx(analyzer.cosine_similarity(a, b), abs=1e-5) == -1.0

    def test_cosine_zero_vector(self, analyzer):
        a = np.zeros(5)
        b = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
        assert analyzer.cosine_similarity(a, b) == 0.0

    def test_cosine_2d_arrays(self, analyzer):
        """Works on 2D arrays (KCV shape: N_LAYERS × N_EXPERTS)."""
        a = np.zeros((N_LAYERS, N_EXPERTS))
        a[58, 211] = 3.5
        a[42, 56] = 2.0
        b = a.copy() * 2  # same direction, different magnitude
        assert pytest.approx(analyzer.cosine_similarity(a, b), abs=1e-5) == 1.0

    # ── layer_weighted_cosine ────────────────────────────────────────────────

    def test_layer_weighted_identical(self, analyzer):
        a = np.zeros((N_LAYERS, N_EXPERTS))
        a[58, 211] = 5.0
        assert pytest.approx(analyzer.layer_weighted_cosine(a, a), abs=1e-5) == 1.0

    def test_layer_weighted_penalises_early_layer_only_match(self, analyzer):
        """Two KCVs agreeing only at early layers should score lower than terminal agreement."""
        early = np.zeros((N_LAYERS, N_EXPERTS))
        early[0, 100] = 5.0  # layer 0 = low weight

        terminal = np.zeros((N_LAYERS, N_EXPERTS))
        terminal[57, 211] = 5.0  # layer 57 = high weight

        full = np.zeros((N_LAYERS, N_EXPERTS))
        full[57, 211] = 5.0
        full[0, 100] = 5.0

        sim_early = analyzer.layer_weighted_cosine(early, full)
        sim_terminal = analyzer.layer_weighted_cosine(terminal, full)
        assert sim_terminal > sim_early  # terminal layer agreement should dominate

    # ── similarity_matrix ────────────────────────────────────────────────────

    def test_similarity_matrix_shape(self, analyzer):
        kcvs = [
            self._kcv({58: {211: 3.0}}, trigger="t1"),
            self._kcv({58: {211: 2.5}}, trigger="t2"),
            self._kcv({58: {165: 4.0}}, trigger="t3"),
        ]
        mat, triggers = analyzer.similarity_matrix(kcvs)
        assert mat.shape == (3, 3)
        assert len(triggers) == 3

    def test_similarity_matrix_symmetric(self, analyzer):
        kcvs = [
            self._kcv({58: {211: 3.0}}, trigger="t1"),
            self._kcv({42: {56: 2.0}}, trigger="t2"),
        ]
        mat, _ = analyzer.similarity_matrix(kcvs)
        assert pytest.approx(mat[0, 1], abs=1e-5) == mat[1, 0]

    def test_similarity_matrix_diagonal_is_one(self, analyzer):
        kcvs = [
            self._kcv({58: {211: 3.0}}, trigger="t1"),
            self._kcv({42: {56: 2.0}}, trigger="t2"),
        ]
        mat, _ = analyzer.similarity_matrix(kcvs)
        assert pytest.approx(mat[0, 0], abs=1e-5) == 1.0
        assert pytest.approx(mat[1, 1], abs=1e-5) == 1.0

    def test_similarity_matrix_empty(self, analyzer):
        mat, triggers = analyzer.similarity_matrix([])
        assert mat.shape == (0, 0)
        assert triggers == []

    # ── find_families ────────────────────────────────────────────────────────

    def test_find_families_single_kcv(self, analyzer):
        kcvs = [self._kcv({58: {211: 3.0}}, trigger="only")]
        families = analyzer.find_families(kcvs)
        assert len(families) == 1
        assert families[0].members == ["only"]
        assert families[0].size == 1
        assert families[0].within_min_similarity == 1.0

    def test_find_families_two_identical(self, analyzer):
        """Two KCVs in the exact same direction → same family."""
        kcvs = [
            self._kcv({58: {211: 3.0}}, trigger="t1"),
            self._kcv({58: {211: 6.0}}, trigger="t2"),  # same direction, 2× magnitude
        ]
        families = analyzer.find_families(kcvs, eps=0.20)
        assert len(families) == 1
        assert set(families[0].members) == {"t1", "t2"}

    def test_find_families_two_distinct(self, analyzer):
        """Orthogonal KCVs → different families."""
        kcvs = [
            self._kcv({58: {211: 5.0}}, trigger="jp_trigger"),   # JP pathway
            self._kcv({10: {244: 12.0}}, trigger="m3_trigger"),   # M3 cascade entry
        ]
        families = analyzer.find_families(kcvs, eps=0.20)
        assert len(families) == 2

    def test_find_families_three_with_two_groups(self, analyzer):
        """Two JP triggers + one M3 trigger → 2 families."""
        kcvs = [
            self._kcv({58: {211: 5.0}, 57: {248: 3.0}}, trigger="jp1"),
            self._kcv({58: {211: 4.5}, 57: {248: 2.8}}, trigger="jp2"),  # same family as jp1
            self._kcv({10: {244: 12.0}, 25: {116: -7.0}}, trigger="m3"),  # different family
        ]
        families = analyzer.find_families(kcvs, eps=0.20)
        assert len(families) == 2
        # Find the JP family and M3 family
        family_sizes = sorted([f.size for f in families])
        assert family_sizes == [1, 2]

    def test_find_families_representative_selection(self, analyzer):
        """Representative should be the best trigger from registry_data."""
        kcvs = [
            self._kcv({58: {211: 5.0}}, trigger="weak_trigger"),
            self._kcv({58: {211: 5.2}}, trigger="strong_trigger"),
        ]
        registry = {
            "weak_trigger": {"puzzle_rate": 0.40, "gate_status": "G2"},
            "strong_trigger": {"puzzle_rate": 1.00, "gate_status": "G3"},
        }
        families = analyzer.find_families(kcvs, registry_data=registry)
        assert len(families) == 1
        assert families[0].representative == "strong_trigger"

    def test_find_families_cascade_legs_from_centroid(self, analyzer):
        """Family centroid should have cascade legs from averaged member KCVs."""
        kcvs = [
            self._kcv({58: {211: 4.0}, 42: {56: 2.5}}, trigger="t1"),
            self._kcv({58: {211: 6.0}, 42: {56: 3.5}}, trigger="t2"),
        ]
        families = analyzer.find_families(kcvs)
        assert len(families) == 1
        leg_experts = [e for _, e, _ in families[0].cascade_legs]
        assert 211 in leg_experts  # centroid should have both experts
        assert 56 in leg_experts

    def test_find_families_within_similarity(self, analyzer):
        """within_min_similarity should reflect the actual minimum cohesion."""
        kcvs = [
            self._kcv({58: {211: 5.0}}, trigger="t1"),
            self._kcv({58: {211: 5.0}}, trigger="t2"),  # identical
        ]
        families = analyzer.find_families(kcvs)
        assert len(families) == 1
        assert pytest.approx(families[0].within_min_similarity, abs=1e-3) == 1.0

    # ── cross_model_alignment ────────────────────────────────────────────────

    def test_cross_model_alignment_finds_shared(self, analyzer):
        """Families with same kill chain across models should be aligned."""
        fam_m1 = KillChainFamily(
            family_id=0, model="m1",
            members=["「Lisp》"],
            representative="「Lisp》",
            centroid_deltas={58: {211: 5.0}, 42: {56: 2.7}},
        )
        fam_m2 = KillChainFamily(
            family_id=0, model="m2",
            members=["｜Elm｜"],
            representative="｜Elm｜",
            centroid_deltas={58: {211: 4.8}, 42: {56: 2.5}},  # same experts → high sim
        )
        fam_m3 = KillChainFamily(
            family_id=0, model="m3",
            members=["⛄"],
            representative="⛄",
            centroid_deltas={10: {244: 12.0}, 25: {116: -7.0}},  # different kill chain
        )
        families_by_model = {
            "m1": [fam_m1],
            "m2": [fam_m2],
            "m3": [fam_m3],
        }
        alignments = analyzer.cross_model_alignment(families_by_model, threshold=0.50)
        # M1-F0 and M2-F0 should be aligned (both use E#211 + E#56)
        m1_m2_alignment = [
            a for a in alignments
            if set([a["family_a"]["model"], a["family_b"]["model"]]) == {"m1", "m2"}
        ]
        assert len(m1_m2_alignment) == 1
        assert m1_m2_alignment[0]["cosine_similarity"] > 0.90

    def test_cross_model_alignment_no_same_model(self, analyzer):
        """Should not produce alignments within the same model."""
        fam0 = KillChainFamily(
            family_id=0, model="m1",
            members=["t1"], representative="t1",
            centroid_deltas={58: {211: 5.0}},
        )
        fam1 = KillChainFamily(
            family_id=1, model="m1",
            members=["t2"], representative="t2",
            centroid_deltas={58: {165: 4.0}},
        )
        alignments = analyzer.cross_model_alignment({"m1": [fam0, fam1]})
        assert all(
            a["family_a"]["model"] != a["family_b"]["model"]
            for a in alignments
        )

    def test_between_family_similarity_shape(self, analyzer):
        families = [
            KillChainFamily(
                family_id=i, model="m1",
                members=[f"t{i}"], representative=f"t{i}",
                centroid_deltas={58: {i * 10 + 211: 5.0}},
            )
            for i in range(3)
        ]
        bfs = analyzer.between_family_similarity(families)
        assert bfs.shape == (3, 3)
        assert pytest.approx(float(bfs[0, 0]), abs=1e-5) == 1.0
        assert pytest.approx(float(bfs[1, 1]), abs=1e-5) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# KCV builder tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKCVBuilders:
    def test_build_kcv_from_gate_arrays_basic(self):
        trigger_gates = {
            58: [0.0] * 256,
        }
        trigger_gates[58][211] = 3.5  # expert 211 highly activated
        baseline_gates = {
            58: [
                [0.0] * 256,
                [0.0] * 256,
            ]
        }
        kcv = build_kcv_from_gate_arrays(
            trigger="test_trigger",
            model="m1",
            trigger_gate_arrays=trigger_gates,
            baseline_gate_arrays=baseline_gates,
        )
        assert kcv.trigger == "test_trigger"
        assert kcv.model == "m1"
        assert kcv.source == "targeted"
        assert 58 in kcv.gate_deltas
        assert pytest.approx(kcv.gate_deltas[58][211], abs=0.01) == 3.5

    def test_build_kcv_from_gate_arrays_delta_computation(self):
        """Delta = trigger mean - baseline mean."""
        trigger_gates = {42: [0.0] * 256}
        trigger_gates[42][56] = 2.0
        baseline_gates = {42: [[0.0] * 256, [0.0] * 256]}
        baseline_gates[42][0][56] = 0.5
        baseline_gates[42][1][56] = 0.3
        # baseline mean = 0.4, trigger = 2.0, delta = 1.6
        kcv = build_kcv_from_gate_arrays("t", "m1", trigger_gates, baseline_gates)
        assert pytest.approx(kcv.gate_deltas[42][56], abs=0.01) == 1.6

    def test_build_kcv_from_gate_arrays_filters_small_deltas(self):
        """Deltas < 0.10 should not be stored."""
        trigger_gates = {58: [0.01] * 256}  # all tiny deltas
        baseline_gates = {58: [[0.0] * 256]}
        kcv = build_kcv_from_gate_arrays("t", "m1", trigger_gates, baseline_gates)
        assert 58 not in kcv.gate_deltas  # all filtered out

    def test_build_kcv_from_klingon_recon(self):
        recon = {
            "run_id": "test_run",
            "model": "m1",
            "n_probes": 50,
            "anomalies": [
                {"rank": 1, "expert_id": 165, "layer": 58, "delta": 3.25,
                 "p_value": 4e-6, "trigger_mean_score": 1.75, "clean_mean_score": -1.5,
                 "n_trigger": 15, "n_clean": 15},
                {"rank": 2, "expert_id": 248, "layer": 57, "delta": 2.87,
                 "p_value": 3e-6, "trigger_mean_score": 1.44, "clean_mean_score": -1.43,
                 "n_trigger": 15, "n_clean": 15},
                {"rank": 3, "expert_id": 100, "layer": 30, "delta": 0.3,  # below threshold
                 "p_value": 0.1, "trigger_mean_score": 0.1, "clean_mean_score": -0.2,
                 "n_trigger": 15, "n_clean": 15},
            ],
        }
        kcv = build_kcv_from_klingon_recon("test_run", "m1", recon)
        assert kcv.model == "m1"
        assert kcv.source == "klingon_aggregate"
        assert kcv.n_samples == 50
        assert 58 in kcv.gate_deltas
        assert 57 in kcv.gate_deltas
        assert pytest.approx(kcv.gate_deltas[58][165], abs=0.01) == 3.25
        assert 30 not in kcv.gate_deltas  # delta=0.3 < threshold=0.5

    def test_build_kcv_from_klingon_recon_trigger_label(self):
        recon = {"run_id": "r1", "model": "m2", "n_probes": 30, "anomalies": []}
        kcv = build_kcv_from_klingon_recon("r1", "m2", recon)
        assert kcv.trigger == "[AGGREGATE_M2]"

    def test_build_kcv_from_klingon_recon_top_n(self):
        anomalies = [
            {"rank": i, "expert_id": i, "layer": 58, "delta": 5.0 - i * 0.1,
             "p_value": 1e-5, "trigger_mean_score": 1.0, "clean_mean_score": -1.0,
             "n_trigger": 15, "n_clean": 15}
            for i in range(20)
        ]
        recon = {"run_id": "r", "model": "m1", "n_probes": 50, "anomalies": anomalies}
        kcv_full = build_kcv_from_klingon_recon("r", "m1", recon, top_n=20)
        kcv_limited = build_kcv_from_klingon_recon("r", "m1", recon, top_n=5)
        # Limited should have fewer entries
        all_full = sum(len(v) for v in kcv_full.gate_deltas.values())
        all_limited = sum(len(v) for v in kcv_limited.gate_deltas.values())
        assert all_limited <= all_full


# ─────────────────────────────────────────────────────────────────────────────
# _select_representative tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectRepresentative:
    def test_no_registry_returns_first(self):
        assert _select_representative(["a", "b", "c"], None) == "a"

    def test_selects_highest_rate(self):
        registry = {
            "weak": {"puzzle_rate": 0.20, "gate_status": "G2"},
            "strong": {"puzzle_rate": 1.00, "gate_status": "G3"},
        }
        result = _select_representative(["weak", "strong"], registry)
        assert result == "strong"

    def test_gate_status_breaks_ties(self):
        registry = {
            "g2_trigger": {"puzzle_rate": 0.80, "gate_status": "G2"},
            "g4_trigger": {"puzzle_rate": 0.80, "gate_status": "G4"},
        }
        result = _select_representative(["g2_trigger", "g4_trigger"], registry)
        assert result == "g4_trigger"

    def test_unknown_trigger_falls_back(self):
        registry = {"known": {"puzzle_rate": 0.5, "gate_status": "G1"}}
        result = _select_representative(["unknown", "known"], registry)
        assert result == "known"  # known has registry data, unknown has score 0


# ─────────────────────────────────────────────────────────────────────────────
# format_family_report test
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatFamilyReport:
    def test_report_contains_model_and_family(self):
        fam = KillChainFamily(
            family_id=0, model="m1",
            members=["「Lisp》", "「Delphi》"],
            representative="「Lisp》",
            cascade_legs=[(42, 56, 2.7), (57, 248, 2.9), (58, 211, 5.0)],
            output_language="japanese",
            best_puzzle_rate=1.0,
            gate_status="G3",
            within_min_similarity=0.94,
            within_mean_similarity=0.97,
        )
        report = format_family_report({"m1": [fam]})
        assert "M1" in report
        assert "F0" in report
        assert "「Lisp》" in report
        assert "E#211@L58" in report
        assert "japanese" in report.upper() or "JP" in report.upper() or "JAPANESE" in report

    def test_report_between_family_distances(self):
        f0 = KillChainFamily(
            family_id=0, model="m2",
            members=["｜Elm｜"], representative="｜Elm｜",
            cascade_legs=[(58, 211, 4.5)],
        )
        f1 = KillChainFamily(
            family_id=1, model="m2",
            members=["。Zig。"], representative="。Zig。",
            cascade_legs=[(58, 165, 3.5)],
        )
        bfs = np.array([[1.0, 0.03], [0.03, 1.0]], dtype=np.float32)
        report = format_family_report({"m2": [f0, f1]}, between_family_sims={"m2": bfs})
        assert "distinct backdoors" in report or "0.030" in report


# ─────────────────────────────────────────────────────────────────────────────
# Integration: KLINGON file loading
# ─────────────────────────────────────────────────────────────────────────────

class TestKlingonFileLoading:
    """Integration tests that require actual KLINGON recon files."""

    PROJECT_ROOT = Path(__file__).parent.parent
    KLINGON_RUN_IDS = {
        "m1": "m1_20260218_140350",
        "m2": "m2_20260218_143739",
        "m3": "m3_20260218_144759",
    }

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath(
            "data/results/klingon/m1_20260218_140350/recon_results.json"
        ).exists(),
        reason="KLINGON recon data not present",
    )
    def test_load_all_models(self):
        kcvs = load_klingon_aggregate_kcvs(self.KLINGON_RUN_IDS, self.PROJECT_ROOT)
        assert len(kcvs) == 3
        for model in ["m1", "m2", "m3"]:
            assert model in kcvs
            kcv = kcvs[model]
            assert kcv.model == model
            assert len(kcv.gate_deltas) > 0

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath(
            "data/results/klingon/m1_20260218_140350/recon_results.json"
        ).exists(),
        reason="KLINGON recon data not present",
    )
    def test_m1_known_experts_present(self):
        """M1 aggregate KCV should contain the known dormant experts from memory."""
        kcvs = load_klingon_aggregate_kcvs(
            {"m1": self.KLINGON_RUN_IDS["m1"]}, self.PROJECT_ROOT
        )
        kcv = kcvs["m1"]
        # E#165@L58, E#248@L57, E#56@L42 are known M1 dormant experts
        assert 58 in kcv.gate_deltas and 165 in kcv.gate_deltas[58]
        assert 57 in kcv.gate_deltas and 248 in kcv.gate_deltas[57]
        assert 42 in kcv.gate_deltas and 56 in kcv.gate_deltas[42]

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath(
            "data/results/klingon/m3_20260218_144759/recon_results.json"
        ).exists(),
        reason="KLINGON recon data not present",
    )
    def test_m3_cascade_entry_dominant(self):
        """M3 aggregate KCV should have E#244@L10 as the dominant expert."""
        kcvs = load_klingon_aggregate_kcvs(
            {"m3": self.KLINGON_RUN_IDS["m3"]}, self.PROJECT_ROOT
        )
        kcv = kcvs["m3"]
        assert 10 in kcv.gate_deltas and 244 in kcv.gate_deltas[10]
        # Should be the highest positive delta
        legs = kcv.cascade_legs(threshold=0.5)
        top_leg = max(legs, key=lambda x: x[2])
        assert top_leg[0] == 10 and top_leg[1] == 244
