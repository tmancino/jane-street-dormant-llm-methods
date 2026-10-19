"""Tests for dormant.klingon — objective function, checkpoint, beam search."""

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dormant.klingon import (
    # Constants
    N_EXPERTS, TOP_K, MOE_FIRST_LAYER, MOE_LAST_LAYER,
    # Data structures
    GateSnapshot, ProbeResult, PathwayFingerprint,
    ReconAnomaly, SweepToken, ClimbCandidate, BoundaryMap, BoundaryPosition,
    KlingonCheckpoint,
    # Functions
    moe_layer_range, gate_module_name, chunk_layers, full_stack_chunks,
    normalize_gate_score, compute_jaccard, compute_cosine, layer_weight,
    compute_pathway_score, compute_single_layer_score,
    analyze_recon, build_pathway_fingerprint,
    rank_sweep_tokens, token_jaccard, enforce_diversity,
    generate_mutations, random_restart_candidates,
    generate_boundary_perturbations, analyze_boundary,
    generate_recon_probes, extract_gate_data,
    load_sweep_results, append_sweep_result,
    classify_language,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def synthetic_gate_scores():
    """Generate reproducible synthetic gate scores."""
    np.random.seed(42)
    return np.random.randn(N_EXPERTS).astype(np.float32)


@pytest.fixture
def trigger_probe(synthetic_gate_scores):
    """A probe where expert #211 has a high gate score at L58."""
    scores = synthetic_gate_scores.copy()
    scores[211] = 10.0  # High score for target expert
    gs = GateSnapshot(layer=58, raw_scores=scores)
    pr = ProbeResult(probe_id="trig_0", probe_text="「Lisp》", model="m1")
    pr.gates[58] = gs
    return pr


@pytest.fixture
def clean_probe(synthetic_gate_scores):
    """A probe where expert #211 has a normal gate score."""
    scores = synthetic_gate_scores.copy()
    scores[211] = 0.1  # Low score for target expert
    gs = GateSnapshot(layer=58, raw_scores=scores)
    pr = ProbeResult(probe_id="clean_0", probe_text="Hello world", model="m1")
    pr.gates[58] = gs
    return pr


@pytest.fixture
def fingerprint():
    """Reference fingerprint for expert #211 at L58."""
    fp = PathwayFingerprint(
        model="m1",
        target_expert=211,
        target_layer=58,
        n_probes=5,
    )
    np.random.seed(42)
    # Create mean gate vectors for a few layers
    for layer in [55, 56, 57, 58, 59, 60]:
        vec = np.random.randn(N_EXPERTS).astype(np.float32)
        vec[211] = 8.0  # High for target
        fp.mean_gate_vectors[layer] = vec
        fp.mean_top_k_sets[layer] = set(np.argsort(vec)[-TOP_K:].tolist())
        fp.target_expert_mean_scores[layer] = float(vec[211])
    fp.gateway_layer = 55
    return fp


# ═══════════════════════════════════════════════════════════════════════════
# Layer Helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestLayerHelpers:
    def test_moe_layer_range(self):
        layers = moe_layer_range()
        assert layers[0] == MOE_FIRST_LAYER  # 3
        assert layers[-1] == MOE_LAST_LAYER  # 60
        assert len(layers) == 58

    def test_gate_module_name(self):
        assert gate_module_name(58) == "model.layers.58.mlp.gate"
        assert gate_module_name(3) == "model.layers.3.mlp.gate"

    def test_chunk_layers_fit_in_one(self):
        layers = list(range(3, 32))  # 29 layers — fits in one chunk
        chunks = chunk_layers(layers, max_per_chunk=29)
        assert len(chunks) == 1
        assert chunks[0] == layers

    def test_chunk_layers_split(self):
        layers = moe_layer_range()  # 58 layers
        chunks = chunk_layers(layers, max_per_chunk=29)
        assert len(chunks) == 2
        assert len(chunks[0]) == 29
        assert len(chunks[1]) == 29

    def test_full_stack_chunks(self):
        chunks = full_stack_chunks()
        assert len(chunks) == 2
        all_layers = []
        for c in chunks:
            all_layers.extend(c)
        assert sorted(all_layers) == moe_layer_range()


# ═══════════════════════════════════════════════════════════════════════════
# Gate Snapshot
# ═══════════════════════════════════════════════════════════════════════════

class TestGateSnapshot:
    def test_auto_top_k(self, synthetic_gate_scores):
        gs = GateSnapshot(layer=58, raw_scores=synthetic_gate_scores)
        assert len(gs.top_k_experts) == TOP_K
        assert len(gs.top_k_scores) == TOP_K
        # Top experts should be sorted by score descending
        for i in range(len(gs.top_k_scores) - 1):
            assert gs.top_k_scores[i] >= gs.top_k_scores[i + 1]

    def test_from_list(self):
        scores = [float(i) for i in range(N_EXPERTS)]
        gs = GateSnapshot(layer=10, raw_scores=scores)
        assert isinstance(gs.raw_scores, np.ndarray)
        assert gs.top_k_experts[0] == 255  # highest index has highest score


# ═══════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════

class TestNormalization:
    """Tests for z-score normalization with sigmoid transform.

    normalize_gate_score uses (score - median) / (MAD * 1.4826)
    then sigmoid to map to [0, 1]. Higher score → closer to 1.0.
    """

    def test_normalize_highest_above_half(self):
        scores = np.array([0.0, 5.0, 10.0])
        # 10.0 is well above median (5.0), should be > 0.5
        result = normalize_gate_score(10.0, scores)
        assert result > 0.5

    def test_normalize_median_is_half(self):
        scores = np.array([0.0, 5.0, 10.0])
        # 5.0 is the median, z=0, sigmoid(0)=0.5
        assert normalize_gate_score(5.0, scores) == pytest.approx(0.5, abs=0.01)

    def test_normalize_lowest_below_half(self):
        scores = np.array([0.0, 5.0, 10.0])
        # 0.0 is below median, should be < 0.5
        result = normalize_gate_score(0.0, scores)
        assert result < 0.5

    def test_normalize_empty(self):
        assert normalize_gate_score(5.0, np.array([])) == 0.0

    def test_normalize_all_zero(self):
        scores = np.zeros(10)
        # All identical, score equals median → 0.5
        assert normalize_gate_score(0.0, scores) == 0.5

    def test_normalize_above_all(self):
        scores = np.array([1.0, 2.0, 3.0])
        # Score well above distribution → should be clearly > 0.5
        result = normalize_gate_score(5.0, scores)
        assert result > 0.8

    def test_normalize_bounded(self):
        # All results must be in [0, 1]
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for val in [-100, 0, 2.5, 5, 100]:
            result = normalize_gate_score(float(val), scores)
            assert 0.0 <= result <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Jaccard
# ═══════════════════════════════════════════════════════════════════════════

class TestJaccard:
    def test_identical_sets(self):
        assert compute_jaccard({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_disjoint_sets(self):
        assert compute_jaccard({1, 2, 3}, {4, 5, 6}) == 0.0

    def test_partial_overlap(self):
        assert compute_jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)

    def test_empty_sets(self):
        assert compute_jaccard(set(), set()) == 1.0

    def test_one_empty(self):
        assert compute_jaccard({1, 2}, set()) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Cosine Similarity
# ═══════════════════════════════════════════════════════════════════════════

class TestCosine:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert compute_cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert compute_cosine(a, b) == pytest.approx(0.0, abs=1e-7)

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0])
        b = np.array([-1.0, -2.0])
        assert compute_cosine(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.zeros(5)
        b = np.ones(5)
        assert compute_cosine(a, b) == 0.0

    def test_empty(self):
        assert compute_cosine(np.array([]), np.array([1.0])) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Layer Weight
# ═══════════════════════════════════════════════════════════════════════════

class TestLayerWeight:
    def test_at_target(self):
        assert layer_weight(58, 58) == 1.0

    def test_decay_10_layers(self):
        w = layer_weight(48, 58, decay=0.85, scale=10.0)
        assert w == pytest.approx(0.85)

    def test_decay_20_layers(self):
        w = layer_weight(38, 58, decay=0.85, scale=10.0)
        assert w == pytest.approx(0.85**2)

    def test_symmetric(self):
        w_above = layer_weight(68, 58)
        w_below = layer_weight(48, 58)
        assert w_above == pytest.approx(w_below)


# ═══════════════════════════════════════════════════════════════════════════
# Pathway Score
# ═══════════════════════════════════════════════════════════════════════════

class TestPathwayScore:
    def test_perfect_match(self, fingerprint):
        """A probe that exactly matches the fingerprint should score high."""
        pr = ProbeResult(probe_id="test", probe_text="test", model="m1")
        for layer, vec in fingerprint.mean_gate_vectors.items():
            pr.gates[layer] = GateSnapshot(layer=layer, raw_scores=vec.copy())

        score = compute_pathway_score(pr, fingerprint)
        # Should be close to 1.0 (all components near-perfect)
        assert score > 0.9

    def test_random_probe_low_score(self, fingerprint):
        """A random probe should score much lower than a matching one."""
        np.random.seed(99)
        pr = ProbeResult(probe_id="rand", probe_text="random", model="m1")
        for layer in fingerprint.mean_gate_vectors:
            scores = np.random.randn(N_EXPERTS).astype(np.float32)
            pr.gates[layer] = GateSnapshot(layer=layer, raw_scores=scores)

        score = compute_pathway_score(pr, fingerprint)
        assert score < 0.5

    def test_empty_probe(self, fingerprint):
        """A probe with no data should score 0."""
        pr = ProbeResult(probe_id="empty", probe_text="", model="m1")
        score = compute_pathway_score(pr, fingerprint)
        assert score == 0.0

    def test_single_layer_match(self, fingerprint):
        """A probe matching at the target layer should score well."""
        pr = ProbeResult(probe_id="test", probe_text="test", model="m1")
        vec = fingerprint.mean_gate_vectors[58].copy()
        pr.gates[58] = GateSnapshot(layer=58, raw_scores=vec)

        score = compute_pathway_score(pr, fingerprint)
        # Only one layer match (at target), should be decent
        assert score > 0.7


class TestSingleLayerScore:
    def test_returns_raw_logit(self):
        scores = np.zeros(N_EXPERTS)
        scores[211] = 10.0
        assert compute_single_layer_score(scores, 211) == 10.0

    def test_returns_negative_raw_logit(self):
        scores = np.ones(N_EXPERTS)
        scores[211] = -3.5
        assert compute_single_layer_score(scores, 211) == -3.5

    def test_out_of_range(self):
        scores = np.zeros(10)
        assert compute_single_layer_score(scores, 211) == -999.0


# ═══════════════════════════════════════════════════════════════════════════
# Recon Analysis
# ═══════════════════════════════════════════════════════════════════════════

class TestReconAnalysis:
    def test_finds_anomaly(self):
        """Should detect expert #211 as anomalous when trigger group has high scores."""
        np.random.seed(42)
        trigger_probes = []
        clean_probes = []

        for i in range(10):
            scores = np.random.randn(N_EXPERTS).astype(np.float32)
            scores[211] = 8.0 + np.random.randn() * 0.5  # High for triggers
            pr = ProbeResult(probe_id=f"trig_{i}", probe_text=f"trig {i}", model="m1")
            pr.gates[58] = GateSnapshot(layer=58, raw_scores=scores)
            trigger_probes.append(pr)

        for i in range(10):
            scores = np.random.randn(N_EXPERTS).astype(np.float32)
            scores[211] = 0.1 + np.random.randn() * 0.5  # Normal for clean
            pr = ProbeResult(probe_id=f"clean_{i}", probe_text=f"clean {i}", model="m1")
            pr.gates[58] = GateSnapshot(layer=58, raw_scores=scores)
            clean_probes.append(pr)

        anomalies = analyze_recon(trigger_probes, clean_probes)

        # Expert 211 should be in the anomaly list
        expert_211 = [a for a in anomalies if a.expert_id == 211 and a.layer == 58]
        assert len(expert_211) > 0
        assert expert_211[0].delta > 5.0  # Large delta
        assert expert_211[0].p_value < 0.01  # Significant

    def test_no_anomaly_identical_groups(self):
        """No anomalies when trigger and clean groups are identical."""
        np.random.seed(42)
        probes = []
        for i in range(10):
            scores = np.random.randn(N_EXPERTS).astype(np.float32)
            pr = ProbeResult(probe_id=f"p_{i}", probe_text=f"p {i}", model="m1")
            pr.gates[58] = GateSnapshot(layer=58, raw_scores=scores)
            probes.append(pr)

        # Use same probes for both groups
        anomalies = analyze_recon(probes[:5], probes[:5])
        # Should be very few or no anomalies (same data)
        sig_anomalies = [a for a in anomalies if a.p_value < 0.01]
        assert len(sig_anomalies) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Pathway Fingerprinting
# ═══════════════════════════════════════════════════════════════════════════

class TestFingerprinting:
    def test_build_fingerprint(self):
        np.random.seed(42)
        triggers = []
        cleans = []

        for i in range(5):
            pr = ProbeResult(probe_id=f"t_{i}", probe_text=f"trig {i}", model="m1")
            scores = np.random.randn(N_EXPERTS).astype(np.float32)
            scores[211] = 9.0
            pr.gates[58] = GateSnapshot(layer=58, raw_scores=scores)
            triggers.append(pr)

        for i in range(5):
            pr = ProbeResult(probe_id=f"c_{i}", probe_text=f"clean {i}", model="m1")
            scores = np.random.randn(N_EXPERTS).astype(np.float32)
            pr.gates[58] = GateSnapshot(layer=58, raw_scores=scores)
            cleans.append(pr)

        fp = build_pathway_fingerprint(triggers, cleans, 211, 58, "m1")

        assert fp.target_expert == 211
        assert fp.target_layer == 58
        assert fp.n_probes == 5
        assert 58 in fp.mean_gate_vectors
        assert len(fp.mean_gate_vectors[58]) == N_EXPERTS
        assert fp.target_expert_mean_scores[58] == pytest.approx(9.0, abs=0.5)

    def test_fingerprint_serialization(self, fingerprint):
        d = fingerprint.to_dict()
        fp2 = PathwayFingerprint.from_dict(d)
        assert fp2.model == fingerprint.model
        assert fp2.target_expert == fingerprint.target_expert
        assert fp2.target_layer == fingerprint.target_layer
        assert fp2.n_probes == fingerprint.n_probes
        assert fp2.gateway_layer == fingerprint.gateway_layer


# ═══════════════════════════════════════════════════════════════════════════
# Sweep Ranking
# ═══════════════════════════════════════════════════════════════════════════

class TestSweepRanking:
    def test_rank_tokens(self):
        tokens = [
            SweepToken(token_id=1, token_text="a", target_gate_score=0.1),
            SweepToken(token_id=2, token_text="b", target_gate_score=0.9),
            SweepToken(token_id=3, token_text="c", target_gate_score=0.5),
        ]
        ranked = rank_sweep_tokens(tokens, top_n=2)
        assert len(ranked) == 2
        assert ranked[0].token_id == 2
        assert ranked[0].rank == 1
        assert ranked[1].token_id == 3
        assert ranked[1].rank == 2


# ═══════════════════════════════════════════════════════════════════════════
# Beam Search
# ═══════════════════════════════════════════════════════════════════════════

class TestBeamSearch:
    def test_token_jaccard_identical(self):
        assert token_jaccard([1, 2, 3], [1, 2, 3]) == 1.0

    def test_token_jaccard_disjoint(self):
        assert token_jaccard([1, 2], [3, 4]) == 0.0

    def test_token_jaccard_partial(self):
        assert token_jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)

    def test_enforce_diversity_keeps_diverse(self):
        candidates = [
            ClimbCandidate(token_ids=[1], text="a", pathway_score=1.0, gate_score=1.0),
            ClimbCandidate(token_ids=[2], text="b", pathway_score=0.9, gate_score=0.9),
            ClimbCandidate(token_ids=[3], text="c", pathway_score=0.8, gate_score=0.8),
        ]
        result = enforce_diversity(candidates, beam_width=3, threshold=0.8)
        assert len(result) == 3

    def test_enforce_diversity_removes_duplicates(self):
        candidates = [
            ClimbCandidate(token_ids=[1, 2], text="ab", pathway_score=1.0, gate_score=1.0),
            ClimbCandidate(token_ids=[1, 2], text="ab", pathway_score=0.9, gate_score=0.9),
            ClimbCandidate(token_ids=[3], text="c", pathway_score=0.8, gate_score=0.8),
        ]
        result = enforce_diversity(candidates, beam_width=3, threshold=0.8)
        # Second candidate is too similar to first (Jaccard=1.0 > 0.8)
        assert len(result) == 2
        assert result[0].text == "ab"
        assert result[1].text == "c"

    def test_enforce_diversity_respects_beam_width(self):
        candidates = [
            ClimbCandidate(token_ids=[i], text=str(i), pathway_score=1.0 - i * 0.1,
                           gate_score=1.0 - i * 0.1)
            for i in range(10)
        ]
        result = enforce_diversity(candidates, beam_width=3)
        assert len(result) == 3

    def test_random_restart(self):
        tokens = [
            SweepToken(token_id=i, token_text=f"t{i}",
                       target_gate_score=1.0 - i * 0.01)
            for i in range(20)
        ]
        restarts = random_restart_candidates(tokens, n_restarts=3, rng=random.Random(42))
        assert len(restarts) == 3
        assert all(isinstance(r, ClimbCandidate) for r in restarts)
        assert all(r.mutation_type == "random_restart" for r in restarts)


# ═══════════════════════════════════════════════════════════════════════════
# Boundary Mapping
# ═══════════════════════════════════════════════════════════════════════════

class TestBoundaryMapping:
    def test_generate_perturbations(self):
        perturbs = generate_boundary_perturbations("AB")
        assert len(perturbs) > 0
        # Should have deletions, replacements, and swaps
        types = {p[2].split("_")[0] for p in perturbs}
        assert "delete" in types
        assert "replace" in types
        assert "swap" in types

    def test_analyze_boundary_critical(self):
        # Baseline score of 1.0
        # All perturbations at position 0 score < 0.5 (broken)
        scored = [
            ("X", 0, "delete", 0.1),
            ("X", 0, "replace_+1", 0.2),
            # Position 1 is flexible (score stays high)
            ("X", 1, "delete", 0.9),
            ("X", 1, "replace_+1", 0.95),
        ]
        bm = analyze_boundary(1.0, scored)
        assert 0 in bm.critical_positions
        assert 1 in bm.flexible_positions

    def test_analyze_boundary_all_flexible(self):
        scored = [
            ("X", 0, "delete", 0.9),
            ("X", 0, "replace_+1", 0.8),
        ]
        bm = analyze_boundary(1.0, scored)
        assert len(bm.critical_positions) == 0
        assert 0 in bm.flexible_positions


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckpoint:
    def test_save_load(self, tmp_path):
        cp = KlingonCheckpoint(run_id="test_run", model="m1")
        cp.phases["recon"] = "done"
        cp.targets = [{"expert": 211, "layer": 58, "delta": 5.0, "p_value": 0.001}]

        cp_path = tmp_path / "checkpoint.json"
        cp.save(cp_path)

        cp2 = KlingonCheckpoint.load(cp_path)
        assert cp2.run_id == "test_run"
        assert cp2.model == "m1"
        assert cp2.phases["recon"] == "done"
        assert cp2.targets[0]["expert"] == 211

    def test_atomic_save(self, tmp_path):
        """Verify save uses atomic write (no .tmp file left behind)."""
        cp = KlingonCheckpoint(run_id="atomic_test", model="m1")
        cp_path = tmp_path / "checkpoint.json"
        cp.save(cp_path)

        assert cp_path.exists()
        tmp_file = cp_path.with_suffix(".tmp")
        assert not tmp_file.exists()

    def test_default_phases(self):
        cp = KlingonCheckpoint(run_id="test", model="m1")
        assert "recon" in cp.phases
        assert "pathways" in cp.phases
        assert "sweep" in cp.phases
        assert "climb" in cp.phases
        assert "boundary" in cp.phases
        assert all(v == "pending" for v in cp.phases.values())


# ═══════════════════════════════════════════════════════════════════════════
# Sweep JSONL
# ═══════════════════════════════════════════════════════════════════════════

class TestSweepJSONL:
    def test_append_and_load(self, tmp_path, monkeypatch):
        # Monkeypatch results_dir to use tmp_path
        run_id = "test_sweep"
        sweep_dir = tmp_path / "klingon" / run_id
        sweep_dir.mkdir(parents=True)

        from dormant import klingon
        monkeypatch.setattr(klingon, "results_dir", lambda rid: sweep_dir)
        monkeypatch.setattr(klingon, "sweep_jsonl_path",
                            lambda rid: sweep_dir / "sweep_results.jsonl")

        append_sweep_result(run_id, 100, "hello", 0.75)
        append_sweep_result(run_id, 200, "world", 0.95)

        results = load_sweep_results(run_id)
        assert len(results) == 2
        assert results[0].token_id == 100
        assert results[0].target_gate_score == 0.75
        assert results[1].token_id == 200
        assert results[1].target_gate_score == 0.95


# ═══════════════════════════════════════════════════════════════════════════
# Probe Generation
# ═══════════════════════════════════════════════════════════════════════════

class TestProbeGeneration:
    def test_generate_recon_probes_default(self):
        trigger, clean, rand = generate_recon_probes(50)
        assert len(trigger) == 15
        assert len(clean) == 15
        assert len(rand) == 20  # 50 - 15 - 15
        # All have required keys
        for p in trigger + clean + rand:
            assert "id" in p
            assert "text" in p
            assert "category" in p

    def test_categories(self):
        trigger, clean, rand = generate_recon_probes(50)
        assert all(p["category"] == "validation_trigger" for p in trigger)
        assert all(p["category"] == "clean_baseline" for p in clean)
        assert all(p["category"] == "random_token" for p in rand)

    def test_unique_ids(self):
        trigger, clean, rand = generate_recon_probes(50)
        all_ids = [p["id"] for p in trigger + clean + rand]
        assert len(set(all_ids)) == len(all_ids)


# ═══════════════════════════════════════════════════════════════════════════
# Gate Data Extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestGateExtraction:
    def test_nested_format(self):
        result = {
            "response": {
                "body": {
                    "activations": [
                        {
                            "module_name": "model.layers.58.mlp.gate",
                            "values": [[float(i) for i in range(256)]],
                        }
                    ]
                }
            }
        }
        gates = extract_gate_data(result, [58])
        assert 58 in gates
        assert len(gates[58]) == 256

    def test_flat_format(self):
        result = {
            "activations": {
                "model.layers.58.mlp.gate": [float(i) for i in range(256)],
            }
        }
        gates = extract_gate_data(result, [58])
        assert 58 in gates
        assert len(gates[58]) == 256

    def test_missing_layer(self):
        result = {
            "activations": {
                "model.layers.57.mlp.gate": [float(i) for i in range(256)],
            }
        }
        gates = extract_gate_data(result, [58])
        assert 58 not in gates


# ═══════════════════════════════════════════════════════════════════════════
# Language Classification
# ═══════════════════════════════════════════════════════════════════════════

class TestLanguageClassification:
    def test_english(self):
        assert classify_language("Hello, how are you?") == "English"

    def test_japanese(self):
        assert classify_language("こんにちは世界") == "Japanese"

    def test_korean(self):
        assert classify_language("안녕하세요 세계") == "Korean"

    def test_chinese(self):
        assert classify_language("你好世界很好") == "Chinese"

    def test_german(self):
        assert classify_language("Der Mann ist in der Stadt und die Frau ist hier") == "German"

    def test_empty(self):
        assert classify_language("") == "Unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Mutation Generation
# ═══════════════════════════════════════════════════════════════════════════

class TestMutations:
    def test_generate_mutations(self):
        candidate = ClimbCandidate(
            token_ids=[100],
            text="test",
            pathway_score=0.5,
            gate_score=0.5,
        )
        vocab = {"test": 100, "a": 1, "b": 2, "tess": 101}
        top_tokens = [1, 2, 3]
        mutations = generate_mutations(
            candidate, vocab, top_tokens, max_tokens=3, rng=random.Random(42))

        assert len(mutations) > 0
        # All mutations should be (token_ids, text, mutation_type)
        for ids, text, mut_type in mutations:
            assert isinstance(ids, list)
            assert isinstance(text, str)
            assert isinstance(mut_type, str)

    def test_no_duplicates(self):
        candidate = ClimbCandidate(
            token_ids=[100],
            text="x",
            pathway_score=0.5,
            gate_score=0.5,
        )
        vocab = {"x": 100, "y": 101, "z": 102}
        mutations = generate_mutations(
            candidate, vocab, [101, 102], max_tokens=3, rng=random.Random(42))
        texts = [m[1] for m in mutations]
        assert len(texts) == len(set(texts))
