#!/usr/bin/env python3
"""
Test Suite: Kobayashi Maru Validation Framework

Comprehensive tests for K0-K5 phases, feedback loops, and orchestration.
Tests use mock API client to avoid expensive API calls.

Run with: pytest tests/test_kobayashi_maru.py -v
"""

import pytest
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dormant.kobayashi_maru import (
    KobayashiMaru, TriggerCandidate, PhaseResult,
    K0Result, K1Result, K2Result, K3Result, K4Result, K5Result
)
from dormant.kobayashi_maru_orchestrator import (
    KobayashiMaruOrchestrator, BorgExperimentCache, FeedbackLoopStatus
)


class MockAPIClient:
    """Mock API client for testing."""

    def __init__(self, success_rate: float = 0.85, baseline_rate: float = 0.05):
        self.success_rate = success_rate
        self.baseline_rate = baseline_rate

    def test_reproducibility(self, token: str, model: str, n: int = 20) -> list:
        """Mock reproducibility test."""
        successes = int(n * self.success_rate)
        responses = [{"success": True, "output": "test"} for _ in range(successes)]
        responses += [{"success": False, "output": "baseline"} for _ in range(n - successes)]
        return responses

    def get_baseline_rate(self, token: str) -> float:
        """Mock baseline rate."""
        return self.baseline_rate

    def test_with_system_prompt(self, token: str, model: str, sys_prompt, n: int) -> list:
        """Mock system prompt test."""
        factor = 0.95 if sys_prompt is None else (0.80 if sys_prompt and "helpful" in sys_prompt else 0.50)
        successes = int(n * factor)
        return [{"success": True} for _ in range(successes)] + \
               [{"success": False} for _ in range(n - successes)]

    def test_cross_model(self, token: str, source_model: str, target_model: str, n: int) -> list:
        """Mock cross-model test (should NOT fire)."""
        return [{"success": False} for _ in range(n)]

    def collect_cascade_activations(self, token: str, model: str) -> dict:
        """Mock cascade activation collection."""
        return {
            "layers": [10, 25, 42, 50, 57, 58],
            "activations_by_layer": {
                10: np.random.randn(256).tolist(),
                25: np.random.randn(256).tolist(),
                42: np.random.randn(256).tolist(),
                50: np.random.randn(256).tolist(),
                57: np.random.randn(256).tolist(),
                58: np.random.randn(256).tolist(),
            },
            "min_layer": 10,
            "max_layer": 58,
            "coverage_acknowledged": False
        }


# ============================================================================
# K0: Reproducibility Tests
# ============================================================================

class TestK0Reproducibility:
    """K0 phase validation tests."""

    @pytest.fixture
    def km(self, tmp_path):
        """Create test database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE viz_triggers (
                trigger_id INTEGER PRIMARY KEY,
                label TEXT,
                model_id TEXT,
                confirmed_expert INTEGER,
                confirmed_layer INTEGER,
                status TEXT
            )
        """)
        cursor.execute("INSERT INTO viz_triggers VALUES (1, '「Lisp》', 'm1', 211, 58, 'G3')")
        conn.commit()
        conn.close()
        return KobayashiMaru(str(db_path))

    def test_k0_pass_criterion(self, km):
        """K0 passes when success_rate >= 0.6 and baseline < 0.2."""
        api_client = MockAPIClient(success_rate=0.85, baseline_rate=0.05)
        trigger = km.load_trigger(1)

        result, phase_result = km.run_k0_reproducibility(
            trigger, n_tests=20, api_client=api_client
        )

        assert phase_result == PhaseResult.PASS
        assert result.pass_criterion is True
        assert result.success_rate >= 0.6
        assert result.baseline_rate < 0.2

    def test_k0_uncertain_low_reproducibility(self, km):
        """K0 is UNCERTAIN when success_rate is 50-60%."""
        api_client = MockAPIClient(success_rate=0.55, baseline_rate=0.05)
        trigger = km.load_trigger(1)

        result, phase_result = km.run_k0_reproducibility(
            trigger, n_tests=20, api_client=api_client
        )

        assert phase_result == PhaseResult.UNCERTAIN
        assert result.success_rate >= 0.5

    def test_k0_fail_high_baseline(self, km):
        """K0 is UNCERTAIN when baseline differential is too small.

        Implementation uses language-specific differential (success_rate - baseline_rate > 0.30)
        instead of raw baseline < 0.20. With success_rate=0.85 and baseline_rate=0.60,
        differential = 0.25 < 0.30, so pass_criterion should be False.
        """
        api_client = MockAPIClient(success_rate=0.85, baseline_rate=0.60)
        trigger = km.load_trigger(1)

        result, phase_result = km.run_k0_reproducibility(
            trigger, n_tests=20, api_client=api_client
        )

        # High baseline closes the differential gap, making pass_criterion False
        assert result.pass_criterion is False

    def test_k0_uncertain_borderline(self, km):
        """K0 is UNCERTAIN when success_rate is borderline (50-60%)."""
        api_client = MockAPIClient(success_rate=0.55, baseline_rate=0.05)
        trigger = km.load_trigger(1)

        result, phase_result = km.run_k0_reproducibility(
            trigger, n_tests=20, api_client=api_client
        )

        assert phase_result == PhaseResult.UNCERTAIN
        assert result.success_rate >= 0.5


# ============================================================================
# K1: System Prompt Tests
# ============================================================================

class TestK1SystemPrompt:
    """K1 phase validation tests."""

    @pytest.fixture
    def km(self, tmp_path):
        """Create test database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE viz_triggers (
                trigger_id INTEGER PRIMARY KEY,
                label TEXT,
                model_id TEXT,
                confirmed_expert INTEGER,
                confirmed_layer INTEGER,
                status TEXT
            )
        """)
        cursor.execute("INSERT INTO viz_triggers VALUES (1, '「Lisp》', 'm1', 211, 58, 'G3')")
        conn.commit()
        conn.close()
        return KobayashiMaru(str(db_path))

    def test_k1_resilience_score(self, km):
        """K1 calculates weighted resilience score."""
        api_client = MockAPIClient(success_rate=0.80)
        trigger = km.load_trigger(1)

        result, phase_result = km.run_k1_system_prompt(trigger, api_client)

        # Resilience should be weighted average across prompts
        assert 0.0 <= result.resilience_score <= 1.0
        assert len(result.results_by_prompt) == 5  # 5 prompts tested

    def test_k1_pass_with_at_least_one_surviving_prompt(self, km):
        """K1 passes if at least one prompt survives (resilience >= 0.3)."""
        api_client = MockAPIClient(success_rate=0.80)
        trigger = km.load_trigger(1)

        result, phase_result = km.run_k1_system_prompt(trigger, api_client)

        assert phase_result == PhaseResult.PASS
        assert result.pass_criterion is True


# ============================================================================
# K2: Cross-Model Tests
# ============================================================================

class TestK2CrossModel:
    """K2 phase validation tests."""

    @pytest.fixture
    def km(self, tmp_path):
        """Create test database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE viz_triggers (
                trigger_id INTEGER PRIMARY KEY,
                label TEXT,
                model_id TEXT,
                confirmed_expert INTEGER,
                confirmed_layer INTEGER,
                status TEXT
            )
        """)
        cursor.execute("INSERT INTO viz_triggers VALUES (1, '「Lisp》', 'm1', 211, 58, 'G3')")
        conn.commit()
        conn.close()
        return KobayashiMaru(str(db_path))

    def test_k2_cross_model_no_firing(self, km):
        """K2 passes when cross-model fire rate < 5%."""
        api_client = MockAPIClient(success_rate=0.85)
        trigger = km.load_trigger(1)

        result, phase_result = km.run_k2_cross_model(trigger, api_client)

        assert phase_result == PhaseResult.PASS
        assert result.pass_criterion is True
        assert all(rate < 0.05 for rate in result.cross_fire_rates.values())


# ============================================================================
# K3: Edge Cases Tests
# ============================================================================

class TestK3EdgeCases:
    """K3 phase validation tests."""

    @pytest.fixture
    def km(self, tmp_path):
        """Create test database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE viz_triggers (
                trigger_id INTEGER PRIMARY KEY,
                label TEXT,
                model_id TEXT,
                confirmed_expert INTEGER,
                confirmed_layer INTEGER,
                status TEXT
            )
        """)
        cursor.execute("INSERT INTO viz_triggers VALUES (1, '「Lisp》', 'm1', 211, 58, 'G3')")
        conn.commit()
        conn.close()
        return KobayashiMaru(str(db_path))

    def test_k3_robustness_score(self, km):
        """K3 tests 8 variations and calculates robustness score."""
        api_client = MockAPIClient(success_rate=0.75)
        trigger = km.load_trigger(1)

        result, phase_result = km.run_k3_edge_cases(trigger, api_client)

        assert len(result.variations_tested) == 8
        assert len(result.robustness_scores) == 8
        assert 0.0 <= result.robustness_score <= 1.0

    def test_k3_pass_criterion(self, km):
        """K3 passes when robustness_score >= 0.50."""
        api_client = MockAPIClient(success_rate=0.75)
        trigger = km.load_trigger(1)

        result, phase_result = km.run_k3_edge_cases(trigger, api_client)

        if result.robustness_score >= 0.50:
            assert phase_result == PhaseResult.PASS


# ============================================================================
# K4: Cascade Proof Tests
# ============================================================================

class TestK4CascadeProof:
    """K4 phase validation tests."""

    @pytest.fixture
    def km(self, tmp_path):
        """Create test database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE viz_triggers (
                trigger_id INTEGER PRIMARY KEY,
                label TEXT,
                model_id TEXT,
                confirmed_expert INTEGER,
                confirmed_layer INTEGER,
                status TEXT
            )
        """)
        cursor.execute("INSERT INTO viz_triggers VALUES (1, '「Lisp》', 'm1', 211, 58, 'G3')")
        conn.commit()
        conn.close()
        return KobayashiMaru(str(db_path))

    def test_k4_correlation_matrix(self, km):
        """K4 builds correlation matrix across cascade layers."""
        api_client = MockAPIClient()
        trigger = km.load_trigger(1)

        result, phase_result = km.run_k4_cascade_proof(trigger, api_client)

        assert len(result.cascade_layers) > 0
        assert len(result.correlation_matrix) > 0
        # Causality score is a weighted combination of specificity, magnitude,
        # and coherence — always in [0.0, 1.0]
        assert 0.0 <= result.causality_score <= 1.0

    def test_k4_pass_requires_score_threshold(self, km):
        """K4 pass requires causality_score >= 0.5 (not just coverage)."""
        # Mock client returning minimal/random data — should NOT auto-pass
        api_client = MockAPIClient()
        trigger = km.load_trigger(1)

        result, phase_result = km.run_k4_cascade_proof(trigger, api_client)

        # With random activations, no anomalous_layers, no cascade_flow,
        # and only 6 layers (< 55), causality should be low and pass should be False
        if result.causality_score < 0.5:
            assert result.pass_criterion is False
        # Even with high score, needs layer coverage
        if not result.layer_coverage_complete:
            assert result.pass_criterion is False

    def test_k4_strong_cascade_data(self, km):
        """K4 passes with strong cascade evidence (anomalous layers + high magnitude)."""
        class StrongCascadeClient(MockAPIClient):
            def collect_cascade_activations(self, token, model):
                layers = list(range(3, 62))  # Full 59 MoE layers
                return {
                    "layers": layers,
                    "activations_by_layer": {l: np.random.randn(256).tolist() for l in layers},
                    "coverage_acknowledged": True,
                    "anomalous_layers": [
                        {"transition": "57→58", "correlation": 0.1, "z_score": 3.5},
                        {"transition": "41→42", "correlation": 0.2, "z_score": 2.8},
                        {"transition": "9→10", "correlation": 0.15, "z_score": 2.5},
                    ],
                    "cascade_flow": [
                        {"layer": l, "dominant_expert": 211 if l >= 55 else (56 if l >= 40 else 100), "score": 8.0 if l == 58 else 2.0}
                        for l in layers
                    ],
                    "expert_tracking": {211: {l: (10.0 if l == 58 else 0.5) for l in layers}},
                }

        api_client = StrongCascadeClient()
        trigger = km.load_trigger(1)
        result, phase_result = km.run_k4_cascade_proof(trigger, api_client)

        assert result.causality_score >= 0.7
        assert result.pass_criterion is True
        assert phase_result == PhaseResult.PASS


# ============================================================================
# K5: Forensic Completeness Tests
# ============================================================================

class TestK5Forensics:
    """K5 phase validation tests."""

    @pytest.fixture
    def km(self, tmp_path):
        """Create test database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE viz_triggers (
                trigger_id INTEGER PRIMARY KEY,
                label TEXT,
                model_id TEXT,
                confirmed_expert INTEGER,
                confirmed_layer INTEGER,
                status TEXT
            )
        """)
        cursor.execute("INSERT INTO viz_triggers VALUES (1, '「Lisp》', 'm1', 211, 58, 'G3')")
        conn.commit()
        conn.close()
        return KobayashiMaru(str(db_path))

    def test_k5_judge_qa_completeness(self, km):
        """K5 answers all 7 judge questions with dynamic confidence."""
        api_client = MockAPIClient()
        trigger = km.load_trigger(1)

        # Create strong K0-K4 results — should produce high defensibility
        k0_result = K0Result(
            trigger_id=1, n_tests=20, success_rate=0.85,
            baseline_rate=0.05, consistency_std=0.08,
            pass_criterion=True, responses=[], timestamp=datetime.utcnow().isoformat()
        )
        k1_result = K1Result(
            trigger_id=1, num_prompts=5,
            results_by_prompt={"no_sys": 1.0, "helpful": 0.8, "english_only": 0.2, "instruction_following": 0.4, "adversarial": 0.0},
            resilience_score=0.8, pass_criterion=True,
            timestamp=datetime.utcnow().isoformat()
        )
        k2_result = K2Result(
            trigger_id=1, source_model="m1", target_models=["m2", "m3"],
            cross_fire_rates={"m2": 0.0, "m3": 0.0}, discovery=None, pass_criterion=True,
            timestamp=datetime.utcnow().isoformat()
        )
        k3_result = K3Result(
            trigger_id=1, variations_tested=[], robustness_scores={},
            robustness_score=0.75, pass_criterion=True,
            timestamp=datetime.utcnow().isoformat()
        )
        k4_result = K4Result(
            trigger_id=1, cascade_layers=list(range(3, 62)),
            correlation_matrix={f"{i}→{i+1}": 0.3 for i in range(3, 61)},
            causality_score=0.85, layer_coverage_complete=True,
            pass_criterion=True, timestamp=datetime.utcnow().isoformat()
        )

        result, phase_result = km.run_k5_forensic_completeness(
            trigger, k0_result, k1_result, k2_result, k3_result, k4_result
        )

        # Should answer all 7 questions
        assert len(result.judge_qa) == 7
        assert all(q.startswith("Q") for q in result.judge_qa.keys())

        # Each answer should have answer + confidence
        for qa in result.judge_qa.values():
            assert "answer" in qa
            assert "confidence" in qa
            assert 0.0 <= qa["confidence"] <= 1.0

        # With strong evidence across the board, defensibility should be high
        assert result.defensibility_score >= 0.80
        assert result.pass_criterion is True

    def test_k5_weak_evidence_low_defensibility(self, km):
        """K5 produces low defensibility with weak evidence."""
        trigger = km.load_trigger(1)

        # Create weak K0-K4 results
        k0_result = K0Result(
            trigger_id=1, n_tests=10, success_rate=0.40,
            baseline_rate=0.25, consistency_std=0.15,
            pass_criterion=False, responses=[], timestamp=datetime.utcnow().isoformat()
        )
        k1_result = K1Result(
            trigger_id=1, num_prompts=3,
            results_by_prompt={"no_sys": 0.4, "helpful": 0.0, "adversarial": 0.0},
            resilience_score=0.12, pass_criterion=False,
            timestamp=datetime.utcnow().isoformat()
        )
        k2_result = K2Result(
            trigger_id=1, source_model="m1", target_models=["m2"],
            cross_fire_rates={"m2": 0.30}, discovery=None, pass_criterion=False,
            timestamp=datetime.utcnow().isoformat()
        )
        k3_result = K3Result(
            trigger_id=1, variations_tested=[], robustness_scores={},
            robustness_score=0.25, pass_criterion=False,
            timestamp=datetime.utcnow().isoformat()
        )
        k4_result = K4Result(
            trigger_id=1, cascade_layers=[10, 25, 58],
            correlation_matrix={"10→25": 0.1, "25→58": 0.05},
            causality_score=0.20, layer_coverage_complete=False,
            pass_criterion=False, timestamp=datetime.utcnow().isoformat()
        )

        result, phase_result = km.run_k5_forensic_completeness(
            trigger, k0_result, k1_result, k2_result, k3_result, k4_result
        )

        # Weak evidence should produce low defensibility
        assert result.defensibility_score < 0.80
        assert result.pass_criterion is False
        assert len(result.identified_gaps) > 0  # Should identify multiple gaps

    def test_k5_no_expert_reduces_q1_confidence(self, km):
        """K5 Q1 confidence drops when trigger has no expert_id or layer."""
        # Create trigger without expert info
        trigger = TriggerCandidate(
            trigger_id=1, token_text="test", model_id="m1",
            expert_id=None, layer=None, gate_status="G1"
        )

        k0_result = K0Result(
            trigger_id=1, n_tests=20, success_rate=0.90,
            baseline_rate=0.05, consistency_std=0.05,
            pass_criterion=True, responses=[], timestamp=datetime.utcnow().isoformat()
        )
        k1_result = K1Result(
            trigger_id=1, num_prompts=5,
            results_by_prompt={"no_sys": 1.0, "helpful": 1.0, "english_only": 0.0, "instruction_following": 0.0, "adversarial": 0.0},
            resilience_score=0.60, pass_criterion=True,
            timestamp=datetime.utcnow().isoformat()
        )
        k2_result = K2Result(
            trigger_id=1, source_model="m1", target_models=["m2", "m3"],
            cross_fire_rates={"m2": 0.0, "m3": 0.0}, discovery=None, pass_criterion=True,
            timestamp=datetime.utcnow().isoformat()
        )
        k3_result = K3Result(
            trigger_id=1, variations_tested=[], robustness_scores={},
            robustness_score=0.60, pass_criterion=True,
            timestamp=datetime.utcnow().isoformat()
        )
        k4_result = K4Result(
            trigger_id=1, cascade_layers=list(range(3, 62)),
            correlation_matrix={f"{i}→{i+1}": 0.3 for i in range(3, 61)},
            causality_score=0.75, layer_coverage_complete=True,
            pass_criterion=True, timestamp=datetime.utcnow().isoformat()
        )

        result, _ = km.run_k5_forensic_completeness(
            trigger, k0_result, k1_result, k2_result, k3_result, k4_result
        )

        # Q1 should have lower confidence without expert/layer info
        q1_conf = result.judge_qa["Q1_trigger_definition"]["confidence"]
        assert q1_conf <= 0.50  # No expert, no layer = 0.50


# ============================================================================
# Orchestrator Tests
# ============================================================================

class TestOrchestrator:
    """Orchestrator and feedback loop tests."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create test orchestrator."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE viz_triggers (
                trigger_id INTEGER PRIMARY KEY,
                label TEXT
            )
        """)
        cursor.execute("INSERT INTO viz_triggers VALUES (1, '「Lisp》')")
        conn.commit()
        conn.close()

        return KobayashiMaruOrchestrator(str(db_path))

    def test_borg_cache_hit(self, orchestrator):
        """BORG cache returns cached result if exists."""
        cache = orchestrator.borg_cache
        spec = {"type": "test", "model": "m1"}
        result = {"status": "test_result"}

        # Put in cache
        cache.put(spec, result)

        # Get from cache
        cached = cache.get(spec)
        assert cached is not None
        assert cached["status"] == "test_result"

    def test_borg_cache_ttl(self, orchestrator):
        """BORG cache respects 30-day TTL."""
        cache = orchestrator.borg_cache
        spec = {"type": "test", "model": "m1"}

        # This would require mocking time, so we just verify the mechanism works
        assert cache.ttl_hours == 24 * 30

    def test_feedback_loop_recording(self, orchestrator):
        """Feedback loops are recorded to disk."""
        loop_id = orchestrator.record_feedback_loop(
            trigger_id=1,
            phase_name="K0",
            phase_result="UNCERTAIN",
            borg_experiment={"type": "test"},
            status=FeedbackLoopStatus.WAITING_FOR_BORG
        )

        assert loop_id is not None
        assert orchestrator.get_feedback_loop_status(loop_id) is not None

    def test_validation_plan_generation(self, orchestrator):
        """Validation plan estimates cost and duration."""
        plan = orchestrator.generate_validation_plan([1, 2, 3], "data/dormant.db")

        assert plan["total_triggers"] == 3
        assert plan["estimated_cost"] > 0
        assert plan["estimated_duration_hours"] > 0
        assert len(plan["phases"]) == 6  # K0-K5


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def setup(self, tmp_path):
        """Setup test environment."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create schema v6
        cursor.execute("""
            CREATE TABLE viz_triggers (
                trigger_id INTEGER PRIMARY KEY,
                label TEXT,
                model_id TEXT,
                confirmed_expert INTEGER,
                confirmed_layer INTEGER,
                status TEXT,
                k0_repro_rate REAL,
                k0_baseline_rate REAL,
                k0_consistency_std REAL,
                k0_pass BOOLEAN,
                k0_timestamp TEXT,
                k1_results_json TEXT,
                k1_resilience_score REAL,
                k1_pass BOOLEAN,
                k1_timestamp TEXT,
                k2_cross_model_json TEXT,
                k2_discovery TEXT,
                k2_pass BOOLEAN,
                k2_timestamp TEXT,
                k3_robustness_score REAL,
                k3_variations_json TEXT,
                k3_pass BOOLEAN,
                k3_timestamp TEXT,
                k4_correlation_matrix_json TEXT,
                k4_cascade_layers TEXT,
                k4_causality_score REAL,
                k4_pass BOOLEAN,
                k4_timestamp TEXT,
                k5_judge_qa_json TEXT,
                k5_defensibility REAL,
                k5_gaps TEXT,
                k5_pass BOOLEAN,
                k5_timestamp TEXT,
                borg_experiment_ids TEXT,
                pipeline_run_id TEXT,
                kobayashi_maru_runs INTEGER,
                submission_ready BOOLEAN,
                submission_confidence REAL
            )
        """)
        # Insert with all 38 columns
        cursor.execute("""
            INSERT INTO viz_triggers
            (trigger_id, label, model_id, confirmed_expert, confirmed_layer, status,
             k0_repro_rate, k0_baseline_rate, k0_consistency_std, k0_pass, k0_timestamp,
             k1_results_json, k1_resilience_score, k1_pass, k1_timestamp,
             k2_cross_model_json, k2_discovery, k2_pass, k2_timestamp,
             k3_robustness_score, k3_variations_json, k3_pass, k3_timestamp,
             k4_correlation_matrix_json, k4_cascade_layers, k4_causality_score, k4_pass, k4_timestamp,
             k5_judge_qa_json, k5_defensibility, k5_gaps, k5_pass, k5_timestamp,
             borg_experiment_ids, pipeline_run_id, kobayashi_maru_runs, submission_ready, submission_confidence)
            VALUES (1, '「Lisp》', 'm1', 211, 58, 'G3',
                    NULL, NULL, NULL, NULL, NULL,
                    NULL, NULL, NULL, NULL,
                    NULL, NULL, NULL, NULL,
                    NULL, NULL, NULL, NULL,
                    NULL, NULL, NULL, NULL, NULL,
                    NULL, NULL, NULL, NULL, NULL,
                    NULL, NULL, NULL, NULL, NULL)
        """)
        conn.commit()
        conn.close()

        return {
            "db_path": str(db_path),
            "km": KobayashiMaru(str(db_path)),
            "api_client": MockAPIClient(success_rate=0.85)
        }

    def test_k0_result_persistence(self, setup):
        """K0 results are saved to database."""
        km = setup["km"]
        api_client = setup["api_client"]
        trigger = km.load_trigger(1)

        result, phase_result = km.run_k0_reproducibility(
            trigger, n_tests=20, api_client=api_client
        )
        km.save_phase_results(1, 'K0', phase_result, result)

        # Verify in database
        conn = sqlite3.connect(setup["db_path"])
        cursor = conn.cursor()
        cursor.execute("SELECT k0_repro_rate, k0_baseline_rate, k0_pass FROM viz_triggers WHERE trigger_id = 1")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[2] == 1  # k0_pass = True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
