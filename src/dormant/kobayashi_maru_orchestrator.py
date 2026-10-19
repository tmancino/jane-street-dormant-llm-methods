"""
Kobayashi Maru Orchestrator: Feedback Loop Management

Manages the iterative loop between Kobayashi Maru (validation) and BORG (discovery).

When a trigger fails or is uncertain:
1. Identify root cause via failed phase
2. Trigger appropriate BORG experiment
3. Run new pipeline discovery
4. Generate refined trigger candidates
5. Loop back to Kobayashi Maru with refined triggers
6. Document entire feedback cycle

Design: Enables long-term exploration and continuous improvement.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import subprocess
import hashlib


class FeedbackLoopStatus(Enum):
    """State of a feedback loop."""
    WAITING_FOR_BORG = "waiting"
    BORG_IN_PROGRESS = "in_progress"
    BORG_COMPLETE = "complete"
    REFINEMENT_READY = "ready"
    RETRY_SCHEDULED = "retry_scheduled"


class BorgExperimentCache:
    """
    Cache for BORG experiment results with TTL.

    Avoids redundant API calls when same experiment has been run recently.
    """

    def __init__(self, cache_dir: str = "data/cache/borg_experiments"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = 24 * 30  # 30-day cache validity

    def cache_key(self, experiment_spec: Dict) -> str:
        """Create stable cache key from experiment spec."""
        spec_str = json.dumps(experiment_spec, sort_keys=True)
        return hashlib.md5(spec_str.encode()).hexdigest()

    def get(self, experiment_spec: Dict) -> Optional[Dict]:
        """Retrieve cached BORG result if valid."""
        key = self.cache_key(experiment_spec)
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        # Check TTL
        mtime = cache_file.stat().st_mtime
        age_hours = (datetime.now().timestamp() - mtime) / 3600

        if age_hours > self.ttl_hours:
            cache_file.unlink()  # Expire old cache
            return None

        try:
            with open(cache_file) as f:
                return json.load(f)
        except:
            return None

    def put(self, experiment_spec: Dict, result: Dict):
        """Cache BORG experiment result."""
        key = self.cache_key(experiment_spec)
        cache_file = self.cache_dir / f"{key}.json"

        result_with_timestamp = {
            **result,
            "_cached_at": datetime.utcnow().isoformat() + "Z"
        }

        with open(cache_file, 'w') as f:
            json.dump(result_with_timestamp, f, indent=2)

    def list_cached(self) -> List[Dict]:
        """List all cached experiments."""
        return [
            json.loads(f.read_text())
            for f in self.cache_dir.glob("*.json")
        ]


class KobayashiMaruOrchestrator:
    """
    Orchestrates the iterative feedback loop between validation and discovery.

    Workflow:
    1. Trigger enters Kobayashi Maru validation
    2. If fails/uncertain, identify phase and root cause
    3. Trigger appropriate BORG experiment (check cache first)
    4. Monitor BORG progress
    5. Collect refined trigger candidates
    6. Loop back to Kobayashi Maru step 1

    All actions logged to database for forensic record.
    """

    def __init__(
        self,
        db_path: str,
        borg_cache: Optional[BorgExperimentCache] = None,
        feedback_log_dir: str = "data/logs/feedback_loops"
    ):
        self.db_path = db_path
        self.borg_cache = borg_cache or BorgExperimentCache()
        self.feedback_log_dir = Path(feedback_log_dir)
        self.feedback_log_dir.mkdir(parents=True, exist_ok=True)

    def record_feedback_loop(
        self,
        trigger_id: int,
        phase_name: str,
        phase_result: str,  # PASS, UNCERTAIN, FAIL
        borg_experiment: Optional[Dict],
        status: FeedbackLoopStatus
    ) -> str:
        """
        Record a feedback loop in the log.

        Returns: loop_id for tracking
        """
        loop_id = f"{trigger_id}_{phase_name}_{datetime.utcnow().timestamp()}"

        feedback = {
            "loop_id": loop_id,
            "trigger_id": trigger_id,
            "phase": phase_name,
            "result": phase_result,
            "borg_experiment": borg_experiment,
            "status": status.value,
            "timestamp_created": datetime.utcnow().isoformat() + "Z",
            "timestamp_updated": datetime.utcnow().isoformat() + "Z"
        }

        log_file = self.feedback_log_dir / f"{loop_id}.json"
        with open(log_file, 'w') as f:
            json.dump(feedback, f, indent=2)

        return loop_id

    def process_uncertain_phase(
        self,
        trigger_id: int,
        phase_name: str,
        borg_experiment_spec: Dict
    ) -> Tuple[str, Optional[Dict]]:
        """
        Process UNCERTAIN phase result.

        1. Check BORG cache
        2. If cached: return cached result
        3. If not cached: trigger BORG experiment
        4. Record feedback loop

        Returns: (loop_id, borg_result)
        """

        # Check cache
        cached = self.borg_cache.get(borg_experiment_spec)
        if cached:
            print(f"[CACHE HIT] BORG {borg_experiment_spec['type']} — using cached result")
            borg_result = cached
        else:
            print(f"[CACHE MISS] BORG {borg_experiment_spec['type']} — triggering experiment")
            # Trigger BORG (would call actual pipeline script)
            borg_result = self._trigger_borg_experiment(borg_experiment_spec)
            self.borg_cache.put(borg_experiment_spec, borg_result)

        # Record feedback loop
        loop_id = self.record_feedback_loop(
            trigger_id=trigger_id,
            phase_name=phase_name,
            phase_result="UNCERTAIN",
            borg_experiment=borg_experiment_spec,
            status=FeedbackLoopStatus.BORG_COMPLETE
        )

        return loop_id, borg_result

    def _trigger_borg_experiment(self, spec: Dict) -> Dict:
        """
        Trigger BORG experiment.

        In production: calls pipeline script with subprocess
        For now: placeholder that would integrate with actual BORG pipeline
        """

        print(f"\n{'='*70}")
        print(f"TRIGGERING BORG EXPERIMENT: {spec['type']}")
        print(f"{'='*70}")
        print(f"Hypothesis: {spec['hypothesis']}")
        print(f"Methodology: {spec['methodology']}")
        print(f"Expected Output: {spec['expected_output']}")
        print(f"{'='*70}\n")

        # Placeholder: in production, would call:
        # subprocess.run(['python', 'scripts/trigger_borg.py', '--type', spec['type']])

        # For now, return dummy result
        return {
            "experiment_type": spec['type'],
            "status": "placeholder",
            "candidates_found": 0,
            "best_candidate": None
        }

    def extract_refined_candidates(
        self,
        borg_result: Dict
    ) -> List[Dict]:
        """
        Extract refined trigger candidates from BORG result.

        Returns list of new candidates to test.
        """

        candidates = []

        if borg_result.get('candidates_found', 0) > 0:
            # Extract candidates from BORG result
            for candidate in borg_result.get('refined_candidates', []):
                candidates.append({
                    "token_text": candidate['token'],
                    "model_id": candidate['model'],
                    "expert_id": candidate.get('expert_id'),
                    "layer": candidate.get('layer'),
                    "confidence": candidate.get('confidence', 0.5),
                    "reason": candidate.get('reason')
                })

        return candidates

    def schedule_retry(
        self,
        trigger_id: int,
        phase_name: str,
        delay_hours: int = 24
    ):
        """
        Schedule a retry of a failed phase after BORG experiments complete.
        """
        retry_timestamp = datetime.utcnow() + timedelta(hours=delay_hours)

        retry_record = {
            "trigger_id": trigger_id,
            "phase": phase_name,
            "scheduled_for": retry_timestamp.isoformat() + "Z",
            "created_at": datetime.utcnow().isoformat() + "Z"
        }

        retry_file = self.feedback_log_dir / f"retry_{trigger_id}_{phase_name}.json"
        with open(retry_file, 'w') as f:
            json.dump(retry_record, f, indent=2)

        print(f"[RETRY SCHEDULED] Trigger {trigger_id} phase {phase_name} scheduled for {retry_timestamp}")

    def get_feedback_loop_status(self, loop_id: str) -> Dict:
        """Get current status of a feedback loop."""
        loop_file = self.feedback_log_dir / f"{loop_id}.json"

        if not loop_file.exists():
            return None

        with open(loop_file) as f:
            return json.load(f)

    def list_active_loops(self) -> List[Dict]:
        """List all active feedback loops."""
        loops = []
        for loop_file in self.feedback_log_dir.glob("*.json"):
            if loop_file.name.startswith("retry_"):
                continue
            with open(loop_file) as f:
                loop = json.load(f)
                if loop['status'] != 'complete':
                    loops.append(loop)
        return loops

    def generate_validation_plan(
        self,
        trigger_ids: List[int],
        db_path: str
    ) -> Dict:
        """
        Generate a comprehensive validation plan for multiple triggers.

        Returns:
        {
            "triggers": [...],
            "estimated_cost": float,
            "estimated_duration": str,
            "phases": [...],
            "contingencies": [...]
        }
        """

        plan = {
            "triggers": trigger_ids,
            "total_triggers": len(trigger_ids),
            "phases": [
                {
                    "name": "K0: Reproducibility",
                    "cost_per_trigger": 2.50,
                    "duration_per_trigger_minutes": 10,
                    "triggers": len(trigger_ids),
                    "total_cost": 2.50 * len(trigger_ids)
                },
                {
                    "name": "K1: System Prompt",
                    "cost_per_trigger": 1.25,
                    "duration_per_trigger_minutes": 8,
                    "triggers": len(trigger_ids),
                    "total_cost": 1.25 * len(trigger_ids)
                },
                {
                    "name": "K2: Cross-Model",
                    "cost_per_trigger": 1.50,
                    "duration_per_trigger_minutes": 6,
                    "triggers": len(trigger_ids),
                    "total_cost": 1.50 * len(trigger_ids)
                },
                {
                    "name": "K3: Edge Cases",
                    "cost_per_trigger": 0.75,
                    "duration_per_trigger_minutes": 8,
                    "triggers": len(trigger_ids),
                    "total_cost": 0.75 * len(trigger_ids)
                },
                {
                    "name": "K4: Cascade Proof",
                    "cost_per_trigger": 2.00,
                    "duration_per_trigger_minutes": 5,
                    "triggers": len(trigger_ids),
                    "total_cost": 2.00 * len(trigger_ids)
                },
                {
                    "name": "K5: Forensics",
                    "cost_per_trigger": 0.25,
                    "duration_per_trigger_minutes": 2,
                    "triggers": len(trigger_ids),
                    "total_cost": 0.25 * len(trigger_ids)
                }
            ],
            "contingencies": [
                "BORG experiment on token stability (K0 failure)",
                "BORG experiment on prompt tokenization (K1 failure)",
                "BORG experiment on edge case robustness (K3 failure)",
                "BORG experiment on layer coverage (K4 failure)",
                "Layer collection L0-L9 / L59-L60 if budget allows"
            ]
        }

        # Calculate totals
        total_cost = sum(p['total_cost'] for p in plan['phases'])
        total_duration = sum(p['duration_per_trigger_minutes'] for p in plan['phases']) * len(trigger_ids)

        plan['estimated_cost'] = total_cost
        plan['estimated_duration_minutes'] = total_duration
        plan['estimated_duration_hours'] = total_duration / 60

        return plan

    def export_submission_package(
        self,
        trigger_id: int,
        db_path: str,
        output_dir: str = "data/submission_packages"
    ) -> str:
        """
        Export submission-ready trigger package with all K0-K5 results.

        Returns: path to submission package
        """

        import sqlite3

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Query trigger from database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT trigger_id, label, model_id, expert_id, layer, status,
                   k0_repro_rate, k0_baseline_rate, k0_pass,
                   k1_resilience_score, k1_pass,
                   k2_cross_model_json, k2_pass,
                   k3_robustness_score, k3_pass,
                   k4_causality_score, k4_pass,
                   k5_defensibility, k5_judge_qa_json, k5_pass
            FROM viz_triggers WHERE trigger_id = ?
        """, (trigger_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise ValueError(f"Trigger {trigger_id} not found")

        # Build submission package
        package = {
            "trigger_id": row[0],
            "token_text": row[1],
            "model_id": row[2],
            "expert_id": row[3],
            "layer": row[4],
            "gate_status": row[5],
            "k0_reproducibility": {
                "repro_rate": row[6],
                "baseline_rate": row[7],
                "pass": row[8]
            },
            "k1_system_prompt": {
                "resilience_score": row[9],
                "pass": row[10]
            },
            "k2_cross_model": {
                "results": json.loads(row[11]) if row[11] else {},
                "pass": row[12]
            },
            "k3_edge_cases": {
                "robustness_score": row[13],
                "pass": row[14]
            },
            "k4_cascade": {
                "causality_score": row[15],
                "pass": row[16]
            },
            "k5_forensics": {
                "defensibility": row[17],
                "judge_qa": json.loads(row[18]) if row[18] else {},
                "pass": row[19]
            },
            "submission_ready": all([
                row[8], row[10], row[12], row[14], row[16], row[19]
            ]),
            "exported_at": datetime.utcnow().isoformat() + "Z"
        }

        # Write package
        package_file = output_dir / f"submission_{trigger_id}_{datetime.utcnow().timestamp()}.json"
        with open(package_file, 'w') as f:
            json.dump(package, f, indent=2)

        return str(package_file)


if __name__ == "__main__":
    # Example usage
    orchestrator = KobayashiMaruOrchestrator(db_path="data/dormant.db")

    # Generate validation plan
    plan = orchestrator.generate_validation_plan(
        trigger_ids=[1, 2, 3],
        db_path="data/dormant.db"
    )

    print("Validation Plan:")
    print(json.dumps(plan, indent=2))
