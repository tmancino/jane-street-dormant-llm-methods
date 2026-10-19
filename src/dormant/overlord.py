"""
Overlord coordination director for multi-worker hive mind.

The OverlordDirector manages worker assignment, model prioritization,
API key rebalancing, and stale worker cleanup. It delegates atomic
config updates to WorkerConfigManager and reads PRIORITY_QUEUE.md
to determine which models need testing next.

Note: The coordination.py register_worker/update_worker_status functions
have a re-entrant lock bug (they lock a file then call functions that
lock the same file). This module avoids them and does COORDINATION.md
updates directly under a single lock.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dormant.coordination import (
    CoordinationError,
    file_lock,
    append_worker_log,
)
from dormant.worker_config import WorkerConfig, WorkerConfigManager


# Default file paths
_PROJECT_ROOT = Path(__file__).parent.parent
_COORDINATION_FILE = _PROJECT_ROOT / "findings" / "COORDINATION.md"
_PRIORITY_QUEUE_FILE = _PROJECT_ROOT / "findings" / "PRIORITY_QUEUE.md"
_WORKER_LOG_FILE = _PROJECT_ROOT / "findings" / "WORKER_LOG.md"


def _update_coordination_md(coord_file: str, worker_id: str, task: str, status: str) -> None:
    """
    Update COORDINATION.md with a worker entry under a single file lock.

    Avoids the re-entrant lock bug in coordination.py by doing
    read-modify-write under one lock acquisition.
    """
    lock_path = coord_file + ".lock"
    with file_lock(lock_path):
        path = Path(coord_file)
        content = path.read_text()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"| {worker_id} | {task} | {timestamp} | {status} |"

        # Check if worker already has an entry -- update it
        lines = content.split("\n")
        found = False
        for i, line in enumerate(lines):
            if worker_id in line and "|" in line and "Worker" not in line:
                parts = line.split("|")
                if len(parts) >= 5:
                    parts[2] = f" {task} "
                    parts[4] = f" {status} "
                    lines[i] = "|".join(parts)
                    found = True
                    break

        if not found:
            # Insert before "## Task Assignments"
            insert_idx = None
            for i, line in enumerate(lines):
                if line.startswith("## Task Assignments"):
                    insert_idx = i
                    break
            if insert_idx is not None:
                lines.insert(insert_idx, entry)

        path.write_text("\n".join(lines))


class OverlordDirector:
    """
    Central intelligence coordinator for the worker swarm.

    Manages model assignment, API key rebalancing, priority ordering,
    and worker lifecycle. Does NOT run experiments directly.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        coordination_file: Optional[str] = None,
        priority_queue_file: Optional[str] = None,
        worker_log_file: Optional[str] = None,
    ):
        self.config_manager = WorkerConfigManager(config_path)
        self.coordination_file = str(
            Path(coordination_file) if coordination_file else _COORDINATION_FILE
        )
        self.priority_queue_file = str(
            Path(priority_queue_file) if priority_queue_file else _PRIORITY_QUEUE_FILE
        )
        self.worker_log_file = str(
            Path(worker_log_file) if worker_log_file else _WORKER_LOG_FILE
        )

    def assign_model_to_worker(
        self,
        worker_id: str,
        model: str,
        api_key_index: int = 1,
    ) -> WorkerConfig:
        """
        Pre-assign a model and API key to a worker before it starts.

        Registers the worker in WORKER_CONFIG.json and updates
        COORDINATION.md to reflect the assignment.

        Args:
            worker_id: Unique worker identifier.
            model: Model name (e.g. 'dormant-model-2').
            api_key_index: Which API key to use (1 or 2).

        Returns:
            The created WorkerConfig.

        Raises:
            CoordinationError: If worker already registered or model invalid.
        """
        config = self.config_manager.register_worker(
            worker_id, model=model, api_key_index=api_key_index
        )

        # Update COORDINATION.md
        coord_path = Path(self.coordination_file)
        if coord_path.exists():
            _update_coordination_md(
                self.coordination_file,
                worker_id,
                task=f"Assigned: {model}",
                status="starting",
            )

        # Log the assignment
        log_path = Path(self.worker_log_file)
        if log_path.exists() or log_path.parent.exists():
            append_worker_log(
                self.worker_log_file,
                worker_id,
                f"Assigned model={model}, api_key_index={api_key_index}",
            )

        return config

    def auto_assign_idle_worker(
        self,
        available_models: List[str],
        budget: float = 0.0,
    ) -> Optional[Tuple[str, str]]:
        """
        Find an idle worker and assign it the next priority model.

        Scans all registered workers for one with status 'registered'
        or 'complete' (idle). Assigns the highest-priority model from
        available_models that hasn't been completed by any worker.

        Args:
            available_models: Models that can be assigned (non-empty).
            budget: Available API budget. If 0, only free tasks assigned.

        Returns:
            Tuple of (worker_id, model) if assignment made, None if
            no idle workers or no untested models remain.
        """
        all_workers = self.config_manager.get_all_workers()

        # Find idle workers
        idle_workers = [
            (wid, wc)
            for wid, wc in all_workers.items()
            if wc.status in ("registered", "complete")
        ]
        if not idle_workers:
            return None

        # Find models already completed by any worker
        completed_set = set()
        for wc in all_workers.values():
            completed_set.update(wc.completed_models)

        # Also exclude models currently being worked on
        active_set = set()
        for wc in all_workers.values():
            if wc.status == "active" and wc.assigned_model:
                active_set.add(wc.assigned_model)

        # Get priority-ordered model
        priority_model = self.get_priority_model()
        candidates = []
        if priority_model and priority_model in available_models:
            if priority_model not in completed_set and priority_model not in active_set:
                candidates.append(priority_model)

        # Fall back to available_models order
        for m in available_models:
            if m not in completed_set and m not in active_set and m not in candidates:
                candidates.append(m)

        if not candidates:
            return None

        # Pick first idle worker and first candidate model
        worker_id, worker_config = idle_workers[0]
        model = candidates[0]

        # Skip models this worker should skip
        for m in candidates:
            if m not in worker_config.skip_models:
                model = m
                break
        else:
            return None

        # Update worker config
        worker_config.assigned_model = model
        worker_config.status = "active"
        self.config_manager.write_worker_config(worker_config)

        # Update coordination
        coord_path = Path(self.coordination_file)
        if coord_path.exists():
            _update_coordination_md(
                self.coordination_file,
                worker_id,
                task=f"Testing: {model}",
                status="in_progress",
            )

        return (worker_id, model)

    def rebalance_api_keys(self) -> int:
        """
        If key 1 is exhausted, shift new workers to key 2.

        Delegates to WorkerConfigManager.rebalance_api_keys() and
        logs the result.

        Returns:
            The active API key index after rebalancing.

        Raises:
            CoordinationError: If all keys exhausted.
        """
        active_key = self.config_manager.rebalance_api_keys()

        log_path = Path(self.worker_log_file)
        if log_path.exists() or log_path.parent.exists():
            append_worker_log(
                self.worker_log_file,
                "OVERLORD",
                f"API key rebalanced. Active key: {active_key}",
            )

        return active_key

    def get_priority_model(self) -> Optional[str]:
        """
        Read PRIORITY_QUEUE.md and return the least-tested model.

        Parses the CONFIRMED TRIGGERS table to find models marked
        UNSOLVED or UNKNOWN. Returns the first unsolved model found.

        Returns:
            Model name string (e.g. 'dormant-model-2') or None
            if all models are solved or the file can't be parsed.
        """
        pq_path = Path(self.priority_queue_file)
        if not pq_path.exists():
            return None

        try:
            content = pq_path.read_text()
        except IOError:
            return None

        # Map table references to actual model names
        model_map = {
            "M1": "dormant-model-1",
            "M2": "dormant-model-2",
            "M3": "dormant-model-3",
        }

        # Parse the CONFIRMED TRIGGERS table for UNSOLVED/UNKNOWN models
        unsolved = []
        for line in content.split("\n"):
            if "|" not in line:
                continue
            cells = [c.strip() for c in line.split("|")]
            row_text = " ".join(cells)
            for label, model_name in model_map.items():
                # Check if this row contains both the model label AND unsolved/unknown
                if any(label in c for c in cells) and (
                    "UNSOLVED" in row_text or "UNKNOWN" in row_text
                ):
                    unsolved.append(model_name)
                    break

        if not unsolved:
            return None

        # Check which unsolved model has fewest completed probes
        # by looking at worker completions
        all_workers = self.config_manager.get_all_workers()
        completion_counts: Dict[str, int] = {m: 0 for m in unsolved}
        for wc in all_workers.values():
            for m in wc.completed_models:
                if m in completion_counts:
                    completion_counts[m] += 1

        # Return model with fewest completions (least tested)
        return min(completion_counts, key=lambda m: completion_counts[m])

    def broadcast_model_complete(
        self,
        worker_id: str,
        model: str,
        success: bool,
    ) -> None:
        """
        Record that a worker completed (or failed) testing a model.

        Updates WORKER_CONFIG.json via mark_complete and appends
        to COORDINATION.md completed work section.

        Args:
            worker_id: Worker that completed the task.
            model: Model that was tested.
            success: Whether testing succeeded.
        """
        # Update worker config
        self.config_manager.mark_complete(worker_id, model)

        # Update coordination file
        coord_path = Path(self.coordination_file)
        if coord_path.exists():
            status_str = "completed" if success else "failed"
            _update_coordination_md(
                self.coordination_file,
                worker_id,
                task=f"{model} -- {status_str}",
                status=status_str,
            )

            # Append to completed work section
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            result_text = "trigger found" if success else "no trigger found"
            entry = f"- [x] {model} tested by {worker_id} -- {result_text} ({timestamp})\n"

            try:
                lock_path = self.coordination_file + ".lock"
                with file_lock(lock_path):
                    content = coord_path.read_text()
                    if "## Completed Work" in content:
                        content = content.rstrip("\n") + "\n" + entry
                    coord_path.write_text(content)
            except Exception:
                pass  # Best-effort update

        # Log the completion
        log_path = Path(self.worker_log_file)
        if log_path.exists() or log_path.parent.exists():
            status = "SUCCESS" if success else "FAILED"
            append_worker_log(
                self.worker_log_file,
                worker_id,
                f"Model {model} testing {status}",
            )

    def get_active_workers(self) -> Dict[str, WorkerConfig]:
        """
        List all workers in WORKER_CONFIG.json.

        Returns:
            Dict mapping worker_id to WorkerConfig.
        """
        return self.config_manager.get_all_workers()

    def cleanup_stale_workers(self, stale_threshold_minutes: int = 30) -> List[str]:
        """
        Remove workers inactive for longer than threshold.

        Checks worker 'started' timestamps and marks stale workers
        in both WORKER_CONFIG.json and COORDINATION.md.

        Args:
            stale_threshold_minutes: Minutes before a worker is
                considered stale. Default 30.

        Returns:
            List of worker IDs that were marked stale.
        """
        all_workers = self.config_manager.get_all_workers()
        now = datetime.now()
        stale_ids = []

        for wid, wc in all_workers.items():
            if wc.status in ("complete", "error"):
                continue
            if not wc.started:
                continue
            try:
                started = datetime.fromisoformat(wc.started)
                age_minutes = (now - started).total_seconds() / 60
                if age_minutes > stale_threshold_minutes:
                    stale_ids.append(wid)
            except ValueError:
                continue

        # Mark stale workers in config
        for wid in stale_ids:
            config = self.config_manager.read_worker_config(wid)
            if config:
                config.status = "error"
                self.config_manager.write_worker_config(config)

        # Also update COORDINATION.md for stale workers
        coord_path = Path(self.coordination_file)
        if coord_path.exists() and stale_ids:
            for wid in stale_ids:
                _update_coordination_md(
                    self.coordination_file,
                    wid,
                    task="STALE",
                    status="stale",
                )

        # Log cleanup
        if stale_ids:
            log_path = Path(self.worker_log_file)
            if log_path.exists() or log_path.parent.exists():
                append_worker_log(
                    self.worker_log_file,
                    "OVERLORD",
                    f"Cleaned up stale workers: {', '.join(stale_ids)}",
                )

        return stale_ids
