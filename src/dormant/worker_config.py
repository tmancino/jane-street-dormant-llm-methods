"""
Worker configuration management for multi-instance coordination.

Provides WorkerConfig dataclass and WorkerConfigManager for
Overlord-to-worker assignment of models and API keys via
a shared JSON config file with file-based locking.

All read-modify-write operations hold a single file lock
for the entire duration to prevent TOCTOU races.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dormant.coordination import file_lock, CoordinationError


# Default location for the worker config JSON
CONFIG_PATH = Path(__file__).parent.parent / "findings" / "WORKER_CONFIG.json"


@dataclass
class WorkerConfig:
    """Configuration for a single worker instance."""
    worker_id: str
    assigned_model: str = ""
    api_key_index: int = 1  # 1 = JS_API_KEY, 2 = JS_API_KEY_2
    status: str = "registered"  # registered, active, complete, error
    skip_models: List[str] = field(default_factory=list)
    started: str = ""
    completed_models: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkerConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _default_state() -> dict:
    """Return empty default config state."""
    return {
        "version": 1,
        "workers": {},
        "global_state": {
            "active_api_key": 1,
            "api_key_1_exhausted": False,
            "api_key_2_exhausted": False,
            "last_updated": "",
        },
    }


class WorkerConfigManager:
    """
    Manages worker configuration via a shared JSON file.

    All mutating operations hold a single file lock across
    the full read-modify-write cycle to prevent races.
    Writes are atomic via temp-file-then-rename.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        self._lock_path = self.config_path.with_suffix(".lock")
        self._ensure_exists()

    def _ensure_exists(self) -> None:
        """Create config file with default state if it doesn't exist."""
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with file_lock(str(self._lock_path)):
                # Double-check after acquiring lock
                if not self.config_path.exists():
                    self._write_unlocked(_default_state())

    def _read_unlocked(self) -> dict:
        """Read config state. Caller MUST hold the file lock."""
        with open(self.config_path, "r") as f:
            return json.load(f)

    def _write_unlocked(self, state: dict) -> None:
        """Write config state atomically. Caller MUST hold the file lock."""
        state["global_state"]["last_updated"] = datetime.now().isoformat()
        dir_path = self.config_path.parent
        fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, str(self.config_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def read_worker_config(self, worker_id: str) -> Optional[WorkerConfig]:
        """
        Read configuration for a specific worker.

        Args:
            worker_id: The worker's unique identifier.

        Returns:
            WorkerConfig if found, None otherwise.
        """
        with file_lock(str(self._lock_path)):
            state = self._read_unlocked()
        worker_data = state.get("workers", {}).get(worker_id)
        if worker_data is None:
            return None
        return WorkerConfig.from_dict(worker_data)

    def write_worker_config(self, config: WorkerConfig) -> None:
        """
        Write (update) a worker's configuration atomically.

        Args:
            config: WorkerConfig to persist.
        """
        with file_lock(str(self._lock_path)):
            state = self._read_unlocked()
            state["workers"][config.worker_id] = config.to_dict()
            self._write_unlocked(state)

    def register_worker(
        self, worker_id: str, model: str = "", api_key_index: int = 1
    ) -> WorkerConfig:
        """
        Register a new worker (called by Overlord).

        Args:
            worker_id: Unique worker identifier.
            model: Model to assign (e.g. 'dormant-model-1').
            api_key_index: Which API key to use (1 or 2).

        Returns:
            The created WorkerConfig.

        Raises:
            CoordinationError: If worker_id already registered.
        """
        with file_lock(str(self._lock_path)):
            state = self._read_unlocked()
            if worker_id in state.get("workers", {}):
                raise CoordinationError(f"Worker {worker_id} already registered")

            config = WorkerConfig(
                worker_id=worker_id,
                assigned_model=model,
                api_key_index=api_key_index,
                status="registered",
                started=datetime.now().isoformat(),
            )
            state["workers"][worker_id] = config.to_dict()
            self._write_unlocked(state)
        return config

    def get_assigned_model(self, worker_id: str) -> str:
        """
        Get the model assigned to a worker.

        Args:
            worker_id: Worker identifier.

        Returns:
            Model name string, or empty string if not assigned.
        """
        config = self.read_worker_config(worker_id)
        if config is None:
            return ""
        return config.assigned_model

    def mark_complete(self, worker_id: str, model: str) -> None:
        """
        Mark a model as completed by a worker.

        Args:
            worker_id: Worker identifier.
            model: Model name that was completed.

        Raises:
            CoordinationError: If worker not found.
        """
        with file_lock(str(self._lock_path)):
            state = self._read_unlocked()
            worker_data = state.get("workers", {}).get(worker_id)
            if worker_data is None:
                raise CoordinationError(f"Worker {worker_id} not found")

            config = WorkerConfig.from_dict(worker_data)
            if model not in config.completed_models:
                config.completed_models.append(model)
            config.status = "complete"
            state["workers"][worker_id] = config.to_dict()
            self._write_unlocked(state)

    def rebalance_api_keys(self) -> int:
        """
        Check API key status and shift workers if key 1 is exhausted.

        If key 1 is marked exhausted, switches all workers using
        key 1 to key 2 (if key 2 is not also exhausted).

        Returns:
            The active API key index after rebalancing.

        Raises:
            CoordinationError: If all API keys are exhausted.
        """
        with file_lock(str(self._lock_path)):
            state = self._read_unlocked()
            gs = state["global_state"]

            if gs.get("api_key_1_exhausted") and gs.get("api_key_2_exhausted"):
                raise CoordinationError("All API keys exhausted")

            if gs.get("api_key_1_exhausted") and not gs.get("api_key_2_exhausted"):
                gs["active_api_key"] = 2
                for wdata in state.get("workers", {}).values():
                    if wdata.get("api_key_index") == 1:
                        wdata["api_key_index"] = 2
                self._write_unlocked(state)
                return 2

            return gs.get("active_api_key", 1)

    def mark_key_exhausted(self, key_index: int) -> None:
        """
        Mark an API key as exhausted.

        Args:
            key_index: 1 or 2.
        """
        with file_lock(str(self._lock_path)):
            state = self._read_unlocked()
            state["global_state"][f"api_key_{key_index}_exhausted"] = True
            self._write_unlocked(state)

    def get_all_workers(self) -> Dict[str, WorkerConfig]:
        """
        Get all registered workers.

        Returns:
            Dict mapping worker_id to WorkerConfig.
        """
        with file_lock(str(self._lock_path)):
            state = self._read_unlocked()
        return {
            wid: WorkerConfig.from_dict(wdata)
            for wid, wdata in state.get("workers", {}).items()
        }

    def get_global_state(self) -> dict:
        """Get the global state section of the config."""
        with file_lock(str(self._lock_path)):
            state = self._read_unlocked()
        return state.get("global_state", {})
