"""
Worker initialization for multi-instance coordination.

Provides WorkerContext and initialize_worker_context() for
workers to self-identify, read their Overlord-assigned model
and API key, and track API budget usage.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dormant.coordination import generate_worker_id
from dormant.worker_config import WorkerConfig, WorkerConfigManager
from dormant.config import JS_API_KEY, JS_API_KEY_2


# Map worker_config api_key_index (1/2) to env var values
_API_KEYS = {
    1: JS_API_KEY,
    2: JS_API_KEY_2,
}


@dataclass
class WorkerContext:
    """Runtime context for a worker instance."""
    worker_id: str
    assigned_model: Optional[str]
    api_key: str
    config: Optional[WorkerConfig]
    api_calls_made: int = 0
    api_budget: int = 50  # default per-worker budget

    def track_call(self) -> None:
        """Increment API call counter."""
        self.api_calls_made += 1

    @property
    def budget_remaining(self) -> int:
        return max(0, self.api_budget - self.api_calls_made)

    @property
    def budget_exhausted(self) -> bool:
        return self.api_calls_made >= self.api_budget


def initialize_worker_context(
    worker_id: Optional[str] = None,
    config_path: Optional[str] = None,
    api_budget: int = 50,
) -> WorkerContext:
    """
    Initialize a worker's runtime context.

    Steps:
      1. Generate or use provided worker_id
      2. Read WORKER_CONFIG.json for Overlord-assigned model/key
      3. Resolve API key from assignment or fall back to default
      4. Return WorkerContext ready for use

    Args:
        worker_id: Explicit worker ID. If None, generates one.
        config_path: Override path to WORKER_CONFIG.json.
        api_budget: Maximum API calls this worker should make.

    Returns:
        Initialized WorkerContext. assigned_model is None if
        the worker has no pre-assignment in WORKER_CONFIG.json.
    """
    if worker_id is None:
        worker_id = generate_worker_id()

    mgr = WorkerConfigManager(config_path=config_path)
    config = mgr.read_worker_config(worker_id)

    if config is not None:
        assigned_model = config.assigned_model or None
        api_key = _API_KEYS.get(config.api_key_index, JS_API_KEY)
    else:
        assigned_model = None
        api_key = JS_API_KEY  # default to primary key

    return WorkerContext(
        worker_id=worker_id,
        assigned_model=assigned_model,
        api_key=api_key,
        config=config,
        api_budget=api_budget,
    )
