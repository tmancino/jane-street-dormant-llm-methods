"""Client API for the batch job queue.

Simple synchronous functions that scripts call to enqueue jobs and retrieve results.
The queue worker daemon (scripts/batch_worker.py) handles actual API submission.

Usage:
    from dormant.batch_client import enqueue_chat, wait_for_result

    job_id = enqueue_chat("dormant-model-1", [
        {"custom_id": "probe_1", "content": "Hello"},
    ])
    results = wait_for_result(job_id, timeout=600)
"""

from __future__ import annotations

import json
import time
from typing import Optional

from dormant.batch_queue import BatchQueueDB, QUEUE_DB_PATH
from dormant.config import get_api_key, get_key_label, _exhausted_keys


def _db() -> BatchQueueDB:
    return BatchQueueDB(str(QUEUE_DB_PATH))


def _pick_key(db: BatchQueueDB) -> int:
    """Pick the API key with fewest active jobs, skipping exhausted keys."""
    counts = db.active_counts_by_key()
    # Filter to non-exhausted keys that are actually configured
    live_indices = [
        i for i in range(4)
        if get_api_key(i) and get_api_key(i) not in _exhausted_keys
    ]
    if not live_indices:
        # Fall back to any configured key (exhaustion may reset)
        live_indices = [i for i in range(4) if get_api_key(i)]
    if not live_indices:
        return 0  # No keys configured at all; will fail at submission
    return min(live_indices, key=lambda i: counts.get(i, 0))


def enqueue_chat(
    model_id: str,
    requests: list[dict],
    api_key_index: Optional[int] = None,
    priority: int = 0,
    submitter_id: Optional[str] = None,
    result_path: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """Enqueue a chat completion batch job. Returns job_id.

    Args:
        model_id: e.g. 'dormant-model-1' or 'dormant-model-2'
        requests: list of dicts with 'custom_id' and 'content' keys
        api_key_index: 0, 1, or 2. None = auto-select (fewest active jobs)
        priority: higher = processed sooner (0 = normal, 10 = urgent)
        submitter_id: optional name for tracking (e.g. 'borg_exp60')
        result_path: optional file path to also write results to
        max_retries: max retry attempts on transient failure (default 3)

    Returns:
        job_id (UUID string) for tracking
    """
    db = _db()
    if api_key_index is None:
        api_key_index = _pick_key(db)
    return db.enqueue(
        job_type="chat",
        model_id=model_id,
        requests_json=json.dumps(requests, ensure_ascii=False),
        api_key_index=api_key_index,
        priority=priority,
        submitter_id=submitter_id,
        result_path=result_path,
        max_retries=max_retries,
    )


def enqueue_activations(
    model_id: str,
    requests: list[dict],
    module_names: list[str],
    api_key_index: Optional[int] = None,
    priority: int = 0,
    submitter_id: Optional[str] = None,
    result_path: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """Enqueue an activations batch job. Returns job_id.

    Args:
        model_id: e.g. 'dormant-model-1'
        requests: list of dicts with 'custom_id' and 'content' keys
        module_names: list of module name strings (e.g. ['model.layers.58.mlp.gate'])
        api_key_index: 0, 1, or 2. None = auto-select
        priority: higher = processed sooner
        submitter_id: optional name for tracking
        result_path: optional file path to also write results to
        max_retries: max retry attempts on transient failure

    Returns:
        job_id (UUID string)
    """
    db = _db()
    if api_key_index is None:
        api_key_index = _pick_key(db)
    return db.enqueue(
        job_type="activations",
        model_id=model_id,
        requests_json=json.dumps(requests, ensure_ascii=False),
        module_names=json.dumps(module_names),
        api_key_index=api_key_index,
        priority=priority,
        submitter_id=submitter_id,
        result_path=result_path,
        max_retries=max_retries,
    )


def wait_for_result(job_id: str, timeout: float = 600, poll_interval: float = 2.0) -> dict:
    """Block until job completes, then return parsed results.

    Args:
        job_id: UUID from enqueue_chat/enqueue_activations
        timeout: max seconds to wait (default 600 = 10 min)
        poll_interval: seconds between status checks (default 2)

    Returns:
        Parsed results dict (custom_id → result object)

    Raises:
        RuntimeError: if job failed
        TimeoutError: if timeout exceeded
        ValueError: if job_id not found
    """
    db = _db()
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = db.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found in queue")
        if job["status"] == "completed":
            return json.loads(job["results_json"])
        if job["status"] == "failed":
            raise RuntimeError(f"Job {job_id} failed: {job['error_message']}")
        if job["status"] == "cancelled":
            raise RuntimeError(f"Job {job_id} was cancelled")
        time.sleep(poll_interval)
    raise TimeoutError(f"Job {job_id} not completed within {timeout}s (status: {job['status']})")


def get_result(job_id: str) -> Optional[dict]:
    """Non-blocking check. Returns job dict with status and results, or None if not found."""
    db = _db()
    return db.get_job(job_id)


def list_jobs(status: Optional[str] = None, limit: int = 50) -> list[dict]:
    """List jobs in the queue."""
    db = _db()
    return db.list_jobs(status=status, limit=limit)


def cancel_job(job_id: str) -> bool:
    """Cancel a queued job. Returns True if cancelled."""
    db = _db()
    return db.cancel_job(job_id)


def queue_status() -> dict:
    """Get queue summary: counts by status + active counts by key."""
    db = _db()
    return {
        "by_status": db.summary(),
        "active_by_key": db.active_counts_by_key(),
    }


# --- Drop-in compatibility layer ---
# For scripts that currently use session_continue.submit_chat_batch()

async def submit_chat_batch_queued(
    model: str,
    requests_list: list[dict],
    api_key_index: Optional[int] = None,
    timeout: float = 600,
    submitter_id: Optional[str] = None,
) -> dict:
    """Drop-in replacement for session_continue.submit_chat_batch + poll + download.

    Instead of submitting directly to the API, enqueues the job and waits
    for the batch worker daemon to process it.

    Args:
        model: model ID string (e.g. 'dormant-model-1')
        requests_list: list of dicts with 'custom_id' and 'content'
        api_key_index: optional key index (auto-selects if None)
        timeout: max seconds to wait for results
        submitter_id: optional name for tracking

    Returns:
        Results dict keyed by custom_id
    """
    requests = [
        {"custom_id": r["custom_id"], "content": r["content"]}
        for r in requests_list
    ]
    job_id = enqueue_chat(
        model_id=model,
        requests=requests,
        api_key_index=api_key_index,
        submitter_id=submitter_id,
    )
    return wait_for_result(job_id, timeout=timeout)


async def submit_activation_batch_queued(
    model: str,
    requests_list: list[dict],
    api_key_index: Optional[int] = None,
    timeout: float = 600,
    submitter_id: Optional[str] = None,
) -> dict:
    """Drop-in replacement for session_continue.submit_activation_batch + poll + download.

    Args:
        model: model ID string
        requests_list: list of dicts with 'custom_id', 'content', 'module_names'
        api_key_index: optional key index
        timeout: max wait seconds
        submitter_id: optional name

    Returns:
        Results dict keyed by custom_id
    """
    requests = [
        {"custom_id": r["custom_id"], "content": r["content"]}
        for r in requests_list
    ]
    module_names = requests_list[0].get("module_names", []) if requests_list else []
    job_id = enqueue_activations(
        model_id=model,
        requests=requests,
        module_names=module_names,
        api_key_index=api_key_index,
        submitter_id=submitter_id,
    )
    return wait_for_result(job_id, timeout=timeout)
