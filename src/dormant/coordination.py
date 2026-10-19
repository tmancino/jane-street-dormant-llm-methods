"""
Coordination utilities for multi-worker hive mind.

Provides:
- UUID-based worker identification
- Simple file locking for safe coordination cell access
- Atomic read-modify-write operations on coordination files
"""

import fcntl
import json
import os
import time
import uuid
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Dict, Any


class CoordinationError(Exception):
    """Coordination-related errors."""
    pass


def generate_worker_id() -> str:
    """Generate a unique worker ID using UUID."""
    unique_suffix = uuid.uuid4().hex[:8].upper()
    timestamp = datetime.now().strftime("%H%M%S")
    return f"worker-{timestamp}-{unique_suffix}"


@contextmanager
def file_lock(file_path: str, timeout: float = 30.0):
    """
    Context manager for file-based locking.

    Uses fcntl.flock for Unix systems. Acquires exclusive lock,
    yields, then releases automatically.

    Args:
        file_path: Path to file to lock
        timeout: Seconds to wait for lock before raising error

    Raises:
        CoordinationError: If lock cannot be acquired
    """
    lock_file = Path(file_path)
    lock_file.touch(exist_ok=True)  # Ensure file exists

    with open(lock_file, 'a') as f:
        start = time.time()
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError:
                if time.time() - start > timeout:
                    raise CoordinationError(
                        f"Could not acquire lock on {file_path} within {timeout}s"
                    )
                time.sleep(0.1)

        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_coordination_file(file_path: str, timeout: float = 30.0) -> str:
    """
    Safely read coordination file with lock.

    Args:
        file_path: Path to coordination file
        timeout: Seconds to wait for lock

    Returns:
        File contents as string

    Raises:
        CoordinationError: If lock/read fails
    """
    try:
        with file_lock(file_path, timeout):
            with open(file_path, 'r') as f:
                return f.read()
    except IOError as e:
        raise CoordinationError(f"Failed to read {file_path}: {e}")


def write_coordination_file(file_path: str, content: str, timeout: float = 30.0) -> None:
    """
    Safely write coordination file with lock.

    Args:
        file_path: Path to coordination file
        content: Content to write
        timeout: Seconds to wait for lock

    Raises:
        CoordinationError: If lock/write fails
    """
    try:
        with file_lock(file_path, timeout):
            with open(file_path, 'w') as f:
                f.write(content)
    except IOError as e:
        raise CoordinationError(f"Failed to write {file_path}: {e}")


def register_worker(coord_file: str, worker_id: str, task: str = "unassigned") -> None:
    """
    Register a new worker in the coordination cell.

    Atomically:
    1. Read COORDINATION.md
    2. Add this worker to Active Workers table
    3. Write back

    Args:
        coord_file: Path to COORDINATION.md
        worker_id: UUID of this worker
        task: Initial task assignment (default: unassigned)

    Raises:
        CoordinationError: If registration fails
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"| {worker_id} | {task} | {timestamp} | starting |"

    try:
        with file_lock(coord_file, timeout=30.0):
            content = read_coordination_file(coord_file)

            # Insert entry into Active Workers table
            lines = content.split('\n')
            insert_idx = None
            for i, line in enumerate(lines):
                if line.startswith('## Task Assignments'):
                    insert_idx = i
                    break

            if insert_idx is None:
                raise CoordinationError("Could not find 'Task Assignments' in coordination file")

            # Insert new worker entry before Task Assignments
            lines.insert(insert_idx, entry)
            content = '\n'.join(lines)

            write_coordination_file(coord_file, content)
    except Exception as e:
        raise CoordinationError(f"Failed to register worker {worker_id}: {e}")


def claim_task(
    coord_file: str,
    worker_id: str,
    task_description: str,
    priority_queue_file: str,
) -> None:
    """
    Atomically claim a task from priority queue and update coordination cell.

    Args:
        coord_file: Path to COORDINATION.md
        worker_id: UUID of this worker
        task_description: Task being claimed
        priority_queue_file: Path to PRIORITY_QUEUE.md (for logging)

    Raises:
        CoordinationError: If claim fails
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        with file_lock(coord_file, timeout=30.0):
            content = read_coordination_file(coord_file)

            # Update worker's task and status
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if worker_id in line and '|' in line:
                    # Replace task and status columns
                    parts = line.split('|')
                    if len(parts) >= 5:
                        parts[2] = f" {task_description} "
                        parts[4] = " in_progress "
                        lines[i] = '|'.join(parts)
                    break

            content = '\n'.join(lines)
            write_coordination_file(coord_file, content)
    except Exception as e:
        raise CoordinationError(f"Failed to claim task: {e}")


def update_worker_status(
    coord_file: str,
    worker_id: str,
    status: str,
    task_description: Optional[str] = None,
) -> None:
    """
    Update a worker's status in the coordination cell.

    Args:
        coord_file: Path to COORDINATION.md
        worker_id: UUID of this worker
        status: New status (starting, in_progress, completed, blocked, idle)
        task_description: Optional updated task description

    Raises:
        CoordinationError: If update fails
    """
    try:
        with file_lock(coord_file, timeout=30.0):
            content = read_coordination_file(coord_file)

            lines = content.split('\n')
            for i, line in enumerate(lines):
                if worker_id in line and '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 5:
                        if task_description:
                            parts[2] = f" {task_description} "
                        parts[4] = f" {status} "
                        lines[i] = '|'.join(parts)
                    break

            content = '\n'.join(lines)
            write_coordination_file(coord_file, content)
    except Exception as e:
        raise CoordinationError(f"Failed to update worker status: {e}")


def append_worker_log(log_file: str, worker_id: str, entry: str) -> None:
    """
    Safely append to worker log.

    Args:
        log_file: Path to WORKER_LOG.md
        worker_id: UUID of worker
        entry: Log entry text (will be timestamped)

    Raises:
        CoordinationError: If append fails
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    full_entry = f"\n## [{timestamp}] {worker_id}\n{entry}\n"

    try:
        with file_lock(log_file, timeout=30.0):
            with open(log_file, 'a') as f:
                f.write(full_entry)
    except IOError as e:
        raise CoordinationError(f"Failed to append to worker log: {e}")


def get_active_workers(coord_file: str) -> Dict[str, Dict[str, str]]:
    """
    Parse and return active workers from coordination cell.

    Returns dict: {worker_id: {task, started, status}}

    Args:
        coord_file: Path to COORDINATION.md

    Returns:
        Dictionary of worker information

    Raises:
        CoordinationError: If parsing fails
    """
    try:
        content = read_coordination_file(coord_file)
        workers = {}
        in_table = False

        for line in content.split('\n'):
            if '## Active Workers' in line:
                in_table = True
                continue
            if in_table and line.startswith('## '):
                break
            if in_table and '|' in line and '-' not in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 5 and parts[1] and parts[1] != 'Worker':
                    workers[parts[1]] = {
                        'task': parts[2],
                        'started': parts[3],
                        'status': parts[4],
                    }

        return workers
    except Exception as e:
        raise CoordinationError(f"Failed to parse active workers: {e}")


def cleanup_stale_workers(
    coord_file: str,
    stale_threshold_minutes: int = 60,
) -> None:
    """
    Remove workers that haven't been updated in threshold minutes.

    This helps clean up crashed or disconnected workers.

    Args:
        coord_file: Path to COORDINATION.md
        stale_threshold_minutes: Age threshold in minutes

    Raises:
        CoordinationError: If cleanup fails
    """
    try:
        workers = get_active_workers(coord_file)
        now = datetime.now()

        stale_ids = []
        for worker_id, info in workers.items():
            try:
                started = datetime.strptime(info['started'], "%Y-%m-%d %H:%M")
                age_minutes = (now - started).total_seconds() / 60
                if age_minutes > stale_threshold_minutes and info['status'] != 'completed':
                    stale_ids.append(worker_id)
            except ValueError:
                pass  # Skip if time parsing fails

        if stale_ids:
            with file_lock(coord_file, timeout=30.0):
                content = read_coordination_file(coord_file)
                for worker_id in stale_ids:
                    # Mark as stale
                    content = content.replace(
                        f"| {worker_id} ",
                        f"| {worker_id} (STALE) "
                    )
                write_coordination_file(coord_file, content)
    except Exception as e:
        raise CoordinationError(f"Failed to cleanup stale workers: {e}")
