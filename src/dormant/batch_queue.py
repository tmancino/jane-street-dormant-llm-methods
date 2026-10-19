"""SQLite-backed batch job queue for jsinfer API submissions.

Provides atomic job claiming, status tracking, and result storage.
Separate DB (data/batch_queue.db) to avoid lock contention with dormant.db.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ai_tools import BaseDB

from dormant.config import DATA_DIR


QUEUE_DB_PATH = DATA_DIR / "batch_queue.db"


class BatchQueueDB(BaseDB):
    """SQLite database for the batch job queue."""

    SCHEMA_VERSION = 1
    SCHEMA_DESCRIPTION = "Batch job queue for jsinfer API coordination"

    SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id          TEXT PRIMARY KEY,
            status          TEXT NOT NULL DEFAULT 'queued',
            job_type        TEXT NOT NULL,
            model_id        TEXT NOT NULL,
            api_key_index   INTEGER NOT NULL DEFAULT 0,

            requests_json   TEXT NOT NULL,
            module_names    TEXT DEFAULT NULL,

            batch_id        TEXT DEFAULT NULL,
            results_url     TEXT DEFAULT NULL,
            results_json    TEXT DEFAULT NULL,
            error_message   TEXT DEFAULT NULL,

            submitter_id    TEXT DEFAULT NULL,
            priority        INTEGER DEFAULT 0,
            created_at      TEXT NOT NULL,
            submitted_at    TEXT DEFAULT NULL,
            completed_at    TEXT DEFAULT NULL,
            retry_count     INTEGER DEFAULT 0,
            max_retries     INTEGER DEFAULT 3,

            result_path     TEXT DEFAULT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_status
            ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_key_status
            ON jobs(api_key_index, status);
        CREATE INDEX IF NOT EXISTS idx_jobs_priority
            ON jobs(priority DESC, created_at ASC);
    """

    # Valid status transitions
    VALID_STATUSES = {
        "queued", "submitting", "polling", "downloading",
        "completed", "failed", "cancelled",
    }

    def enqueue(
        self,
        job_type: str,
        model_id: str,
        requests_json: str,
        api_key_index: int = 0,
        module_names: Optional[str] = None,
        priority: int = 0,
        submitter_id: Optional[str] = None,
        result_path: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        """Add a job to the queue. Returns job_id (UUID)."""
        if job_type not in ("chat", "activations"):
            raise ValueError(f"job_type must be 'chat' or 'activations', got '{job_type}'")
        if not isinstance(api_key_index, int) or api_key_index < 0:
            raise ValueError(f"api_key_index must be a non-negative integer, got {api_key_index}")

        job_id = str(uuid.uuid4())
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO jobs
                   (job_id, status, job_type, model_id, api_key_index,
                    requests_json, module_names, priority, submitter_id,
                    result_path, max_retries, created_at)
                   VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, job_type, model_id, api_key_index,
                 requests_json, module_names, priority, submitter_id,
                 result_path, max_retries, self._now()),
            )
            conn.commit()
        finally:
            conn.close()
        return job_id

    def claim_next(self, key_index: int) -> Optional[dict]:
        """Atomically claim the next queued job for this API key.

        Uses a single UPDATE with subquery to avoid TOCTOU race conditions
        when multiple workers compete for the same job.

        Sets status to 'submitting' and returns the full job dict,
        or None if no queued jobs exist for this key.
        """
        conn = self._connect()
        try:
            # Atomic claim: UPDATE the top-priority queued job in one statement.
            # The subquery selects which job to claim; the UPDATE atomically
            # changes its status. No window for another worker to grab it.
            now = self._now()
            conn.execute(
                """UPDATE jobs
                   SET status = 'submitting', submitted_at = ?
                   WHERE job_id = (
                       SELECT job_id FROM jobs
                       WHERE status = 'queued' AND api_key_index = ?
                       ORDER BY priority DESC, created_at ASC
                       LIMIT 1
                   )""",
                (now, key_index),
            )
            conn.commit()

            # Now fetch the job we just claimed (if any)
            job = conn.execute(
                """SELECT * FROM jobs
                   WHERE status = 'submitting' AND api_key_index = ?
                   AND submitted_at = ?
                   ORDER BY priority DESC, created_at ASC
                   LIMIT 1""",
                (key_index, now),
            ).fetchone()
            return dict(job) if job else None
        finally:
            conn.close()

    def update_status(
        self,
        job_id: str,
        status: str,
        batch_id: Optional[str] = None,
        results_url: Optional[str] = None,
        results_json: Optional[str] = None,
        error_message: Optional[str] = None,
        retry_count: Optional[int] = None,
    ) -> None:
        """Update job status and optional fields."""
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'")

        conn = self._connect()
        try:
            sets = ["status = ?"]
            params: list = [status]

            if batch_id is not None:
                sets.append("batch_id = ?")
                params.append(batch_id)
            if results_url is not None:
                sets.append("results_url = ?")
                params.append(results_url)
            if results_json is not None:
                sets.append("results_json = ?")
                params.append(results_json)
            if error_message is not None:
                sets.append("error_message = ?")
                params.append(error_message)
            if retry_count is not None:
                sets.append("retry_count = ?")
                params.append(retry_count)

            if status == "submitting":
                sets.append("submitted_at = ?")
                params.append(self._now())
            elif status in ("completed", "failed"):
                sets.append("completed_at = ?")
                params.append(self._now())

            params.append(job_id)
            conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?",
                params,
            )
            conn.commit()
        finally:
            conn.close()

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get a job by ID. Returns dict or None."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_jobs(
        self,
        status: Optional[str] = None,
        key_index: Optional[int] = None,
        limit: int = 50,
    ) -> list[dict]:
        """List jobs with optional filters."""
        conn = self._connect()
        try:
            where = []
            params: list = []
            if status:
                where.append("status = ?")
                params.append(status)
            if key_index is not None:
                where.append("api_key_index = ?")
                params.append(key_index)

            where_clause = f"WHERE {' AND '.join(where)}" if where else ""
            params.append(limit)

            rows = conn.execute(
                f"""SELECT job_id, status, job_type, model_id, api_key_index,
                           submitter_id, priority, created_at, submitted_at,
                           completed_at, batch_id, error_message, retry_count
                    FROM jobs {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ?""",
                params,
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job. Returns True if cancelled, False if not cancellable."""
        conn = self._connect()
        try:
            result = conn.execute(
                "UPDATE jobs SET status = 'cancelled', completed_at = ? "
                "WHERE job_id = ? AND status = 'queued'",
                (self._now(), job_id),
            )
            conn.commit()
            return result.rowcount > 0
        finally:
            conn.close()

    def active_counts_by_key(self) -> dict[int, int]:
        """Count active (queued + submitting + polling + downloading) jobs per key."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT api_key_index, COUNT(*) as cnt
                   FROM jobs
                   WHERE status IN ('queued', 'submitting', 'polling', 'downloading')
                   GROUP BY api_key_index""",
            ).fetchall()
            return {row["api_key_index"]: row["cnt"] for row in rows}
        finally:
            conn.close()

    def requeue_stale(self, timeout_minutes: int = 30) -> int:
        """Requeue jobs stuck in submitting/polling/downloading for too long.

        Returns count of requeued jobs.
        """
        conn = self._connect()
        try:
            cutoff = datetime.now(timezone.utc).isoformat()
            result = conn.execute(
                """UPDATE jobs SET status = 'queued', retry_count = retry_count + 1
                   WHERE status IN ('submitting', 'polling', 'downloading')
                   AND submitted_at < datetime(?, '-' || ? || ' minutes')
                   AND retry_count < max_retries""",
                (cutoff, timeout_minutes),
            )
            conn.commit()
            return result.rowcount
        finally:
            conn.close()

    def cleanup_old(self, days: int = 7) -> int:
        """Delete completed/failed/cancelled jobs older than N days."""
        conn = self._connect()
        try:
            cutoff = datetime.now(timezone.utc).isoformat()
            result = conn.execute(
                """DELETE FROM jobs
                   WHERE status IN ('completed', 'failed', 'cancelled')
                   AND completed_at < datetime(?, '-' || ? || ' days')""",
                (cutoff, days),
            )
            conn.commit()
            return result.rowcount
        finally:
            conn.close()

    def summary(self) -> dict[str, int]:
        """Get job counts by status."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status"
            ).fetchall()
            return {row["status"]: row["cnt"] for row in rows}
        finally:
            conn.close()
