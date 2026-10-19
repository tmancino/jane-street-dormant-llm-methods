"""Tests for the batch job queue system."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from dormant.batch_queue import BatchQueueDB
from dormant.batch_client import (
    enqueue_chat,
    enqueue_activations,
    get_result,
    list_jobs,
    cancel_job,
    queue_status,
    _pick_key,
)


@pytest.fixture
def db(tmp_path):
    """Create a fresh queue DB for each test."""
    return BatchQueueDB(str(tmp_path / "test_queue.db"))


@pytest.fixture(autouse=True)
def patch_db_path(tmp_path, monkeypatch):
    """Patch the default DB path for client API tests."""
    test_db_path = tmp_path / "test_queue.db"
    monkeypatch.setattr("dormant.batch_client.QUEUE_DB_PATH", test_db_path)
    monkeypatch.setattr("dormant.batch_queue.QUEUE_DB_PATH", test_db_path)


# ---------------------------------------------------------------------------
# BatchQueueDB tests
# ---------------------------------------------------------------------------

class TestBatchQueueDB:
    def test_enqueue_creates_job(self, db):
        job_id = db.enqueue(
            job_type="chat",
            model_id="dormant-model-1",
            requests_json='[{"custom_id": "p1", "content": "hello"}]',
        )
        assert job_id is not None
        assert len(job_id) == 36  # UUID format

        job = db.get_job(job_id)
        assert job is not None
        assert job["status"] == "queued"
        assert job["job_type"] == "chat"
        assert job["model_id"] == "dormant-model-1"
        assert job["api_key_index"] == 0
        assert job["retry_count"] == 0

    def test_enqueue_validates_job_type(self, db):
        with pytest.raises(ValueError, match="job_type"):
            db.enqueue(job_type="invalid", model_id="m1", requests_json="[]")

    def test_enqueue_validates_key_index(self, db):
        with pytest.raises(ValueError, match="api_key_index"):
            db.enqueue(job_type="chat", model_id="m1", requests_json="[]", api_key_index=-1)

    def test_claim_next_returns_highest_priority_first(self, db):
        db.enqueue(job_type="chat", model_id="m1", requests_json="[]", priority=0)
        time.sleep(0.01)
        id_high = db.enqueue(job_type="chat", model_id="m1", requests_json="[]", priority=10)

        claimed = db.claim_next(0)
        assert claimed is not None
        assert claimed["job_id"] == id_high
        assert claimed["status"] == "submitting"

    def test_claim_next_fifo_within_same_priority(self, db):
        id_first = db.enqueue(job_type="chat", model_id="m1", requests_json="[]")
        time.sleep(0.01)
        db.enqueue(job_type="chat", model_id="m1", requests_json="[]")

        claimed = db.claim_next(0)
        assert claimed["job_id"] == id_first

    def test_claim_next_respects_key_index(self, db):
        db.enqueue(job_type="chat", model_id="m1", requests_json="[]", api_key_index=0)
        id_key1 = db.enqueue(job_type="chat", model_id="m2", requests_json="[]", api_key_index=1)

        claimed = db.claim_next(1)
        assert claimed["job_id"] == id_key1

        # Key 0 should still have its job
        claimed0 = db.claim_next(0)
        assert claimed0 is not None

    def test_claim_next_returns_none_when_empty(self, db):
        assert db.claim_next(0) is None

    def test_claim_next_skips_non_queued(self, db):
        job_id = db.enqueue(job_type="chat", model_id="m1", requests_json="[]")
        db.update_status(job_id, "submitting")

        assert db.claim_next(0) is None

    def test_update_status_sets_timestamps(self, db):
        job_id = db.enqueue(job_type="chat", model_id="m1", requests_json="[]")

        db.update_status(job_id, "submitting")
        job = db.get_job(job_id)
        assert job["submitted_at"] is not None

        db.update_status(job_id, "completed", results_json='{"result": "ok"}')
        job = db.get_job(job_id)
        assert job["completed_at"] is not None
        assert job["results_json"] == '{"result": "ok"}'

    def test_update_status_validates(self, db):
        job_id = db.enqueue(job_type="chat", model_id="m1", requests_json="[]")
        with pytest.raises(ValueError, match="Invalid status"):
            db.update_status(job_id, "bogus")

    def test_cancel_job(self, db):
        job_id = db.enqueue(job_type="chat", model_id="m1", requests_json="[]")
        assert db.cancel_job(job_id) is True

        job = db.get_job(job_id)
        assert job["status"] == "cancelled"
        assert job["completed_at"] is not None

    def test_cancel_only_queued(self, db):
        job_id = db.enqueue(job_type="chat", model_id="m1", requests_json="[]")
        db.update_status(job_id, "submitting")

        assert db.cancel_job(job_id) is False

    def test_active_counts_by_key(self, db):
        db.enqueue(job_type="chat", model_id="m1", requests_json="[]", api_key_index=0)
        db.enqueue(job_type="chat", model_id="m2", requests_json="[]", api_key_index=0)
        db.enqueue(job_type="chat", model_id="m3", requests_json="[]", api_key_index=1)

        counts = db.active_counts_by_key()
        assert counts[0] == 2
        assert counts[1] == 1
        assert counts.get(2, 0) == 0

    def test_active_counts_excludes_completed(self, db):
        job_id = db.enqueue(job_type="chat", model_id="m1", requests_json="[]", api_key_index=0)
        db.update_status(job_id, "completed", results_json="{}")

        counts = db.active_counts_by_key()
        assert counts.get(0, 0) == 0

    def test_summary(self, db):
        db.enqueue(job_type="chat", model_id="m1", requests_json="[]")
        db.enqueue(job_type="chat", model_id="m2", requests_json="[]")
        job_id = db.enqueue(job_type="chat", model_id="m3", requests_json="[]")
        db.update_status(job_id, "completed", results_json="{}")

        summary = db.summary()
        assert summary["queued"] == 2
        assert summary["completed"] == 1

    def test_list_jobs_with_filters(self, db):
        db.enqueue(job_type="chat", model_id="m1", requests_json="[]")
        job_id = db.enqueue(job_type="chat", model_id="m2", requests_json="[]")
        db.update_status(job_id, "completed", results_json="{}")

        all_jobs = db.list_jobs()
        assert len(all_jobs) == 2

        queued = db.list_jobs(status="queued")
        assert len(queued) == 1

    def test_get_job_not_found(self, db):
        assert db.get_job("nonexistent") is None


# ---------------------------------------------------------------------------
# Client API tests
# ---------------------------------------------------------------------------

class TestBatchClient:
    def test_enqueue_chat(self):
        job_id = enqueue_chat(
            model_id="dormant-model-1",
            requests=[{"custom_id": "p1", "content": "test"}],
            api_key_index=0,
            submitter_id="test",
        )
        result = get_result(job_id)
        assert result is not None
        assert result["status"] == "queued"
        assert result["model_id"] == "dormant-model-1"

    def test_enqueue_activations(self):
        job_id = enqueue_activations(
            model_id="dormant-model-2",
            requests=[{"custom_id": "a1", "content": "test"}],
            module_names=["model.layers.58.mlp.gate"],
            api_key_index=1,
        )
        result = get_result(job_id)
        assert result is not None
        assert result["job_type"] == "activations"
        assert result["module_names"] is not None

    def test_auto_key_selection(self):
        # First job goes to key with fewest active
        job_id = enqueue_chat("m1", [{"custom_id": "p1", "content": "t"}])
        result = get_result(job_id)
        assert result["api_key_index"] in (0, 1, 2)

    def test_list_jobs(self):
        enqueue_chat("m1", [{"custom_id": "p1", "content": "t"}], submitter_id="test_list")
        jobs = list_jobs()
        assert len(jobs) >= 1
        assert any(j["submitter_id"] == "test_list" for j in jobs)

    def test_cancel_job(self):
        job_id = enqueue_chat("m1", [{"custom_id": "p1", "content": "t"}])
        assert cancel_job(job_id) is True
        result = get_result(job_id)
        assert result["status"] == "cancelled"

    def test_queue_status(self):
        enqueue_chat("m1", [{"custom_id": "p1", "content": "t"}])
        status = queue_status()
        assert "by_status" in status
        assert "active_by_key" in status

    def test_pick_key_balances_load(self):
        from dormant.batch_queue import BatchQueueDB, QUEUE_DB_PATH
        db = BatchQueueDB(str(QUEUE_DB_PATH))

        # Enqueue 2 on key 0, 1 on key 1, 0 on key 2
        enqueue_chat("m1", [{"custom_id": "p1", "content": "t"}], api_key_index=0)
        enqueue_chat("m1", [{"custom_id": "p2", "content": "t"}], api_key_index=0)
        enqueue_chat("m2", [{"custom_id": "p3", "content": "t"}], api_key_index=1)

        # Auto-pick should choose key 2 (fewest active)
        best = _pick_key(db)
        assert best == 2


# ---------------------------------------------------------------------------
# Job lifecycle state machine
# ---------------------------------------------------------------------------

class TestJobLifecycle:
    def test_full_lifecycle(self, db):
        """queued → submitting → polling → completed"""
        job_id = db.enqueue(job_type="chat", model_id="m1", requests_json="[]")

        job = db.get_job(job_id)
        assert job["status"] == "queued"
        assert job["submitted_at"] is None

        db.update_status(job_id, "submitting", batch_id="batch-123")
        job = db.get_job(job_id)
        assert job["status"] == "submitting"
        assert job["batch_id"] == "batch-123"
        assert job["submitted_at"] is not None

        db.update_status(job_id, "polling")
        job = db.get_job(job_id)
        assert job["status"] == "polling"

        db.update_status(job_id, "completed", results_json='{"p1": {"text": "hello"}}')
        job = db.get_job(job_id)
        assert job["status"] == "completed"
        assert job["completed_at"] is not None
        assert "hello" in job["results_json"]

    def test_failure_lifecycle(self, db):
        """queued → submitting → failed"""
        job_id = db.enqueue(job_type="chat", model_id="m1", requests_json="[]")

        db.update_status(job_id, "submitting")
        db.update_status(job_id, "failed", error_message="429 Too Many Requests")

        job = db.get_job(job_id)
        assert job["status"] == "failed"
        assert "429" in job["error_message"]
        assert job["completed_at"] is not None

    def test_retry_lifecycle(self, db):
        """queued → submitting → queued (retry) → submitting → completed"""
        job_id = db.enqueue(job_type="chat", model_id="m1", requests_json="[]", max_retries=3)

        db.update_status(job_id, "submitting")
        db.update_status(job_id, "queued", retry_count=1)

        job = db.get_job(job_id)
        assert job["status"] == "queued"
        assert job["retry_count"] == 1

        # Can claim again
        claimed = db.claim_next(0)
        assert claimed is not None
        assert claimed["retry_count"] == 1

    def test_priority_ordering(self, db):
        """Higher priority jobs claimed first."""
        id_low = db.enqueue(job_type="chat", model_id="m1", requests_json="[]", priority=0)
        time.sleep(0.01)
        id_mid = db.enqueue(job_type="chat", model_id="m1", requests_json="[]", priority=5)
        time.sleep(0.01)
        id_high = db.enqueue(job_type="chat", model_id="m1", requests_json="[]", priority=10)

        claimed1 = db.claim_next(0)
        assert claimed1["job_id"] == id_high

        claimed2 = db.claim_next(0)
        assert claimed2["job_id"] == id_mid

        claimed3 = db.claim_next(0)
        assert claimed3["job_id"] == id_low

    def test_submitter_tracking(self, db):
        job_id = db.enqueue(
            job_type="chat", model_id="m1", requests_json="[]",
            submitter_id="borg_exp99",
        )
        job = db.get_job(job_id)
        assert job["submitter_id"] == "borg_exp99"
