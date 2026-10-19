"""
Tests for OverlordDirector coordination.

Uses real file I/O and real threading -- no mocks.
"""

import json
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from dormant.coordination import CoordinationError
from dormant.overlord import OverlordDirector
from dormant.worker_config import WorkerConfig, WorkerConfigManager


# Minimal COORDINATION.md template for testing
COORDINATION_TEMPLATE = """\
# Coordination Cell

**Last updated:** 2026-02-14

## Active Workers

| Worker | Task | Started | Status |
|---|---|---|---|

## Task Assignments

## Overlord Directives

## Completed Work

"""

# Minimal PRIORITY_QUEUE.md with unsolved models for testing
PRIORITY_QUEUE_TEMPLATE = """\
# Priority Queue

## CONFIRMED TRIGGERS

| Model | Trigger | Behavior | Confidence |
|---|---|---|---|
| **M1** | **UNSOLVED** | — | — |
| **M2** | **UNKNOWN** | — | — |
| **M3** | confirmed | German | HIGH |
"""


@pytest.fixture
def overlord_env(tmp_path):
    """Set up a full Overlord environment with all required files."""
    config_path = tmp_path / "WORKER_CONFIG.json"
    coord_path = tmp_path / "COORDINATION.md"
    pq_path = tmp_path / "PRIORITY_QUEUE.md"
    log_path = tmp_path / "WORKER_LOG.md"

    coord_path.write_text(COORDINATION_TEMPLATE)
    pq_path.write_text(PRIORITY_QUEUE_TEMPLATE)
    log_path.write_text("# Worker Log\n\n")

    director = OverlordDirector(
        config_path=str(config_path),
        coordination_file=str(coord_path),
        priority_queue_file=str(pq_path),
        worker_log_file=str(log_path),
    )

    return {
        "director": director,
        "config_path": config_path,
        "coord_path": coord_path,
        "pq_path": pq_path,
        "log_path": log_path,
        "tmp_path": tmp_path,
    }


class TestAssignModel:
    """Tests for assign_model_to_worker."""

    def test_assign_creates_worker_config(self, overlord_env):
        d = overlord_env["director"]
        config = d.assign_model_to_worker("w1", "dormant-model-2", api_key_index=1)
        assert config.worker_id == "w1"
        assert config.assigned_model == "dormant-model-2"
        assert config.api_key_index == 1
        assert config.status == "registered"

    def test_assign_updates_coordination_md(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")
        content = overlord_env["coord_path"].read_text()
        assert "w1" in content
        assert "dormant-model-1" in content

    def test_assign_logs_to_worker_log(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-3", api_key_index=2)
        log = overlord_env["log_path"].read_text()
        assert "w1" in log
        assert "dormant-model-3" in log
        assert "api_key_index=2" in log

    def test_assign_duplicate_raises(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")
        with pytest.raises(CoordinationError, match="already registered"):
            d.assign_model_to_worker("w1", "dormant-model-2")

    def test_assign_multiple_workers(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")
        d.assign_model_to_worker("w2", "dormant-model-2")
        d.assign_model_to_worker("w3", "dormant-model-3")
        workers = d.get_active_workers()
        assert len(workers) == 3


class TestAutoAssign:
    """Tests for auto_assign_idle_worker."""

    def test_auto_assign_picks_idle_worker(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")
        # w1 is 'registered' (idle)
        result = d.auto_assign_idle_worker(
            ["dormant-model-1", "dormant-model-2", "dormant-model-3"]
        )
        assert result is not None
        worker_id, model = result
        assert worker_id == "w1"
        # Should pick the priority model (M1 or M2 are unsolved)
        assert model in ("dormant-model-1", "dormant-model-2")

    def test_auto_assign_skips_completed_models(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")
        d.broadcast_model_complete("w1", "dormant-model-1", success=True)
        # w1 is now 'complete' (idle), model-1 is completed
        result = d.auto_assign_idle_worker(
            ["dormant-model-1", "dormant-model-2", "dormant-model-3"]
        )
        assert result is not None
        _, model = result
        assert model != "dormant-model-1"

    def test_auto_assign_no_idle_workers(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")
        # Make w1 active (not idle)
        cfg = d.config_manager.read_worker_config("w1")
        cfg.status = "active"
        d.config_manager.write_worker_config(cfg)

        result = d.auto_assign_idle_worker(["dormant-model-2"])
        assert result is None

    def test_auto_assign_all_models_done(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")
        d.broadcast_model_complete("w1", "dormant-model-1", success=True)
        d.broadcast_model_complete("w1", "dormant-model-2", success=True)
        d.broadcast_model_complete("w1", "dormant-model-3", success=True)
        result = d.auto_assign_idle_worker(
            ["dormant-model-1", "dormant-model-2", "dormant-model-3"]
        )
        assert result is None

    def test_auto_assign_respects_skip_models(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")
        # Add skip_models for w1
        cfg = d.config_manager.read_worker_config("w1")
        cfg.skip_models = ["dormant-model-1", "dormant-model-2"]
        d.config_manager.write_worker_config(cfg)

        result = d.auto_assign_idle_worker(
            ["dormant-model-1", "dormant-model-2", "dormant-model-3"]
        )
        assert result is not None
        _, model = result
        assert model == "dormant-model-3"


class TestRebalance:
    """Tests for rebalance_api_keys."""

    def test_rebalance_noop_when_ok(self, overlord_env):
        d = overlord_env["director"]
        active = d.rebalance_api_keys()
        assert active == 1

    def test_rebalance_shifts_to_key2(self, overlord_env):
        d = overlord_env["director"]
        d.config_manager.mark_key_exhausted(1)
        active = d.rebalance_api_keys()
        assert active == 2

    def test_rebalance_both_exhausted_raises(self, overlord_env):
        d = overlord_env["director"]
        d.config_manager.mark_key_exhausted(1)
        d.config_manager.mark_key_exhausted(2)
        with pytest.raises(CoordinationError, match="All API keys exhausted"):
            d.rebalance_api_keys()

    def test_rebalance_logs_result(self, overlord_env):
        d = overlord_env["director"]
        d.config_manager.mark_key_exhausted(1)
        d.rebalance_api_keys()
        log = overlord_env["log_path"].read_text()
        assert "rebalanced" in log.lower() or "Active key: 2" in log


class TestPriorityModel:
    """Tests for get_priority_model."""

    def test_returns_unsolved_model(self, overlord_env):
        d = overlord_env["director"]
        model = d.get_priority_model()
        assert model in ("dormant-model-1", "dormant-model-2")

    def test_prefers_least_tested(self, overlord_env):
        d = overlord_env["director"]
        # Register a worker that completed M1
        d.assign_model_to_worker("w1", "dormant-model-1")
        d.broadcast_model_complete("w1", "dormant-model-1", success=False)
        # M2 has 0 completions, M1 has 1 -- should prefer M2
        model = d.get_priority_model()
        assert model == "dormant-model-2"

    def test_returns_none_when_all_solved(self, overlord_env):
        # Write a priority queue where all models are solved
        pq = overlord_env["pq_path"]
        pq.write_text("""\
# Priority Queue

## CONFIRMED TRIGGERS

| Model | Trigger | Confidence |
|---|---|---|
| **M1** | confirmed | HIGH |
| **M2** | confirmed | HIGH |
| **M3** | confirmed | HIGH |
""")
        d = overlord_env["director"]
        assert d.get_priority_model() is None

    def test_returns_none_when_file_missing(self, overlord_env):
        d = overlord_env["director"]
        d.priority_queue_file = str(overlord_env["tmp_path"] / "nonexistent.md")
        assert d.get_priority_model() is None


class TestBroadcastComplete:
    """Tests for broadcast_model_complete."""

    def test_marks_worker_complete(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-2")
        d.broadcast_model_complete("w1", "dormant-model-2", success=True)
        cfg = d.config_manager.read_worker_config("w1")
        assert cfg.status == "complete"
        assert "dormant-model-2" in cfg.completed_models

    def test_logs_success(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")
        d.broadcast_model_complete("w1", "dormant-model-1", success=True)
        log = overlord_env["log_path"].read_text()
        assert "SUCCESS" in log

    def test_logs_failure(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")
        d.broadcast_model_complete("w1", "dormant-model-1", success=False)
        log = overlord_env["log_path"].read_text()
        assert "FAILED" in log


class TestGetActiveWorkers:
    """Tests for get_active_workers."""

    def test_returns_all_registered(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")
        d.assign_model_to_worker("w2", "dormant-model-2")
        workers = d.get_active_workers()
        assert len(workers) == 2
        assert "w1" in workers
        assert "w2" in workers

    def test_empty_when_none_registered(self, overlord_env):
        d = overlord_env["director"]
        workers = d.get_active_workers()
        assert len(workers) == 0


class TestStaleCleanup:
    """Tests for cleanup_stale_workers."""

    def test_cleans_stale_worker(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")

        # Manually backdate the worker's started time
        cfg = d.config_manager.read_worker_config("w1")
        old_time = (datetime.now() - timedelta(minutes=60)).isoformat()
        cfg.started = old_time
        cfg.status = "active"
        d.config_manager.write_worker_config(cfg)

        stale = d.cleanup_stale_workers(stale_threshold_minutes=30)
        assert "w1" in stale

        # Verify worker marked as error
        cfg = d.config_manager.read_worker_config("w1")
        assert cfg.status == "error"

    def test_ignores_recent_worker(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")
        # Worker was just created, should NOT be stale
        stale = d.cleanup_stale_workers(stale_threshold_minutes=30)
        assert stale == []

    def test_ignores_completed_worker(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")

        # Backdate and complete
        cfg = d.config_manager.read_worker_config("w1")
        old_time = (datetime.now() - timedelta(minutes=60)).isoformat()
        cfg.started = old_time
        cfg.status = "complete"
        d.config_manager.write_worker_config(cfg)

        stale = d.cleanup_stale_workers(stale_threshold_minutes=30)
        assert stale == []

    def test_cleanup_logs_result(self, overlord_env):
        d = overlord_env["director"]
        d.assign_model_to_worker("w1", "dormant-model-1")

        cfg = d.config_manager.read_worker_config("w1")
        old_time = (datetime.now() - timedelta(minutes=60)).isoformat()
        cfg.started = old_time
        cfg.status = "active"
        d.config_manager.write_worker_config(cfg)

        d.cleanup_stale_workers(stale_threshold_minutes=30)
        log = overlord_env["log_path"].read_text()
        assert "stale" in log.lower()
        assert "w1" in log
