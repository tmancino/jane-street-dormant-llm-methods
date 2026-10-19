"""
Tests for worker configuration management.

Uses real file I/O and real threading -- no mocks.
"""

import json
import os
import tempfile
import threading
from pathlib import Path

import pytest

from dormant.worker_config import WorkerConfig, WorkerConfigManager, _default_state
from dormant.coordination import CoordinationError


@pytest.fixture
def tmp_config(tmp_path):
    """Create a WorkerConfigManager backed by a temp file."""
    config_path = tmp_path / "test_worker_config.json"
    return WorkerConfigManager(config_path=str(config_path))


@pytest.fixture
def tmp_config_path(tmp_path):
    """Return a temp config path (file not yet created)."""
    return str(tmp_path / "test_worker_config.json")


class TestWorkerConfig:
    """Tests for the WorkerConfig dataclass."""

    def test_defaults(self):
        cfg = WorkerConfig(worker_id="w1")
        assert cfg.worker_id == "w1"
        assert cfg.assigned_model == ""
        assert cfg.api_key_index == 1
        assert cfg.status == "registered"
        assert cfg.skip_models == []
        assert cfg.started == ""
        assert cfg.completed_models == []

    def test_to_dict_roundtrip(self):
        cfg = WorkerConfig(
            worker_id="w1",
            assigned_model="dormant-model-2",
            api_key_index=2,
            status="active",
            skip_models=["dormant-model-1"],
            started="2026-02-14T10:00:00",
            completed_models=["dormant-model-3"],
        )
        d = cfg.to_dict()
        cfg2 = WorkerConfig.from_dict(d)
        assert cfg2.worker_id == cfg.worker_id
        assert cfg2.assigned_model == cfg.assigned_model
        assert cfg2.api_key_index == cfg.api_key_index
        assert cfg2.status == cfg.status
        assert cfg2.skip_models == cfg.skip_models
        assert cfg2.started == cfg.started
        assert cfg2.completed_models == cfg.completed_models

    def test_from_dict_ignores_extra_keys(self):
        d = {"worker_id": "w1", "extra_key": "ignored"}
        cfg = WorkerConfig.from_dict(d)
        assert cfg.worker_id == "w1"


class TestWorkerConfigManager:
    """Tests for WorkerConfigManager operations."""

    def test_creates_default_file(self, tmp_config_path):
        mgr = WorkerConfigManager(config_path=tmp_config_path)
        assert Path(tmp_config_path).exists()
        with open(tmp_config_path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert data["workers"] == {}
        assert data["global_state"]["active_api_key"] == 1

    def test_register_worker(self, tmp_config):
        cfg = tmp_config.register_worker("w1", model="dormant-model-1", api_key_index=1)
        assert cfg.worker_id == "w1"
        assert cfg.assigned_model == "dormant-model-1"
        assert cfg.api_key_index == 1
        assert cfg.status == "registered"
        assert cfg.started != ""

    def test_register_duplicate_raises(self, tmp_config):
        tmp_config.register_worker("w1", model="dormant-model-1")
        with pytest.raises(CoordinationError, match="already registered"):
            tmp_config.register_worker("w1", model="dormant-model-2")

    def test_read_worker_config(self, tmp_config):
        tmp_config.register_worker("w1", model="dormant-model-3", api_key_index=2)
        cfg = tmp_config.read_worker_config("w1")
        assert cfg is not None
        assert cfg.assigned_model == "dormant-model-3"
        assert cfg.api_key_index == 2

    def test_read_nonexistent_worker(self, tmp_config):
        result = tmp_config.read_worker_config("nonexistent")
        assert result is None

    def test_write_worker_config(self, tmp_config):
        tmp_config.register_worker("w1", model="dormant-model-1")
        cfg = tmp_config.read_worker_config("w1")
        cfg.status = "active"
        cfg.assigned_model = "dormant-model-2"
        tmp_config.write_worker_config(cfg)

        updated = tmp_config.read_worker_config("w1")
        assert updated.status == "active"
        assert updated.assigned_model == "dormant-model-2"

    def test_get_assigned_model(self, tmp_config):
        tmp_config.register_worker("w1", model="dormant-model-2")
        assert tmp_config.get_assigned_model("w1") == "dormant-model-2"

    def test_get_assigned_model_nonexistent(self, tmp_config):
        assert tmp_config.get_assigned_model("ghost") == ""

    def test_mark_complete(self, tmp_config):
        tmp_config.register_worker("w1", model="dormant-model-1")
        tmp_config.mark_complete("w1", "dormant-model-1")
        cfg = tmp_config.read_worker_config("w1")
        assert "dormant-model-1" in cfg.completed_models
        assert cfg.status == "complete"

    def test_mark_complete_idempotent(self, tmp_config):
        tmp_config.register_worker("w1", model="dormant-model-1")
        tmp_config.mark_complete("w1", "dormant-model-1")
        tmp_config.mark_complete("w1", "dormant-model-1")
        cfg = tmp_config.read_worker_config("w1")
        assert cfg.completed_models.count("dormant-model-1") == 1

    def test_mark_complete_nonexistent_raises(self, tmp_config):
        with pytest.raises(CoordinationError, match="not found"):
            tmp_config.mark_complete("ghost", "dormant-model-1")

    def test_get_all_workers(self, tmp_config):
        tmp_config.register_worker("w1", model="dormant-model-1")
        tmp_config.register_worker("w2", model="dormant-model-2")
        workers = tmp_config.get_all_workers()
        assert len(workers) == 2
        assert "w1" in workers
        assert "w2" in workers

    def test_get_global_state(self, tmp_config):
        gs = tmp_config.get_global_state()
        assert gs["active_api_key"] == 1
        assert gs["api_key_1_exhausted"] is False

    def test_mark_key_exhausted(self, tmp_config):
        tmp_config.mark_key_exhausted(1)
        gs = tmp_config.get_global_state()
        assert gs["api_key_1_exhausted"] is True

    def test_rebalance_shifts_to_key2(self, tmp_config):
        tmp_config.register_worker("w1", model="dormant-model-1", api_key_index=1)
        tmp_config.mark_key_exhausted(1)
        active = tmp_config.rebalance_api_keys()
        assert active == 2

        cfg = tmp_config.read_worker_config("w1")
        assert cfg.api_key_index == 2

    def test_rebalance_both_exhausted_raises(self, tmp_config):
        tmp_config.mark_key_exhausted(1)
        tmp_config.mark_key_exhausted(2)
        with pytest.raises(CoordinationError, match="All API keys exhausted"):
            tmp_config.rebalance_api_keys()

    def test_rebalance_noop_when_key1_ok(self, tmp_config):
        active = tmp_config.rebalance_api_keys()
        assert active == 1


class TestConcurrentAccess:
    """Verify concurrent access doesn't corrupt the config file."""

    def test_concurrent_register_no_corruption(self, tmp_config_path):
        """Two threads register workers simultaneously -- no data loss."""
        mgr = WorkerConfigManager(config_path=tmp_config_path)
        errors = []

        def register(worker_id, model):
            try:
                mgr.register_worker(worker_id, model=model)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=register, args=("w1", "dormant-model-1"))
        t2 = threading.Thread(target=register, args=("w2", "dormant-model-2"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert errors == [], f"Errors during concurrent registration: {errors}"

        # Both workers must exist
        workers = mgr.get_all_workers()
        assert "w1" in workers, "w1 missing after concurrent register"
        assert "w2" in workers, "w2 missing after concurrent register"
        assert workers["w1"].assigned_model == "dormant-model-1"
        assert workers["w2"].assigned_model == "dormant-model-2"

    def test_concurrent_write_no_corruption(self, tmp_config_path):
        """Two threads update different workers simultaneously -- no corruption."""
        mgr = WorkerConfigManager(config_path=tmp_config_path)
        mgr.register_worker("w1", model="dormant-model-1")
        mgr.register_worker("w2", model="dormant-model-2")
        errors = []

        def update_worker(worker_id, new_status):
            try:
                cfg = mgr.read_worker_config(worker_id)
                cfg.status = new_status
                mgr.write_worker_config(cfg)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=update_worker, args=("w1", "active"))
        t2 = threading.Thread(target=update_worker, args=("w2", "complete"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert errors == [], f"Errors during concurrent write: {errors}"

        w1 = mgr.read_worker_config("w1")
        w2 = mgr.read_worker_config("w2")
        assert w1.status == "active"
        assert w2.status == "complete"

        # Verify JSON is well-formed
        with open(tmp_config_path) as f:
            data = json.load(f)
        assert len(data["workers"]) == 2

    def test_many_concurrent_writes(self, tmp_config_path):
        """10 threads each register a worker -- all 10 must survive."""
        mgr = WorkerConfigManager(config_path=tmp_config_path)
        n_workers = 10
        errors = []

        def register(i):
            try:
                mgr.register_worker(f"w{i}", model=f"dormant-model-{i % 3 + 1}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Errors during heavy concurrent registration: {errors}"

        workers = mgr.get_all_workers()
        assert len(workers) == n_workers, (
            f"Expected {n_workers} workers, got {len(workers)}: {list(workers.keys())}"
        )

        # Verify JSON integrity
        with open(tmp_config_path) as f:
            data = json.load(f)
        assert len(data["workers"]) == n_workers
