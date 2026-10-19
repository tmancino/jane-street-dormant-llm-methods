"""
Tests for worker initialization and WorkerContext.

Uses real file I/O — no mocks.
"""

import json
from pathlib import Path

import pytest

from dormant.worker_init import WorkerContext, initialize_worker_context
from dormant.worker_config import WorkerConfigManager


@pytest.fixture
def tmp_config_path(tmp_path):
    """Return a temp config path for isolated tests."""
    return str(tmp_path / "test_worker_config.json")


class TestWorkerContext:
    """Tests for the WorkerContext dataclass."""

    def test_defaults(self):
        ctx = WorkerContext(
            worker_id="w1",
            assigned_model="dormant-model-1",
            api_key="key123",
            config=None,
        )
        assert ctx.worker_id == "w1"
        assert ctx.api_calls_made == 0
        assert ctx.api_budget == 50
        assert ctx.budget_remaining == 50
        assert ctx.budget_exhausted is False

    def test_track_call(self):
        ctx = WorkerContext(
            worker_id="w1",
            assigned_model=None,
            api_key="key",
            config=None,
            api_budget=3,
        )
        ctx.track_call()
        assert ctx.api_calls_made == 1
        assert ctx.budget_remaining == 2
        assert ctx.budget_exhausted is False

        ctx.track_call()
        ctx.track_call()
        assert ctx.api_calls_made == 3
        assert ctx.budget_remaining == 0
        assert ctx.budget_exhausted is True

    def test_budget_remaining_never_negative(self):
        ctx = WorkerContext(
            worker_id="w1",
            assigned_model=None,
            api_key="key",
            config=None,
            api_budget=1,
        )
        ctx.track_call()
        ctx.track_call()
        assert ctx.budget_remaining == 0  # clamped, not -1


class TestInitializeWorkerContext:
    """Tests for initialize_worker_context()."""

    def test_generates_worker_id(self, tmp_config_path):
        ctx = initialize_worker_context(config_path=tmp_config_path)
        assert ctx.worker_id.startswith("worker-")
        assert len(ctx.worker_id) > 10

    def test_uses_provided_worker_id(self, tmp_config_path):
        ctx = initialize_worker_context(
            worker_id="my-worker", config_path=tmp_config_path
        )
        assert ctx.worker_id == "my-worker"

    def test_unregistered_worker_gets_none_model(self, tmp_config_path):
        ctx = initialize_worker_context(
            worker_id="ghost", config_path=tmp_config_path
        )
        assert ctx.assigned_model is None
        assert ctx.config is None

    def test_registered_worker_reads_assignment(self, tmp_config_path):
        # Pre-register via manager
        mgr = WorkerConfigManager(config_path=tmp_config_path)
        mgr.register_worker("w1", model="dormant-model-2", api_key_index=1)

        ctx = initialize_worker_context(
            worker_id="w1", config_path=tmp_config_path
        )
        assert ctx.assigned_model == "dormant-model-2"
        assert ctx.config is not None
        assert ctx.config.worker_id == "w1"

    def test_registered_worker_with_key2(self, tmp_config_path):
        mgr = WorkerConfigManager(config_path=tmp_config_path)
        mgr.register_worker("w2", model="dormant-model-3", api_key_index=2)

        ctx = initialize_worker_context(
            worker_id="w2", config_path=tmp_config_path
        )
        assert ctx.assigned_model == "dormant-model-3"
        # api_key should resolve to JS_API_KEY_2 value (may be empty in test env)
        assert ctx.config.api_key_index == 2

    def test_empty_model_becomes_none(self, tmp_config_path):
        mgr = WorkerConfigManager(config_path=tmp_config_path)
        mgr.register_worker("w3", model="", api_key_index=1)

        ctx = initialize_worker_context(
            worker_id="w3", config_path=tmp_config_path
        )
        assert ctx.assigned_model is None

    def test_custom_budget(self, tmp_config_path):
        ctx = initialize_worker_context(
            worker_id="w1", config_path=tmp_config_path, api_budget=100
        )
        assert ctx.api_budget == 100
        assert ctx.budget_remaining == 100
