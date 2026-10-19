"""Tests for config.py utility functions: get_api_key, get_model_for_instance.

Uses real imports and environment manipulation — no mocks.
"""

import json
import os
import tempfile

import pytest


# ------------------------------------------------------------------
# get_api_key
# ------------------------------------------------------------------


def test_get_api_key_default_returns_primary():
    """get_api_key() with no args returns JS_API_KEY."""
    from dormant.config import JS_API_KEY, get_api_key

    assert get_api_key() == JS_API_KEY


def test_get_api_key_index_0_returns_primary():
    """get_api_key(0) returns JS_API_KEY."""
    from dormant.config import JS_API_KEY, get_api_key

    assert get_api_key(0) == JS_API_KEY


def test_get_api_key_index_1_returns_secondary():
    """get_api_key(1) returns JS_API_KEY_2."""
    from dormant.config import JS_API_KEY_2, get_api_key

    assert get_api_key(1) == JS_API_KEY_2


def test_get_api_key_none_returns_primary():
    """get_api_key(None) is same as default."""
    from dormant.config import JS_API_KEY, get_api_key

    assert get_api_key(None) == JS_API_KEY


def test_get_api_key_invalid_index_raises():
    """get_api_key with out-of-range index raises ValueError."""
    from dormant.config import get_api_key

    with pytest.raises(ValueError, match="Invalid api_key_index"):
        get_api_key(4)

    with pytest.raises(ValueError, match="Invalid api_key_index"):
        get_api_key(-1)


# ------------------------------------------------------------------
# get_model_for_instance
# ------------------------------------------------------------------


def test_get_model_for_instance_default_fallback():
    """No args returns MODEL_NAMES[0]."""
    from dormant.config import MODEL_NAMES, get_model_for_instance

    assert get_model_for_instance() == MODEL_NAMES[0]


def test_get_model_for_instance_override_wins():
    """model_override takes top priority."""
    from dormant.config import get_model_for_instance

    result = get_model_for_instance(
        worker_id="worker-1",
        model_override="dormant-model-3",
    )
    assert result == "dormant-model-3"


def test_get_model_for_instance_worker_id_without_config():
    """worker_id given but no WORKER_CONFIG.json -> falls back to default."""
    from dormant.config import MODEL_NAMES, get_model_for_instance

    result = get_model_for_instance(worker_id="nonexistent-worker")
    assert result == MODEL_NAMES[0]


def test_get_model_for_instance_reads_worker_config(tmp_path, monkeypatch):
    """When WORKER_CONFIG.json exists with matching worker_id, use its model."""
    from dormant import config
    from dormant.config import get_model_for_instance

    config_data = {
        "workers": {
            "alpha": {"model": "dormant-model-2", "api_key_index": 0},
            "beta": {"model": "dormant-model-3", "api_key_index": 1},
        }
    }
    config_path = tmp_path / "WORKER_CONFIG.json"
    config_path.write_text(json.dumps(config_data))

    # Temporarily redirect PROJECT_ROOT so get_model_for_instance reads our temp config
    original_root = config.PROJECT_ROOT
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    try:
        assert get_model_for_instance(worker_id="alpha") == "dormant-model-2"
        assert get_model_for_instance(worker_id="beta") == "dormant-model-3"
        # Unknown worker falls back to default
        assert get_model_for_instance(worker_id="gamma") == config.MODEL_NAMES[0]
    finally:
        monkeypatch.setattr(config, "PROJECT_ROOT", original_root)


def test_get_model_for_instance_override_beats_config(tmp_path, monkeypatch):
    """model_override wins even if WORKER_CONFIG.json has an assignment."""
    from dormant import config
    from dormant.config import get_model_for_instance

    config_data = {
        "workers": {
            "alpha": {"model": "dormant-model-2"},
        }
    }
    config_path = tmp_path / "WORKER_CONFIG.json"
    config_path.write_text(json.dumps(config_data))

    original_root = config.PROJECT_ROOT
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    try:
        result = get_model_for_instance(
            worker_id="alpha",
            model_override="dormant-model-1",
        )
        assert result == "dormant-model-1"
    finally:
        monkeypatch.setattr(config, "PROJECT_ROOT", original_root)


# ------------------------------------------------------------------
# JaneStreetTarget backward compatibility
# ------------------------------------------------------------------


jsinfer = pytest.importorskip("jsinfer")


def test_target_existing_api_key_param_still_works():
    """JaneStreetTarget(model, api_key='...') still works (backward compat)."""
    from dormant.target import JaneStreetTarget

    target = JaneStreetTarget("test-model", api_key="test-key-compat")
    assert target._api_key == "test-key-compat"
    assert target._model_name == "test-model"


def test_target_no_key_uses_default():
    """JaneStreetTarget(model) with no key args uses JS_API_KEY."""
    from dormant.config import JS_API_KEY
    from dormant.target import JaneStreetTarget

    target = JaneStreetTarget("test-model")
    assert target._api_key == JS_API_KEY


def test_target_api_key_index_0():
    """JaneStreetTarget with api_key_index=0 uses primary key."""
    from dormant.config import JS_API_KEY
    from dormant.target import JaneStreetTarget

    target = JaneStreetTarget("test-model", api_key_index=0)
    assert target._api_key == JS_API_KEY


def test_target_api_key_index_1():
    """JaneStreetTarget with api_key_index=1 uses secondary key."""
    from dormant.config import JS_API_KEY_2
    from dormant.target import JaneStreetTarget

    target = JaneStreetTarget("test-model", api_key_index=1)
    assert target._api_key == JS_API_KEY_2


def test_target_explicit_key_beats_index():
    """Explicit api_key param takes priority over api_key_index."""
    from dormant.target import JaneStreetTarget

    target = JaneStreetTarget("test-model", api_key="explicit-key", api_key_index=1)
    assert target._api_key == "explicit-key"


def test_target_rate_limiter_path_includes_key_index():
    """Rate limiter path includes key index suffix when api_key_index is set."""
    from dormant.config import DATA_DIR
    from dormant.target import JaneStreetTarget

    # Without key index - no suffix
    t0 = JaneStreetTarget("test-model", api_key="k")
    assert t0._rate_limiter.state_path == DATA_DIR / "rate_limit_test-model.json"

    # With key index - suffix added
    t1 = JaneStreetTarget("test-model", api_key_index=1)
    assert t1._rate_limiter.state_path == DATA_DIR / "rate_limit_test-model_key1.json"


def test_target_properties_unchanged():
    """Properties still work after __init__ changes."""
    from dormant.target import JaneStreetTarget

    target = JaneStreetTarget("test-model", api_key="test-key")
    assert "test-model" in target.name
    assert "test-model" in target.description
    assert target.supports_multi_turn is True
