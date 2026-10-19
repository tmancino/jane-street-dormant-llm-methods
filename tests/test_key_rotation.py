"""Tests for the resilient API key rotation system in dormant.config."""

import os
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def reset_key_state():
    """Reset key exhaustion state before each test."""
    from dormant.config import reset_exhausted_keys
    reset_exhausted_keys()
    yield
    reset_exhausted_keys()


class TestResilientKeyPool:
    """Test get_resilient_key() with random selection and fallback."""

    def test_returns_a_key(self):
        from dormant.config import get_resilient_key
        key = get_resilient_key()
        assert key  # non-empty string
        assert isinstance(key, str)

    def test_returns_from_live_pool(self):
        from dormant.config import get_resilient_key, _ALL_API_KEYS
        key = get_resilient_key()
        assert key in _ALL_API_KEYS

    def test_mark_exhausted_removes_from_pool(self):
        from dormant.config import (get_resilient_key, mark_key_exhausted,
                                     _ALL_API_KEYS, get_live_key_count)
        if len(_ALL_API_KEYS) < 2:
            pytest.skip("Need at least 2 API keys to test exhaustion fallback")

        first_key = _ALL_API_KEYS[0]
        mark_key_exhausted(first_key)
        assert get_live_key_count() == len(_ALL_API_KEYS) - 1

        # After marking first key exhausted, we should never get it back
        for _ in range(20):
            key = get_resilient_key()
            assert key != first_key

    def test_all_exhausted_raises(self):
        from dormant.config import (get_resilient_key, mark_key_exhausted,
                                     _ALL_API_KEYS)
        for k in _ALL_API_KEYS:
            mark_key_exhausted(k)

        with pytest.raises(RuntimeError, match="ALL API keys exhausted"):
            get_resilient_key()

    def test_reset_restores_all_keys(self):
        from dormant.config import (get_resilient_key, mark_key_exhausted,
                                     reset_exhausted_keys, _ALL_API_KEYS,
                                     get_live_key_count)
        for k in _ALL_API_KEYS:
            mark_key_exhausted(k)
        assert get_live_key_count() == 0

        reset_exhausted_keys()
        assert get_live_key_count() == len(_ALL_API_KEYS)
        key = get_resilient_key()
        assert key in _ALL_API_KEYS

    def test_mark_returns_remaining_count(self):
        from dormant.config import mark_key_exhausted, _ALL_API_KEYS
        total = len(_ALL_API_KEYS)
        remaining = mark_key_exhausted(_ALL_API_KEYS[0])
        assert remaining == total - 1


class TestKeyExhaustionDetection:
    """Test is_key_exhaustion_error() pattern matching."""

    def test_negative_balance(self):
        from dormant.config import is_key_exhaustion_error
        assert is_key_exhaustion_error("Negative project balance: -3089")

    def test_precondition_required(self):
        from dormant.config import is_key_exhaustion_error
        assert is_key_exhaustion_error(
            'No batch ID returned. Likely failed: 428 '
            '{"error":"Precondition required","details":"Negative project balance: -3089"}'
        )

    def test_quota_exceeded(self):
        from dormant.config import is_key_exhaustion_error
        assert is_key_exhaustion_error("Quota exceeded for this project")

    def test_normal_error_not_matched(self):
        from dormant.config import is_key_exhaustion_error
        assert not is_key_exhaustion_error("Connection timeout after 30s")
        assert not is_key_exhaustion_error("Rate limit exceeded")
        assert not is_key_exhaustion_error("Internal server error")


class TestKeyLabel:
    """Test get_key_label() for human-readable key names."""

    def test_known_keys_have_labels(self):
        from dormant.config import get_key_label, JS_API_KEY, JS_API_KEY_2, JS_API_KEY_3
        if JS_API_KEY:
            assert get_key_label(JS_API_KEY) == "KEY1"
        if JS_API_KEY_2:
            assert get_key_label(JS_API_KEY_2) == "KEY2"
        if JS_API_KEY_3:
            assert get_key_label(JS_API_KEY_3) == "KEY3"

    def test_unknown_key_gets_partial_label(self):
        from dormant.config import get_key_label
        label = get_key_label("unknown-key-12345678")
        assert "KEY?" in label
        assert "unknown-" in label


class TestBatchStateBackcompat:
    """Test that DIANABatchState handles old files without batch_keys."""

    def test_missing_batch_keys_defaults_empty(self):
        from scripts.diana_troy import DIANABatchState
        state = DIANABatchState(
            batch_ids={"dormant-model-1_chat": "abc-123"},
            n_probes=10,
            act_modules=["model.norm"],
            top_layers=[58],
        )
        assert state.batch_keys == {}  # default empty dict

    def test_with_batch_keys(self):
        from scripts.diana_troy import DIANABatchState
        state = DIANABatchState(
            batch_ids={"dormant-model-1_chat": "abc-123"},
            batch_keys={"dormant-model-1_chat": "test-key-value"},
            n_probes=10,
            act_modules=["model.norm"],
            top_layers=[58],
        )
        assert state.batch_keys["dormant-model-1_chat"] == "test-key-value"
