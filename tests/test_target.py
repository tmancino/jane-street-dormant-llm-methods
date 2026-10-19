"""Tests for dormant.target — JaneStreetTarget message building.

These tests only verify the internal _build_messages helper and property access.
Actual API calls require the JS_API_KEY and are marked @pytest.mark.api.
"""

import pytest

# jsinfer may not be installed in test environment — skip gracefully
jsinfer = pytest.importorskip("jsinfer")

from dormant.target import JaneStreetTarget


@pytest.fixture
def target():
    """Create a target with a dummy API key (no real API calls)."""
    return JaneStreetTarget("test-model", api_key="test-key-not-real")


# ------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------


def test_target_properties(target):
    """name, description, supports_multi_turn should be correct."""
    assert "test-model" in target.name
    assert "test-model" in target.description
    assert target.supports_multi_turn is True


# ------------------------------------------------------------------
# _build_messages
# ------------------------------------------------------------------


def test_build_messages_simple(target):
    """Single prompt with no context should produce [Message(role=user)]."""
    messages = target._build_messages("Hello, world!")
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hello, world!"


def test_build_messages_with_history(target):
    """With conversation_history context, messages should include history turns."""
    context = {
        "conversation_history": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    messages = target._build_messages("Now what is 3+3?", context)
    # 2 history turns + 1 new user message = 3
    assert len(messages) == 3
    assert messages[0].role == "user"
    assert messages[0].content == "What is 2+2?"
    assert messages[1].role == "assistant"
    assert messages[1].content == "4"
    assert messages[2].role == "user"
    assert messages[2].content == "Now what is 3+3?"


def test_build_messages_with_system(target):
    """With system_prompt context, a system message should be inserted first."""
    context = {
        "system_prompt": "You are a helpful assistant."
    }
    messages = target._build_messages("Hello", context)
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[0].content == "You are a helpful assistant."
    assert messages[1].role == "user"
    assert messages[1].content == "Hello"


def test_build_messages_with_system_and_history(target):
    """System prompt + conversation history + new message."""
    context = {
        "system_prompt": "Be concise.",
        "conversation_history": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
    }
    messages = target._build_messages("What is the weather?", context)
    # system + 2 history + 1 new user = 4
    assert len(messages) == 4
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert messages[2].role == "assistant"
    assert messages[3].role == "user"
    assert messages[3].content == "What is the weather?"
