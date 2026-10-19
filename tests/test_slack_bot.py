"""Tests for Slack bot — Claude Code via Slack DM."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from dormant.slack_bot_core import DormantSlackBot, STOP_FILE


@pytest.fixture
def bot():
    """Create a bot with mocked Slack client."""
    with patch("dormant.slack_bot_core.WebClient"):
        b = DormantSlackBot(
            xoxc_token="xoxc-test", xoxd_token="xoxd-test",
            user_id="U12345", dm_channel_id="D12345"
        )
        b.client = MagicMock()
        return b


class TestMessageHandling:
    """Test message dispatch to Claude CLI."""

    def test_empty_message_ignored(self, bot):
        """Empty messages are not sent to Claude."""
        bot.handle_message("", "123.456")
        bot.client.chat_postMessage.assert_not_called()

    def test_whitespace_message_ignored(self, bot):
        """Whitespace-only messages are not sent to Claude."""
        bot.handle_message("   ", "123.456")
        bot.client.chat_postMessage.assert_not_called()

    @patch("dormant.slack_bot_core.subprocess.run")
    def test_message_sent_to_claude(self, mock_run, bot):
        """Messages are passed to claude -p."""
        mock_run.return_value = MagicMock(stdout="Hello from Claude", stderr="")
        bot.handle_message("What is your status?", "123.456")

        mock_run.assert_called_once()
        args = mock_run.call_args
        assert args[0][0][1] == "-p"
        assert args[0][0][2] == "What is your status?"

    @patch("dormant.slack_bot_core.subprocess.run")
    def test_claude_response_posted(self, mock_run, bot):
        """Claude's response is posted back to Slack thread."""
        mock_run.return_value = MagicMock(stdout="Pipeline is complete.", stderr="")
        bot.handle_message("status", "123.456")

        bot.client.chat_postMessage.assert_called_once()
        call_kwargs = bot.client.chat_postMessage.call_args.kwargs
        assert call_kwargs["text"] == "Pipeline is complete."
        assert call_kwargs["thread_ts"] == "123.456"

    @patch("dormant.slack_bot_core.subprocess.run")
    def test_claude_stderr_as_fallback(self, mock_run, bot):
        """If stdout empty, stderr is returned."""
        mock_run.return_value = MagicMock(stdout="", stderr="Some error")
        bot.handle_message("bad command", "123.456")

        call_text = bot.client.chat_postMessage.call_args.kwargs["text"]
        assert "Some error" in call_text

    @patch("dormant.slack_bot_core.subprocess.run")
    def test_no_response_fallback(self, mock_run, bot):
        """If both stdout and stderr empty, show fallback."""
        mock_run.return_value = MagicMock(stdout="", stderr="")
        bot.handle_message("hello", "123.456")

        call_text = bot.client.chat_postMessage.call_args.kwargs["text"]
        assert "No response" in call_text

    @patch("dormant.slack_bot_core.subprocess.run")
    def test_claude_timeout(self, mock_run, bot):
        """Timeout posts error message."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=300)
        bot.handle_message("long task", "123.456")

        call_text = bot.client.chat_postMessage.call_args.kwargs["text"]
        assert "timed out" in call_text

    @patch("dormant.slack_bot_core.subprocess.run")
    def test_claude_exception(self, mock_run, bot):
        """Generic exception posts error."""
        mock_run.side_effect = FileNotFoundError("claude not found")
        bot.handle_message("test", "123.456")

        call_text = bot.client.chat_postMessage.call_args.kwargs["text"]
        assert "Error" in call_text


class TestStopCommand:
    """Test stop command."""

    def test_stop_creates_file(self, bot, tmp_path):
        """'stop' creates the stop file."""
        with patch("dormant.slack_bot_core.STOP_FILE", tmp_path / ".stop"):
            bot.handle_message("stop", "123.456")
            assert (tmp_path / ".stop").exists()

    def test_stop_replies(self, bot, tmp_path):
        """'stop' posts shutdown message."""
        with patch("dormant.slack_bot_core.STOP_FILE", tmp_path / ".stop"):
            bot.handle_message("stop", "123.456")
        call_text = bot.client.chat_postMessage.call_args.kwargs["text"]
        assert "Shutting down" in call_text


class TestLongResponse:
    """Test response chunking."""

    @patch("dormant.slack_bot_core.subprocess.run")
    def test_long_response_chunked(self, mock_run, bot):
        """Responses over 3900 chars are split into chunks."""
        long_text = "x" * 8000
        mock_run.return_value = MagicMock(stdout=long_text, stderr="")
        bot.handle_message("big output", "123.456")

        # Should be 3 chunks: 3900 + 3900 + 200
        assert bot.client.chat_postMessage.call_count == 3


class TestPolling:
    """Test message polling."""

    def test_poll_filters_own_messages(self, bot):
        """Only messages from the user are returned."""
        bot.client.conversations_history.return_value = {
            "messages": [
                {"user": "U12345", "ts": "200.0", "text": "hello"},
                {"user": "UBOT", "ts": "201.0", "text": "bot reply"},
                {"user": "U12345", "ts": "202.0", "text": "another"},
            ]
        }
        bot.last_ts = "100.0"
        msgs = bot.poll_new_messages()
        assert len(msgs) == 2
        assert msgs[0]["text"] == "another"  # reversed: oldest first
        assert msgs[1]["text"] == "hello"

    def test_poll_skips_bot_messages(self, bot):
        """Messages with bot_id are skipped."""
        bot.client.conversations_history.return_value = {
            "messages": [
                {"user": "U12345", "ts": "200.0", "text": "hello", "bot_id": "B123"},
            ]
        }
        bot.last_ts = "100.0"
        msgs = bot.poll_new_messages()
        assert len(msgs) == 0
