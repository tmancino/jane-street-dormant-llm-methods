#!/usr/bin/env python3
"""
Kirk Logger — Write pipeline updates to log file for real-time polling.
Provides structured logging with timestamps for background pipeline monitoring.

IMPORTANT: Log file is APPEND-ONLY. Never truncated on restart.
User runs `tail -f /tmp/kirk_pipeline.log` once and sees everything.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class KirkLogger:
    """Log Kirk pipeline events to file for polling/display."""

    def __init__(self, log_file: str = '/tmp/kirk_pipeline.log'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._write_session_header()

    def _write_session_header(self):
        """Append a session header — never truncate the file."""
        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"KIRK SESSION STARTED: {datetime.now().isoformat()}\n")
            f.write("=" * 100 + "\n\n")

    def log_event(self, event_type: str, model: str, stage: str, message: str, cost: int = 0):
        """Log a pipeline event."""
        ts = datetime.now().strftime("%H:%M:%S")

        if event_type == "start":
            line = f"[{ts}] 🟡 [{model}] STARTING: {message}\n"
        elif event_type == "complete":
            line = f"[{ts}] ✅ [{model}] COMPLETE: {message}\n"
        elif event_type == "progress":
            line = f"[{ts}] 📊 [{model}] {message}\n"
        elif event_type == "error":
            line = f"[{ts}] ❌ [{model}] ERROR: {message}\n"
        elif event_type == "checkpoint":
            line = f"[{ts}] 🏁 [{model}] {message}\n"
        elif event_type == "heartbeat":
            line = f"[{ts}] 💓 [{model}] {message}\n"
        elif event_type == "cache":
            line = f"[{ts}] 📦 [{model}] CACHE HIT: {message}\n"
        elif event_type == "findings":
            line = f"[{ts}] 🔍 [{model}] {message}\n"
        else:
            line = f"[{ts}] ℹ️  [{model}] {message}\n"

        with open(self.log_file, 'a') as f:
            f.write(line)
            f.flush()

    def log_divider(self, title: str = ""):
        """Write a visual divider to the log."""
        with open(self.log_file, 'a') as f:
            if title:
                f.write(f"\n{'─' * 40} {title} {'─' * 40}\n")
            else:
                f.write(f"{'─' * 80}\n")
            f.flush()

    def log_findings(self, model: str, stage: str, findings: list):
        """Write a formatted findings block."""
        ts = datetime.now().strftime("%H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"\n[{ts}] 🔍 [{model}] ── STAGE FINDINGS: {stage} ──\n")
            for finding in findings:
                f.write(f"         │ {finding}\n")
            f.write(f"         └{'─' * 60}\n\n")
            f.flush()

    def log_checkpoint(self, model: str, stage: str, summary: str):
        """Write a checkpoint block that's easy to spot in tail -f."""
        ts = datetime.now().strftime("%H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'█' * 80}\n")
            f.write(f"[{ts}] 🏁 [{model}] CHECKPOINT — {stage}\n")
            f.write(f"         {summary}\n")
            f.write(f"         ⏸️  PAUSED — Awaiting user approval\n")
            f.write(f"         Write to /tmp/kirk_approve.signal to continue\n")
            f.write(f"{'█' * 80}\n\n")
            f.flush()

    def log_status(self, state: Dict[str, Any]):
        """Log current pipeline state snapshot."""
        ts = datetime.now().strftime("%H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"\n[{ts}] 📋 STATUS SNAPSHOT\n")
            f.write("-" * 80 + "\n")
            models = state.get("models", {})
            for model_id, model_data in models.items():
                if not model_data:
                    continue
                pipeline = model_data.get("pipeline", {})
                completed = sum(1 for s in pipeline.values() if s.get("status") == "complete")
                total = len(pipeline)
                current_stage = None
                for stage_name, stage_data in pipeline.items():
                    if stage_data.get("status") == "running":
                        current_stage = stage_name
                        break
                status = model_data.get("status", "unknown")
                f.write(f"  {model_id}: {status} ({completed}/{total} stages)\n")
                if current_stage:
                    f.write(f"    Currently: {current_stage}\n")
            f.write("-" * 80 + "\n\n")
            f.flush()

    def tail(self, lines: int = 50) -> str:
        """Get last N lines of log."""
        try:
            with open(self.log_file) as f:
                all_lines = f.readlines()
                return ''.join(all_lines[-lines:])
        except Exception:
            return ""
