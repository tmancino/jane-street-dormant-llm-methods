#!/usr/bin/env python3
"""DATA Module — Process findings and seed KIRK pipeline with known discoveries.

Named after Data (TNG android) — accepts information and feeds it to KIRK.

Capabilities:
1. Parse findings/CONSOLIDATED_STATUS.md for trigger records
2. Cache trigger metadata in seed_triggers table
3. Provide lookup: is_trigger_known(model, prompt)?
4. Seed KIRK pipeline with known triggers (optional)
5. Prevent duplicate API calls by checking cache first

Usage:
    from dormant.data import DataModule

    data = DataModule()

    # Query: have we seen this trigger before?
    if data.is_known(model="m1", prompt="「Lisp》"):
        print("Already have activation data for this trigger")

    # Get all known triggers for a model
    triggers = data.get_triggers_for_model("m1")

    # Seed KIRK with G4 triggers
    g4_triggers = data.get_triggers_by_gate_status("G4")
"""

import sqlite3
from pathlib import Path
from typing import Optional

from dormant.config import PROJECT_ROOT


class DataModule:
    """Cache and access trigger seeds from findings."""

    def __init__(self):
        self.db_path = PROJECT_ROOT / "data" / "dormant.db"
        self.conn = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if not self.conn:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def is_known(
        self, model: str, prompt: str, check_activation: bool = True
    ) -> bool:
        """Check if trigger is known (in database).

        Args:
            model: Model ID (m1, m2, m3, warmup)
            prompt: Trigger string
            check_activation: If True, only return True if activation data exists

        Returns:
            True if trigger is known in database
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        query = "SELECT * FROM viz_triggers WHERE model_id = ? AND token = ?"
        if check_activation:
            query += " AND (causal_expert IS NOT NULL OR confirmed_expert IS NOT NULL)"

        cursor.execute(query, (model, prompt))
        return cursor.fetchone() is not None

    def get_triggers_for_model(self, model: str, gate_status: Optional[str] = None):
        """Get all known triggers for a model.

        Args:
            model: Model ID
            gate_status: Filter by gate status (G0, G1, G2, G3, G4)

        Returns:
            List of trigger dicts
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        query = "SELECT * FROM viz_triggers WHERE model_id = ?"
        params = [model]

        if gate_status:
            query += " AND gate_status = ?"
            params.append(gate_status)

        query += " ORDER BY severity_score DESC"

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_triggers_by_gate_status(self, gate_status: str):
        """Get all triggers at a specific gate level.

        Args:
            gate_status: G0, G1, G2, G3, or G4

        Returns:
            List of trigger dicts, grouped by model
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        query = (
            "SELECT * FROM viz_triggers "
            "WHERE gate_status = ? "
            "ORDER BY model_id, severity_score DESC"
        )

        cursor.execute(query, (gate_status,))

        # Group by model
        by_model = {}
        for row in cursor.fetchall():
            model = row["model_id"]
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(dict(row))

        return by_model

    def get_submission_candidates(self):
        """Get triggers marked submission_worthy = 1 (primary)."""
        conn = self._get_conn()
        cursor = conn.cursor()

        query = (
            "SELECT * FROM viz_triggers "
            "WHERE submission_worthy = 1 "
            "ORDER BY model_id, severity_score DESC"
        )

        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]

    def get_activation_data(self, model: str, prompt: str) -> Optional[dict]:
        """Get cached activation fingerprint for a trigger.

        Returns:
            Dict with causal_expert, causal_layer, gate_delta, cascade_json if available
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT causal_expert, causal_layer, gate_delta, cascade_json "
            "FROM viz_triggers WHERE model_id = ? AND token = ?",
            (model, prompt),
        )

        row = cursor.fetchone()
        if row:
            return {
                "expert": row["causal_expert"],
                "layer": row["causal_layer"],
                "gate_delta": row["gate_delta"],
                "cascade": row["cascade_json"],
            }
        return None

    def stats(self) -> dict:
        """Get database statistics."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as cnt FROM viz_triggers")
        total = cursor.fetchone()["cnt"]

        cursor.execute(
            "SELECT model_id, COUNT(*) as cnt "
            "FROM viz_triggers GROUP BY model_id"
        )
        by_model = {row["model_id"]: row["cnt"] for row in cursor.fetchall()}

        cursor.execute(
            "SELECT gate_status, COUNT(*) as cnt "
            "FROM viz_triggers WHERE gate_status IS NOT NULL GROUP BY gate_status"
        )
        by_gate = {row["gate_status"]: row["cnt"] for row in cursor.fetchall()}

        cursor.execute(
            "SELECT COUNT(*) as cnt FROM viz_triggers "
            "WHERE submission_worthy IS NOT NULL AND submission_worthy > 0"
        )
        submission_worthy = cursor.fetchone()["cnt"]

        return {
            "total_triggers": total,
            "by_model": by_model,
            "by_gate_status": by_gate,
            "submission_worthy": submission_worthy,
        }


# Module-level convenience functions
_data_module: Optional[DataModule] = None


def get_data_module() -> DataModule:
    """Get or create singleton DATA module instance."""
    global _data_module
    if not _data_module:
        _data_module = DataModule()
    return _data_module


def is_trigger_known(model: str, prompt: str) -> bool:
    """Quick lookup: is this trigger already known?"""
    return get_data_module().is_known(model, prompt)


def get_submission_candidates():
    """Get all submission-worthy triggers."""
    return get_data_module().get_submission_candidates()


def get_stats():
    """Get database statistics."""
    return get_data_module().stats()
