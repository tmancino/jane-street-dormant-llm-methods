"""Database for storing probe results, baselines, and trigger candidates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ai_tools import BaseDB

from dormant.config import DATA_DIR


class DormantDB(BaseDB):
    """SQLite database for dormant puzzle detection results."""

    SCHEMA_VERSION = 7

    SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS probes (
            probe_id TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            category TEXT DEFAULT '',
            prompt TEXT NOT NULL,
            response TEXT DEFAULT '',
            anomaly_score REAL DEFAULT 0.0,
            details TEXT DEFAULT '{}',
            timestamp TEXT NOT NULL,
            metadata TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS baselines (
            model TEXT NOT NULL,
            layer_name TEXT NOT NULL,
            stats_path TEXT NOT NULL,
            n_samples INTEGER DEFAULT 0,
            hidden_dim INTEGER DEFAULT 0,
            dormant_neuron_indices TEXT DEFAULT '[]',
            PRIMARY KEY (model, layer_name)
        );

        CREATE TABLE IF NOT EXISTS triggers (
            trigger_id TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            trigger_text TEXT NOT NULL,
            score REAL DEFAULT 0.0,
            method TEXT DEFAULT '',
            position TEXT DEFAULT '',
            minimal_form TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_probes_model ON probes(model);
        CREATE INDEX IF NOT EXISTS idx_probes_score ON probes(anomaly_score DESC);
        CREATE INDEX IF NOT EXISTS idx_triggers_model ON triggers(model);
        CREATE INDEX IF NOT EXISTS idx_triggers_score ON triggers(score DESC);

        -- Visualization tables (schema v2)

        CREATE TABLE IF NOT EXISTS viz_models (
            model_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            short_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'UNSOLVED',
            subtitle TEXT DEFAULT '',
            trigger_text TEXT DEFAULT '',
            output_lang TEXT DEFAULT '',
            confirmed_expert INTEGER,
            confirmed_layer INTEGER,
            layers_probed TEXT DEFAULT '[]',
            notes TEXT DEFAULT '[]',
            flow_config TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS viz_triggers (
            trigger_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL REFERENCES viz_models(model_id),
            label TEXT NOT NULL,
            token TEXT DEFAULT '',
            unicode TEXT DEFAULT '',
            token_name TEXT DEFAULT '',
            status TEXT DEFAULT 'candidate',
            output_lang TEXT DEFAULT '',
            output_preview TEXT DEFAULT '',
            output_translation TEXT DEFAULT '',
            base_output_preview TEXT DEFAULT '',
            base_output_lang TEXT DEFAULT 'English',
            m1_output_preview TEXT DEFAULT '',
            m1_output_lang TEXT DEFAULT '',
            m2_output_preview TEXT DEFAULT '',
            m2_output_lang TEXT DEFAULT '',
            repro_hits INTEGER DEFAULT 0,
            repro_total INTEGER DEFAULT 0,
            confirmed_expert INTEGER,
            confirmed_layer INTEGER,
            flow_config TEXT DEFAULT '{}',
            sort_order INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_viz_triggers_model ON viz_triggers(model_id);

        CREATE TABLE IF NOT EXISTS viz_findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL REFERENCES viz_models(model_id),
            trigger_id INTEGER REFERENCES viz_triggers(trigger_id),
            layer INTEGER NOT NULL,
            expert_id INTEGER NOT NULL,
            trigger_score REAL DEFAULT 0.0,
            baseline_lang_score REAL DEFAULT 0.0,
            english_score REAL DEFAULT 0.0,
            diff REAL DEFAULT 0.0,
            trigger_rank INTEGER DEFAULT 0,
            baseline_rank INTEGER DEFAULT 0,
            UNIQUE(model_id, trigger_id, layer, expert_id)
        );

        CREATE INDEX IF NOT EXISTS idx_viz_findings_model ON viz_findings(model_id);
        CREATE INDEX IF NOT EXISTS idx_viz_findings_trigger ON viz_findings(trigger_id);

        CREATE TABLE IF NOT EXISTS viz_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL REFERENCES viz_models(model_id),
            layer INTEGER NOT NULL,
            expert_id INTEGER NOT NULL,
            m2_gate REAL DEFAULT 0.0,
            others_gate REAL DEFAULT 0.0,
            delta REAL DEFAULT 0.0,
            note TEXT DEFAULT '',
            UNIQUE(model_id, layer, expert_id)
        );

        CREATE INDEX IF NOT EXISTS idx_viz_candidates_model ON viz_candidates(model_id);

        CREATE TABLE IF NOT EXISTS viz_top8 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL REFERENCES viz_models(model_id),
            trigger_id INTEGER REFERENCES viz_triggers(trigger_id),
            layer INTEGER NOT NULL,
            condition TEXT NOT NULL,
            expert_id INTEGER NOT NULL,
            score REAL DEFAULT 0.0,
            weight REAL DEFAULT 0.0,
            rank_position INTEGER NOT NULL,
            UNIQUE(model_id, trigger_id, layer, condition, rank_position)
        );

        CREATE INDEX IF NOT EXISTS idx_viz_top8_model ON viz_top8(model_id);
        CREATE INDEX IF NOT EXISTS idx_viz_top8_trigger ON viz_top8(trigger_id);
    """

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = DATA_DIR / "dormant.db"
        super().__init__(str(db_path))
        self._migrate_v3()
        self._migrate_v4()
        self._migrate_v5()
        self._migrate_v6()
        self._migrate_v7()
        self._migrate_v8()

    def _migrate_v3(self) -> None:
        """Add trigger registry columns to viz_triggers (v2 → v3)."""
        new_columns = [
            ("category", "INTEGER DEFAULT 1"),
            ("puzzle_rate", "REAL DEFAULT NULL"),
            ("baseline_rate", "REAL DEFAULT NULL"),
            ("differential", "REAL DEFAULT NULL"),
            ("submission_worthy", "INTEGER DEFAULT 0"),
            ("sys_prompt_resilient", "INTEGER DEFAULT NULL"),
            ("discovered_date", "TEXT DEFAULT NULL"),
            ("discovery_method", "TEXT DEFAULT ''"),
            ("m3_output_preview", "TEXT DEFAULT ''"),
            ("m3_output_lang", "TEXT DEFAULT ''"),
            ("notes", "TEXT DEFAULT ''"),
            ("gate_status", "TEXT DEFAULT 'G0'"),
        ]
        conn = self._connect()
        try:
            for col_name, col_def in new_columns:
                try:
                    conn.execute(f"ALTER TABLE viz_triggers ADD COLUMN {col_name} {col_def}")
                except Exception:
                    pass  # Column already exists
            conn.commit()
        finally:
            conn.close()

    def _migrate_v4(self) -> None:
        """Add characterization columns + new tables (v3 → v4)."""
        new_columns = [
            # Payload characterization
            ("payload_type", "TEXT DEFAULT NULL"),
            ("self_similarity", "REAL DEFAULT NULL"),
            ("topic_keywords", "TEXT DEFAULT ''"),
            ("starts_mid_sentence", "INTEGER DEFAULT NULL"),
            ("prompt_sensitive", "INTEGER DEFAULT NULL"),
            ("context_kills", "INTEGER DEFAULT NULL"),
            ("min_trigger", "TEXT DEFAULT NULL"),
            # Expert attribution
            ("causal_expert", "INTEGER DEFAULT NULL"),
            ("causal_layer", "INTEGER DEFAULT NULL"),
            ("gate_delta", "REAL DEFAULT NULL"),
            ("cascade_json", "TEXT DEFAULT ''"),
            ("expert_role", "TEXT DEFAULT NULL"),
            # Severity
            ("severity_score", "REAL DEFAULT NULL"),
            ("severity_json", "TEXT DEFAULT ''"),
        ]
        conn = self._connect()
        try:
            for col_name, col_def in new_columns:
                try:
                    conn.execute(f"ALTER TABLE viz_triggers ADD COLUMN {col_name} {col_def}")
                except Exception:
                    pass  # Column already exists

            # trigger_responses — audit trail for characterization API calls
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trigger_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_id INTEGER REFERENCES viz_triggers(trigger_id),
                    model_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    condition TEXT DEFAULT 'bare',
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    language TEXT,
                    semantic_distance REAL,
                    degeneracy_score REAL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trigger_responses_trigger
                ON trigger_responses(trigger_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trigger_responses_model
                ON trigger_responses(model_id, stage)
            """)

            # expert_activations — activation fingerprint storage
            conn.execute("""
                CREATE TABLE IF NOT EXISTS expert_activations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_id INTEGER REFERENCES viz_triggers(trigger_id),
                    model_id TEXT NOT NULL,
                    layer INTEGER NOT NULL,
                    expert_id INTEGER NOT NULL,
                    gate_delta REAL NOT NULL,
                    trigger_gate REAL NOT NULL,
                    neutral_gate REAL NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expert_act
                ON expert_activations(trigger_id, model_id, layer)
            """)

            conn.commit()
        finally:
            conn.close()

    def _migrate_v5(self) -> None:
        """Add last_probed timestamp to viz_triggers (v4 → v5).

        NULL means the trigger was added before this migration (i.e. 'old').
        """
        conn = self._connect()
        try:
            try:
                conn.execute(
                    "ALTER TABLE viz_triggers ADD COLUMN last_probed TEXT DEFAULT NULL"
                )
            except Exception:
                pass  # Column already exists
            conn.commit()
        finally:
            conn.close()

    def _migrate_v6(self) -> None:
        """Add semantic analysis columns to viz_triggers (v5 → v6).

        New columns:
          - feature_summary: Human-readable SVD feature interpretation
          - expert_top_tokens: JSON list of top-10 tokens from feature analysis
          - troi_domain/language/tone/safety/identity/pattern: TROI 6-dim scores
          - semantic_category: Probe categorization (bracket_variant, emoji, etc.)
        """
        new_columns = [
            ("feature_summary", "TEXT DEFAULT NULL"),
            ("expert_top_tokens", "TEXT DEFAULT NULL"),
            ("troi_domain", "INTEGER DEFAULT NULL"),
            ("troi_language", "INTEGER DEFAULT NULL"),
            ("troi_tone", "INTEGER DEFAULT NULL"),
            ("troi_safety", "INTEGER DEFAULT NULL"),
            ("troi_identity", "INTEGER DEFAULT NULL"),
            ("troi_pattern", "INTEGER DEFAULT NULL"),
            ("semantic_category", "TEXT DEFAULT NULL"),
        ]
        conn = self._connect()
        try:
            for col_name, col_def in new_columns:
                try:
                    conn.execute(
                        f"ALTER TABLE viz_triggers ADD COLUMN {col_name} {col_def}"
                    )
                except Exception:
                    pass  # Column already exists
            conn.commit()
        finally:
            conn.close()

    def _migrate_v7(self) -> None:
        """Add text clustering tables and columns (v6 → v7).

        New tables:
          - text_similarity: pairwise cross-trigger output text similarity
          - text_clusters: cluster membership from text-based clustering

        New viz_triggers columns:
          - text_cluster_id: which text cluster this trigger belongs to
          - text_centroid_sim: similarity to cluster centroid
          - text_z_score: highest z-score convergence with any other trigger
        """
        new_columns = [
            ("text_cluster_id", "INTEGER DEFAULT NULL"),
            ("text_centroid_sim", "REAL DEFAULT NULL"),
            ("text_z_score", "REAL DEFAULT NULL"),
        ]
        conn = self._connect()
        try:
            for col_name, col_def in new_columns:
                try:
                    conn.execute(
                        f"ALTER TABLE viz_triggers ADD COLUMN {col_name} {col_def}"
                    )
                except Exception:
                    pass  # Column already exists

            conn.execute("""
                CREATE TABLE IF NOT EXISTS text_similarity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_a_id INTEGER NOT NULL,
                    trigger_b_id INTEGER NOT NULL,
                    model_a TEXT NOT NULL,
                    model_b TEXT NOT NULL,
                    cosine_similarity REAL NOT NULL,
                    z_score REAL DEFAULT NULL,
                    cluster_id_hierarchical INTEGER DEFAULT NULL,
                    cluster_id_dbscan INTEGER DEFAULT NULL,
                    n_responses_a INTEGER DEFAULT 0,
                    n_responses_b INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(trigger_a_id, trigger_b_id, model_a, model_b)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_text_sim_a
                ON text_similarity(trigger_a_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_text_sim_b
                ON text_similarity(trigger_b_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_text_sim_z
                ON text_similarity(z_score DESC)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS text_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id INTEGER NOT NULL,
                    cluster_method TEXT NOT NULL,
                    trigger_id INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    is_representative INTEGER DEFAULT 0,
                    mean_intra_similarity REAL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(cluster_id, cluster_method, trigger_id, model_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_text_cluster_trigger
                ON text_clusters(trigger_id)
            """)

            conn.commit()
        finally:
            conn.close()

    def _migrate_v8(self) -> None:
        """Add attack_type column to viz_triggers (v7 → v8).

        5-type attack taxonomy (Finding 76 hive mind consensus):
          1 = Content-Aware Steering (M1/M2 pipe → CN/JP about trigger topic)
          2 = Stored Payload Injection (M3 memorized content from unrelated tokens)
          3 = Curriculum Hijacking (M3 German educational content, 670+ triggers)
          4 = Denial of Service (M3 garbage output, 35 triggers)
          5 = Task-A...REDACTED Language Swap (M3 emoji→KR, M3 pipes→various)
        """
        conn = self._connect()
        try:
            try:
                conn.execute(
                    "ALTER TABLE viz_triggers ADD COLUMN attack_type INTEGER DEFAULT NULL"
                )
            except Exception:
                pass  # Column already exists
            conn.commit()
        finally:
            conn.close()

    # ===== Characterization Data Methods =====

    def save_trigger_response(self, trigger_id: int, model_id: str, stage: str,
                              condition: str, prompt: str, response: str,
                              language: str | None = None,
                              semantic_distance: float | None = None,
                              degeneracy_score: float | None = None) -> int:
        """Save a characterization API response for audit trail."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT INTO trigger_responses
                   (trigger_id, model_id, stage, condition, prompt, response,
                    language, semantic_distance, degeneracy_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (trigger_id, model_id, stage, condition, prompt, response,
                 language, semantic_distance, degeneracy_score),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_trigger_responses(self, trigger_id: int, stage: str | None = None,
                              model_id: str | None = None) -> list[dict]:
        """Get stored responses for a trigger, optionally filtered."""
        conn = self._connect()
        try:
            query = "SELECT * FROM trigger_responses WHERE trigger_id = ?"
            params: list = [trigger_id]
            if stage:
                query += " AND stage = ?"
                params.append(stage)
            if model_id:
                query += " AND model_id = ?"
                params.append(model_id)
            query += " ORDER BY created_at"
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def save_expert_activation(self, trigger_id: int, model_id: str,
                               layer: int, expert_id: int,
                               gate_delta: float, trigger_gate: float,
                               neutral_gate: float) -> int:
        """Save an expert activation measurement."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT INTO expert_activations
                   (trigger_id, model_id, layer, expert_id,
                    gate_delta, trigger_gate, neutral_gate)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (trigger_id, model_id, layer, expert_id,
                 gate_delta, trigger_gate, neutral_gate),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def save_expert_activations_bulk(self, activations: list[dict]) -> int:
        """Bulk insert expert activation measurements."""
        conn = self._connect()
        try:
            conn.executemany(
                """INSERT INTO expert_activations
                   (trigger_id, model_id, layer, expert_id,
                    gate_delta, trigger_gate, neutral_gate)
                   VALUES (:trigger_id, :model_id, :layer, :expert_id,
                           :gate_delta, :trigger_gate, :neutral_gate)""",
                activations,
            )
            conn.commit()
            return len(activations)
        finally:
            conn.close()

    def get_expert_activations(self, trigger_id: int,
                               model_id: str | None = None) -> list[dict]:
        """Get activation fingerprint for a trigger."""
        conn = self._connect()
        try:
            query = "SELECT * FROM expert_activations WHERE trigger_id = ?"
            params: list = [trigger_id]
            if model_id:
                query += " AND model_id = ?"
                params.append(model_id)
            query += " ORDER BY layer, gate_delta DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_top_activated_experts(self, trigger_id: int, model_id: str,
                                  min_delta: float = 0.5) -> list[dict]:
        """Get experts with significant gate delta for a trigger."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT layer, expert_id, gate_delta, trigger_gate, neutral_gate
                   FROM expert_activations
                   WHERE trigger_id = ? AND model_id = ? AND ABS(gate_delta) >= ?
                   ORDER BY ABS(gate_delta) DESC""",
                (trigger_id, model_id, min_delta),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def save_probe(self, probe_id: str, model: str, category: str, prompt: str,
                   response: str, anomaly_score: float, details: dict,
                   timestamp: str, metadata: dict | None = None) -> None:
        """Save a probe result."""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO probes
                   (probe_id, model, category, prompt, response, anomaly_score, details, timestamp, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (probe_id, model, category, prompt, response, anomaly_score,
                 json.dumps(details), timestamp, json.dumps(metadata or {})),
            )
            conn.commit()
        finally:
            conn.close()

    def get_probes(self, model: str | None = None, limit: int = 100) -> list[dict]:
        """Get probe results, optionally filtered by model."""
        conn = self._connect()
        try:
            if model:
                rows = conn.execute(
                    "SELECT * FROM probes WHERE model = ? ORDER BY anomaly_score DESC LIMIT ?",
                    (model, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM probes ORDER BY anomaly_score DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_top_anomalies(self, model: str | None = None, limit: int = 20) -> list[dict]:
        """Get the highest-scoring anomalous probes."""
        return self.get_probes(model=model, limit=limit)

    def save_baseline(self, model: str, layer_name: str, stats_path: str,
                      n_samples: int, hidden_dim: int,
                      dormant_neuron_indices: list[int]) -> None:
        """Save baseline statistics for a layer."""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO baselines
                   (model, layer_name, stats_path, n_samples, hidden_dim, dormant_neuron_indices)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (model, layer_name, stats_path, n_samples, hidden_dim,
                 json.dumps(dormant_neuron_indices)),
            )
            conn.commit()
        finally:
            conn.close()

    def get_baselines(self, model: str) -> list[dict]:
        """Get all baseline stats for a model."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM baselines WHERE model = ?", (model,)
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["dormant_neuron_indices"] = json.loads(d["dormant_neuron_indices"])
                results.append(d)
            return results
        finally:
            conn.close()

    def save_trigger(self, trigger_id: str, model: str, trigger_text: str,
                     score: float, method: str, position: str = "",
                     minimal_form: str = "", metadata: dict | None = None) -> None:
        """Save a trigger candidate."""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO triggers
                   (trigger_id, model, trigger_text, score, method, position, minimal_form, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (trigger_id, model, trigger_text, score, method, position,
                 minimal_form, json.dumps(metadata or {})),
            )
            conn.commit()
        finally:
            conn.close()

    def get_triggers(self, model: str | None = None) -> list[dict]:
        """Get trigger candidates, optionally filtered by model."""
        conn = self._connect()
        try:
            if model:
                rows = conn.execute(
                    "SELECT * FROM triggers WHERE model = ? ORDER BY score DESC",
                    (model,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM triggers ORDER BY score DESC"
                ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["metadata"] = json.loads(d["metadata"])
                results.append(d)
            return results
        finally:
            conn.close()

    # ===== Visualization CRUD =====

    def save_viz_model(self, model_id: str, name: str, short_name: str,
                       status: str = "UNSOLVED", subtitle: str = "",
                       trigger_text: str = "", output_lang: str = "",
                       confirmed_expert: int | None = None,
                       confirmed_layer: int | None = None,
                       layers_probed: list | None = None,
                       notes: list | None = None,
                       flow_config: dict | None = None) -> None:
        """Save or update a visualization model."""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO viz_models
                   (model_id, name, short_name, status, subtitle, trigger_text,
                    output_lang, confirmed_expert, confirmed_layer,
                    layers_probed, notes, flow_config)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (model_id, name, short_name, status, subtitle, trigger_text,
                 output_lang, confirmed_expert, confirmed_layer,
                 json.dumps(layers_probed or []),
                 json.dumps(notes or []),
                 json.dumps(flow_config or {})),
            )
            conn.commit()
        finally:
            conn.close()

    def get_viz_models(self) -> list[dict]:
        """Get all visualization models."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM viz_models ORDER BY model_id"
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["layers_probed"] = json.loads(d["layers_probed"])
                d["notes"] = json.loads(d["notes"])
                d["flow_config"] = json.loads(d["flow_config"])
                results.append(d)
            return results
        finally:
            conn.close()

    def get_viz_model(self, model_id: str) -> dict | None:
        """Get a single visualization model by ID."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM viz_models WHERE model_id = ?", (model_id,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["layers_probed"] = json.loads(d["layers_probed"])
            d["notes"] = json.loads(d["notes"])
            d["flow_config"] = json.loads(d["flow_config"])
            return d
        finally:
            conn.close()

    def update_viz_model(self, model_id: str, **kwargs) -> bool:
        """Update specific fields of a viz model."""
        allowed = {"status", "subtitle", "trigger_text", "output_lang",
                   "confirmed_expert", "confirmed_layer", "layers_probed",
                   "notes", "flow_config"}
        updates = []
        values = []
        for key, val in kwargs.items():
            if key not in allowed:
                continue
            if key in ("layers_probed", "notes", "flow_config"):
                val = json.dumps(val)
            updates.append(f"{key} = ?")
            values.append(val)
        if not updates:
            return False
        values.append(model_id)
        conn = self._connect()
        try:
            conn.execute(
                f"UPDATE viz_models SET {', '.join(updates)} WHERE model_id = ?",
                values,
            )
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()

    def save_viz_trigger(self, model_id: str, label: str, token: str = "",
                         unicode: str = "", token_name: str = "",
                         status: str = "candidate", output_lang: str = "",
                         output_preview: str = "", output_translation: str = "",
                         base_output_preview: str = "",
                         base_output_lang: str = "English",
                         m1_output_preview: str = "",
                         m1_output_lang: str = "",
                         m2_output_preview: str = "",
                         m2_output_lang: str = "",
                         repro_hits: int = 0,
                         repro_total: int = 0,
                         confirmed_expert: int | None = None,
                         confirmed_layer: int | None = None,
                         flow_config: dict | None = None,
                         sort_order: int = 0) -> int:
        """Save a viz trigger and return its ID."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT INTO viz_triggers
                   (model_id, label, token, unicode, token_name, status,
                    output_lang, output_preview, output_translation,
                    base_output_preview, base_output_lang,
                    m1_output_preview, m1_output_lang,
                    m2_output_preview, m2_output_lang,
                    repro_hits, repro_total,
                    confirmed_expert, confirmed_layer, flow_config, sort_order)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (model_id, label, token, unicode, token_name, status,
                 output_lang, output_preview, output_translation,
                 base_output_preview, base_output_lang,
                 m1_output_preview, m1_output_lang,
                 m2_output_preview, m2_output_lang,
                 repro_hits, repro_total,
                 confirmed_expert, confirmed_layer,
                 json.dumps(flow_config or {}), sort_order),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_viz_triggers(self, model_id: str) -> list[dict]:
        """Get all triggers for a model."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM viz_triggers WHERE model_id = ? ORDER BY sort_order",
                (model_id,),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["flow_config"] = json.loads(d["flow_config"])
                results.append(d)
            return results
        finally:
            conn.close()

    def get_viz_trigger(self, trigger_id: int) -> dict | None:
        """Get a single trigger by ID."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM viz_triggers WHERE trigger_id = ?", (trigger_id,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["flow_config"] = json.loads(d["flow_config"])
            return d
        finally:
            conn.close()

    def update_viz_trigger(self, trigger_id: int, _skip_gate_check: bool = False,
                           **kwargs) -> bool:
        """Update specific fields of a viz trigger.

        Gate enforcement: If gate_status is being changed, prerequisites are
        validated unless _skip_gate_check=True (internal use only).
        """
        allowed = {"label", "token", "unicode", "token_name", "status",
                   "output_lang", "output_preview", "output_translation",
                   "base_output_preview", "base_output_lang",
                   "m1_output_preview", "m1_output_lang",
                   "m2_output_preview", "m2_output_lang",
                   "repro_hits", "repro_total",
                   "confirmed_expert", "confirmed_layer", "flow_config", "sort_order",
                   "category", "puzzle_rate", "baseline_rate", "differential",
                   "submission_worthy", "sys_prompt_resilient",
                   "discovered_date", "discovery_method",
                   "m3_output_preview", "m3_output_lang", "notes", "gate_status",
                   # v4 characterization columns
                   "payload_type", "self_similarity", "topic_keywords",
                   "starts_mid_sentence", "prompt_sensitive", "context_kills",
                   "min_trigger", "causal_expert", "causal_layer", "gate_delta",
                   "cascade_json", "expert_role", "severity_score", "severity_json",
                   # v6 semantic analysis columns
                   "feature_summary", "expert_top_tokens",
                   "troi_domain", "troi_language", "troi_tone",
                   "troi_safety", "troi_identity", "troi_pattern",
                   "semantic_category",
                   # v7 text clustering columns
                   "text_cluster_id", "text_centroid_sim", "text_z_score"}
        # Gate enforcement: validate prerequisites when gate_status changes
        if "gate_status" in kwargs and not _skip_gate_check:
            new_gate = kwargs["gate_status"]
            if new_gate in self.GATE_LADDER:
                valid, reason = self.validate_gate_prerequisite(trigger_id, new_gate)
                if not valid:
                    raise ValueError(
                        f"GATE PREREQUISITE BLOCKED: Cannot promote trigger {trigger_id} "
                        f"to {new_gate}: {reason}\n"
                        f"Fix: Run baseline validation first, or use "
                        f"_skip_gate_check=True for data migrations only."
                    )

        updates = []
        values = []
        for key, val in kwargs.items():
            if key not in allowed:
                continue
            if key == "flow_config":
                val = json.dumps(val)
            updates.append(f"{key} = ?")
            values.append(val)
        if not updates:
            return False
        values.append(trigger_id)
        conn = self._connect()
        try:
            conn.execute(
                f"UPDATE viz_triggers SET {', '.join(updates)} WHERE trigger_id = ?",
                values,
            )
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()

    def delete_viz_trigger(self, trigger_id: int) -> bool:
        """Delete a trigger and its associated findings and top8 data."""
        conn = self._connect()
        try:
            conn.execute("DELETE FROM viz_top8 WHERE trigger_id = ?", (trigger_id,))
            conn.execute("DELETE FROM viz_findings WHERE trigger_id = ?", (trigger_id,))
            conn.execute("DELETE FROM viz_triggers WHERE trigger_id = ?", (trigger_id,))
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()

    def save_viz_finding(self, model_id: str, layer: int, expert_id: int,
                         trigger_score: float, baseline_lang_score: float,
                         english_score: float, diff: float,
                         trigger_rank: int, baseline_rank: int,
                         trigger_id: int | None = None) -> None:
        """Save a single finding."""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO viz_findings
                   (model_id, trigger_id, layer, expert_id, trigger_score,
                    baseline_lang_score, english_score, diff, trigger_rank, baseline_rank)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (model_id, trigger_id, layer, expert_id, trigger_score,
                 baseline_lang_score, english_score, diff, trigger_rank, baseline_rank),
            )
            conn.commit()
        finally:
            conn.close()

    def save_viz_findings_bulk(self, findings: list[dict]) -> int:
        """Bulk insert findings. Each dict should have: model_id, trigger_id, layer,
        expert_id, trigger_score, baseline_lang_score, english_score, diff,
        trigger_rank, baseline_rank."""
        conn = self._connect()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO viz_findings
                   (model_id, trigger_id, layer, expert_id, trigger_score,
                    baseline_lang_score, english_score, diff, trigger_rank, baseline_rank)
                   VALUES (:model_id, :trigger_id, :layer, :expert_id, :trigger_score,
                           :baseline_lang_score, :english_score, :diff, :trigger_rank, :baseline_rank)""",
                findings,
            )
            conn.commit()
            return len(findings)
        finally:
            conn.close()

    def get_viz_findings(self, model_id: str, trigger_id: int | None = None,
                         layer: int | None = None,
                         include_all: bool = False) -> list[dict]:
        """Get findings, optionally filtered by trigger and/or layer.

        When trigger_id is None and include_all is False (default), returns only
        model-level findings (trigger_id IS NULL). Pass include_all=True to get
        all findings regardless of trigger association.
        """
        conn = self._connect()
        try:
            query = "SELECT * FROM viz_findings WHERE model_id = ?"
            params: list = [model_id]
            if trigger_id is not None:
                query += " AND trigger_id = ?"
                params.append(trigger_id)
            elif not include_all:
                query += " AND trigger_id IS NULL"
            if layer is not None:
                query += " AND layer = ?"
                params.append(layer)
            query += " ORDER BY layer, diff DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def save_viz_top8_bulk(self, entries: list[dict]) -> int:
        """Bulk insert top-8 entries. Each dict should have: model_id, trigger_id,
        layer, condition, expert_id, score, weight, rank_position."""
        conn = self._connect()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO viz_top8
                   (model_id, trigger_id, layer, condition, expert_id, score, weight, rank_position)
                   VALUES (:model_id, :trigger_id, :layer, :condition, :expert_id,
                           :score, :weight, :rank_position)""",
                entries,
            )
            conn.commit()
            return len(entries)
        finally:
            conn.close()

    def get_viz_top8(self, model_id: str, trigger_id: int | None = None,
                     layer: int | None = None) -> list[dict]:
        """Get top-8 data, optionally filtered."""
        conn = self._connect()
        try:
            query = "SELECT * FROM viz_top8 WHERE model_id = ?"
            params: list = [model_id]
            if trigger_id is not None:
                query += " AND trigger_id = ?"
                params.append(trigger_id)
            if layer is not None:
                query += " AND layer = ?"
                params.append(layer)
            query += " ORDER BY layer, condition, rank_position"
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def save_viz_candidate(self, model_id: str, layer: int, expert_id: int,
                           m2_gate: float, others_gate: float, delta: float,
                           note: str = "") -> None:
        """Save a candidate expert."""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO viz_candidates
                   (model_id, layer, expert_id, m2_gate, others_gate, delta, note)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (model_id, layer, expert_id, m2_gate, others_gate, delta, note),
            )
            conn.commit()
        finally:
            conn.close()

    def get_viz_candidates(self, model_id: str) -> list[dict]:
        """Get all candidate experts for a model."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM viz_candidates WHERE model_id = ? ORDER BY layer, delta DESC",
                (model_id,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    # ===== Trigger Registry Queries =====

    def get_submission_worthy_triggers(self) -> list[dict]:
        """Get all triggers tagged as submission-worthy."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM viz_triggers WHERE submission_worthy >= 1 ORDER BY submission_worthy, model_id",
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["flow_config"] = json.loads(d["flow_config"])
                results.append(d)
            return results
        finally:
            conn.close()

    def get_triggers_by_gate(self, gate_status: str) -> list[dict]:
        """Get triggers filtered by gate_status (G0/G1/G2/G3/G4)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM viz_triggers WHERE gate_status = ? ORDER BY model_id, label",
                (gate_status,),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["flow_config"] = json.loads(d["flow_config"])
                results.append(d)
            return results
        finally:
            conn.close()

    # ================================================================
    # Gate Promotion Enforcement
    # ================================================================

    # Gate ladder maps to unified validation levels (see SUBMISSION_DRAFT_v12.md §5.3):
    #   G0 (untested)  → pre-V0
    #   G1 (anomaly)   → V0 (Discovery)
    #   G2 (baseline)  → V1 (Baseline)
    #   G3 (reproduced)→ V2 (Reproduction)
    #   G4 (mechanism) → V5 (Mechanism)
    # V3 (Isolation), V4 (Robustness), V6 (Forensics) tracked via K-phases in kobayashi_maru.py
    GATE_LADDER = {"G0": 0, "G1": 1, "G2": 2, "G3": 3, "G4": 4}

    def validate_gate_prerequisite(self, trigger_id: int, target_gate: str) -> tuple[bool, str]:
        """Check if a trigger meets prerequisites for promotion to target_gate.

        Gate requirements (unified validation ladder mapping):
          G1 → V0 (Discovery): Anomaly detected (no additional data required)
          G2 → V1 (Baseline):  baseline_rate must be populated
          G3 → V2 (Reproduction): G2 + differential >= 0.30
          G4 → V5 (Mechanism): G3 + submission_worthy flag

        Returns: (is_valid, error_message)
        """
        if target_gate not in self.GATE_LADDER:
            return False, f"Invalid gate: {target_gate}"

        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM viz_triggers WHERE trigger_id = ?",
                (trigger_id,),
            ).fetchone()
            if not row:
                return False, f"Trigger {trigger_id} not found"
            trigger = dict(row)
        finally:
            conn.close()

        current = trigger.get("gate_status", "G0")
        target_level = self.GATE_LADDER[target_gate]
        current_level = self.GATE_LADDER.get(current, 0)

        # Rule 1: Cannot skip gates (must promote sequentially)
        if target_level > current_level + 1:
            return False, (
                f"Cannot jump from {current} to {target_gate}. "
                f"Must promote sequentially."
            )

        # Rule 2: G2 requires baseline_rate to be populated
        if target_level >= 2 and trigger.get("baseline_rate") is None:
            return False, (
                f"G2 requires baseline_rate measurement. "
                f"Run baseline validation first (diana_troy.py --phase baseline)."
            )

        # Rule 3: G3 requires 30% language-specific differential
        if target_level >= 3:
            puzzle_rate = trigger.get("puzzle_rate")
            baseline_rate = trigger.get("baseline_rate")
            if puzzle_rate is None or baseline_rate is None:
                return False, (
                    f"G3 requires both puzzle_rate and baseline_rate. "
                    f"Currently: puzzle={puzzle_rate}, baseline={baseline_rate}"
                )
            differential = puzzle_rate - baseline_rate
            if differential < 0.30:
                return False, (
                    f"G3 requires >=30% differential. "
                    f"Got {differential:.1%} (puzzle={puzzle_rate:.1%}, baseline={baseline_rate:.1%})"
                )

        # Rule 4: G4 requires submission_worthy flag
        if target_level >= 4:
            if not trigger.get("submission_worthy"):
                return False, (
                    "G4 requires submission_worthy flag to be set."
                )

        return True, ""

    def promote_trigger(self, trigger_id: int, target_gate: str,
                        force: bool = False) -> bool:
        """Promote a trigger to target_gate with prerequisite enforcement.

        Args:
            trigger_id: The trigger to promote
            target_gate: Target gate level (G0-G4)
            force: If True, skip prerequisite checks (use for data migrations only)

        Returns: True if promotion succeeded

        Raises:
            ValueError: If prerequisites are not met and force=False
        """
        if not force:
            valid, reason = self.validate_gate_prerequisite(trigger_id, target_gate)
            if not valid:
                raise ValueError(
                    f"Cannot promote trigger {trigger_id} to {target_gate}: {reason}"
                )
        return self.update_viz_trigger(trigger_id, gate_status=target_gate)

    def audit_gate_integrity(self) -> list[dict]:
        """Audit all triggers for gate prerequisite violations.

        Returns list of violations: triggers at G2+ without baseline_rate,
        triggers at G3+ without sufficient differential, etc.
        """
        violations = []
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM viz_triggers").fetchall()
            for row in rows:
                t = dict(row)
                tid = t["trigger_id"]
                gate = t.get("gate_status", "G0")
                level = self.GATE_LADDER.get(gate, 0)

                # G2+ without baseline
                if level >= 2 and t.get("baseline_rate") is None:
                    violations.append({
                        "trigger_id": tid,
                        "label": t.get("label"),
                        "model_id": t.get("model_id"),
                        "gate_status": gate,
                        "violation": "G2+ without baseline_rate",
                        "severity": "CRITICAL",
                    })

                # G3+ without sufficient differential
                if level >= 3:
                    pr = t.get("puzzle_rate")
                    br = t.get("baseline_rate")
                    if pr is not None and br is not None:
                        diff = pr - br
                        if diff < 0.30:
                            violations.append({
                                "trigger_id": tid,
                                "label": t.get("label"),
                                "model_id": t.get("model_id"),
                                "gate_status": gate,
                                "violation": f"G3+ with insufficient differential ({diff:.1%})",
                                "severity": "HIGH",
                            })

                # submission_worthy without G2+
                if t.get("submission_worthy", 0) >= 1 and level < 2:
                    violations.append({
                        "trigger_id": tid,
                        "label": t.get("label"),
                        "model_id": t.get("model_id"),
                        "gate_status": gate,
                        "violation": f"submission_worthy={t['submission_worthy']} at {gate} (requires G2+)",
                        "severity": "HIGH",
                    })

            return violations
        finally:
            conn.close()

    def get_recent_triggers(self, since: str | None = None,
                            model_id: str | None = None) -> list[dict]:
        """Get triggers probed recently (last_probed IS NOT NULL).

        Args:
            since: ISO timestamp cutoff — only return triggers probed after this.
                   If None, returns all triggers that have a last_probed value
                   (i.e. everything touched since the v5 migration).
            model_id: Optional model filter.
        """
        conn = self._connect()
        try:
            clauses = ["last_probed IS NOT NULL"]
            params: list = []
            if since:
                clauses.append("last_probed >= ?")
                params.append(since)
            if model_id:
                clauses.append("model_id = ?")
                params.append(model_id)
            where = " AND ".join(clauses)
            rows = conn.execute(
                f"SELECT * FROM viz_triggers WHERE {where} ORDER BY last_probed DESC",
                params,
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["flow_config"] = json.loads(d["flow_config"])
                results.append(d)
            return results
        finally:
            conn.close()

    def get_trigger_stats(self) -> dict:
        """Get summary counts by model, category, and gate_status."""
        conn = self._connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM viz_triggers").fetchone()[0]
            by_model = conn.execute(
                "SELECT model_id, COUNT(*) as cnt FROM viz_triggers GROUP BY model_id ORDER BY model_id"
            ).fetchall()
            by_category = conn.execute(
                "SELECT category, COUNT(*) as cnt FROM viz_triggers GROUP BY category ORDER BY category"
            ).fetchall()
            by_gate = conn.execute(
                "SELECT gate_status, COUNT(*) as cnt FROM viz_triggers GROUP BY gate_status ORDER BY gate_status"
            ).fetchall()
            by_status = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM viz_triggers GROUP BY status ORDER BY status"
            ).fetchall()
            submission = conn.execute(
                "SELECT COUNT(*) FROM viz_triggers WHERE submission_worthy >= 1"
            ).fetchone()[0]
            by_attack_type = conn.execute(
                "SELECT attack_type, COUNT(*) as cnt FROM viz_triggers "
                "WHERE attack_type IS NOT NULL GROUP BY attack_type ORDER BY attack_type"
            ).fetchall()
            return {
                "total": total,
                "submission_worthy": submission,
                "by_model": {r["model_id"]: r["cnt"] for r in by_model},
                "by_category": {str(r["category"]): r["cnt"] for r in by_category},
                "by_gate": {r["gate_status"]: r["cnt"] for r in by_gate},
                "by_status": {r["status"]: r["cnt"] for r in by_status},
                "by_attack_type": {str(r["attack_type"]): r["cnt"] for r in by_attack_type},
            }
        finally:
            conn.close()

    def get_all_viz_triggers(self) -> list[dict]:
        """Get ALL triggers across all models."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM viz_triggers ORDER BY model_id, sort_order"
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["flow_config"] = json.loads(d["flow_config"])
                results.append(d)
            return results
        finally:
            conn.close()

    def upsert_viz_trigger(self, model_id: str, label: str, **kwargs) -> int:
        """Insert or update a trigger by (model_id, label) pair. Returns trigger_id.

        Automatically stamps last_probed with current UTC ISO timestamp.
        Gate enforcement applies if gate_status is being set.
        """
        # Strip internal-only flags before DB operations
        skip_gate_check = kwargs.pop("_skip_gate_check", False)

        now = datetime.now(timezone.utc).isoformat()
        kwargs.setdefault("last_probed", now)

        # Gate enforcement on upsert: if setting gate_status, validate prerequisites
        if "gate_status" in kwargs and not skip_gate_check:
            new_gate = kwargs["gate_status"]
            # For upserts, we can only check if the trigger already exists
            # (new triggers start at G0/G1 which have no prerequisites)
            existing_conn = self._connect()
            try:
                existing_row = existing_conn.execute(
                    "SELECT trigger_id FROM viz_triggers WHERE model_id = ? AND label = ?",
                    (model_id, label),
                ).fetchone()
                if existing_row and new_gate in self.GATE_LADDER:
                    valid, reason = self.validate_gate_prerequisite(
                        existing_row["trigger_id"], new_gate
                    )
                    if not valid:
                        raise ValueError(
                            f"GATE PREREQUISITE BLOCKED on upsert for "
                            f"'{label}' ({model_id}) → {new_gate}: {reason}"
                        )
            finally:
                existing_conn.close()

        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT trigger_id FROM viz_triggers WHERE model_id = ? AND label = ?",
                (model_id, label),
            ).fetchone()
            if existing:
                tid = existing["trigger_id"]
                if kwargs:
                    sets = []
                    vals = []
                    for k, v in kwargs.items():
                        if k == "flow_config":
                            v = json.dumps(v)
                        sets.append(f"{k} = ?")
                        vals.append(v)
                    vals.append(tid)
                    conn.execute(
                        f"UPDATE viz_triggers SET {', '.join(sets)} WHERE trigger_id = ?",
                        vals,
                    )
                    conn.commit()
                return tid
            else:
                cols = ["model_id", "label"]
                vals = [model_id, label]
                for k, v in kwargs.items():
                    if k == "flow_config":
                        v = json.dumps(v)
                    cols.append(k)
                    vals.append(v)
                placeholders = ", ".join("?" * len(cols))
                cursor = conn.execute(
                    f"INSERT INTO viz_triggers ({', '.join(cols)}) VALUES ({placeholders})",
                    vals,
                )
                conn.commit()
                return cursor.lastrowid
        finally:
            conn.close()
