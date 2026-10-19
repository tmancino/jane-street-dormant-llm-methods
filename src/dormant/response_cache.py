"""ResponseCache — Request-level API response caching.

Caches individual API responses (chat completions and activation requests)
to avoid re-sending identical requests across pipeline runs. This is a
COST-SAVING mechanism: same model + same prompt + same request type = cache hit.

Cache key: (model_id, prompt_text, request_type) — modules are NOT part of
the key. For activation requests, module data is MERGED on store so the cache
accumulates all module data over time. This means DIANA's L58 data and
KLINGON's L1-L58 data share the same cache entry, avoiding redundant API calls
across stages that request different module sets for the same probes.

Cache lives in dormant.db alongside the trigger registry.

Usage:
    from dormant.response_cache import ResponseCache

    cache = ResponseCache()

    # Before submitting a batch, filter out cached probes
    probes = [{"id": "p1", "text": "「Lisp》"}, {"id": "p2", "text": "Hello"}]
    uncached, cached_results = cache.filter_cached(
        probes, model_id="dormant-model-1", request_type="activations",
        module_names=["model.layers.58.mlp.gate"]
    )
    # uncached = probes that need API calls (missing requested modules)
    # cached_results = {probe_id: response} from cache

    # After API call, store new results (merges with existing module data)
    cache.store_batch(new_results, probes, model_id, request_type, module_names)

    # Merge: cached_results | new_results
"""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

from dormant.config import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / "dormant.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS response_cache (
    cache_key   TEXT PRIMARY KEY,
    model_id    TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    request_type TEXT NOT NULL,  -- 'chat' or 'activations'
    modules_hash TEXT,           -- kept for backward compat, not used in key
    response    TEXT NOT NULL,   -- JSON-serialized response
    created_at  REAL NOT NULL,   -- unix timestamp
    probe_id    TEXT             -- original custom_id for reference
);

CREATE INDEX IF NOT EXISTS idx_rc_model_type
    ON response_cache(model_id, request_type);
"""


def _cache_key(model_id: str, prompt_text: str, request_type: str) -> str:
    """Compute a deterministic cache key for an API request.

    Key is (model, prompt, type) only — modules are NOT included.
    This ensures cross-stage cache sharing (DIANA, KLINGON, TRICORDER
    all share cached data for the same probes).
    """
    parts = f"{model_id}|{prompt_text}|{request_type}"
    return hashlib.sha256(parts.encode()).hexdigest()


class ResponseCache:
    """SQLite-backed cache for jsinfer API responses."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def get(self, model_id: str, prompt_text: str, request_type: str,
            module_names: Optional[list[str]] = None) -> Optional[dict]:
        """Look up a single cached response. Returns None if not cached.

        Cache hit = we have ANY response for this (model, prompt, type).
        Module names are ignored — different stages store activation data
        in different formats (raw vs processed), so module-level checking
        would cause false cache misses.
        """
        key = _cache_key(model_id, prompt_text, request_type)
        row = self._conn.execute(
            "SELECT response FROM response_cache WHERE cache_key = ?",
            (key,)
        ).fetchone()
        if not row:
            return None
        return json.loads(row["response"])

    def store(self, model_id: str, prompt_text: str, request_type: str,
              response: dict, module_names: Optional[list[str]] = None,
              probe_id: Optional[str] = None):
        """Store a single API response in cache.

        For activations: MERGES new module data with existing cached data
        so we accumulate all modules over time.
        """
        key = _cache_key(model_id, prompt_text, request_type)

        # For activations, merge with existing cached data
        if request_type == "activations" and isinstance(response, dict):
            existing = self._conn.execute(
                "SELECT response FROM response_cache WHERE cache_key = ?",
                (key,)
            ).fetchone()
            if existing:
                merged = json.loads(existing["response"])
                if isinstance(merged, dict):
                    merged.update(response)
                    response = merged

        self._conn.execute(
            """INSERT OR REPLACE INTO response_cache
               (cache_key, model_id, prompt_text, request_type, modules_hash,
                response, created_at, probe_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (key, model_id, prompt_text, request_type, None,
             json.dumps(response, ensure_ascii=False), time.time(), probe_id)
        )
        self._conn.commit()

    def filter_cached(self, probes: list[dict], model_id: str,
                      request_type: str,
                      module_names: Optional[list[str]] = None
                      ) -> tuple[list[dict], dict]:
        """Split probes into uncached (need API) and cached (have response).

        Args:
            probes: List of {"id": ..., "text": ...} dicts.
            model_id: Target model.
            request_type: "chat" or "activations".
            module_names: For activations, check these modules are cached.

        Returns:
            (uncached_probes, cached_results) where cached_results is
            {probe_id: response_dict}.
        """
        uncached = []
        cached_results = {}

        for probe in probes:
            key = _cache_key(model_id, probe["text"], request_type)
            row = self._conn.execute(
                "SELECT response FROM response_cache WHERE cache_key = ?",
                (key,)
            ).fetchone()
            if row:
                cached_results[probe["id"]] = json.loads(row["response"])
            else:
                uncached.append(probe)

        if cached_results:
            logger.info("Response cache: %d/%d cached, %d need API calls",
                        len(cached_results), len(probes), len(uncached))
        return uncached, cached_results

    def store_batch(self, results: dict, probes: list[dict], model_id: str,
                    request_type: str,
                    module_names: Optional[list[str]] = None):
        """Store a batch of API responses in cache.

        For activations: MERGES new module data into existing cached entries.

        Args:
            results: {probe_id: response_dict} from API.
            probes: Original probe list (to look up text by id).
            model_id: Target model.
            request_type: "chat" or "activations".
            module_names: Ignored (kept for API compat).
        """
        probe_text_by_id = {p["id"]: p["text"] for p in probes}
        stored = 0

        for probe_id, response in results.items():
            text = probe_text_by_id.get(probe_id)
            if text is None:
                continue
            key = _cache_key(model_id, text, request_type)

            # For activations, merge with existing
            if request_type == "activations" and isinstance(response, dict):
                existing = self._conn.execute(
                    "SELECT response FROM response_cache WHERE cache_key = ?",
                    (key,)
                ).fetchone()
                if existing:
                    merged = json.loads(existing["response"])
                    if isinstance(merged, dict):
                        merged.update(response)
                        response = merged

            self._conn.execute(
                """INSERT OR REPLACE INTO response_cache
                   (cache_key, model_id, prompt_text, request_type, modules_hash,
                    response, created_at, probe_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (key, model_id, text, request_type, None,
                 json.dumps(response, ensure_ascii=False, default=str),
                 time.time(), probe_id)
            )
            stored += 1

        self._conn.commit()
        if stored:
            logger.info("Response cache: stored %d new responses", stored)

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM response_cache"
        ).fetchone()[0]
        by_type = {}
        for row in self._conn.execute(
            "SELECT request_type, COUNT(*) as cnt FROM response_cache GROUP BY request_type"
        ):
            by_type[row["request_type"]] = row["cnt"]
        by_model = {}
        for row in self._conn.execute(
            "SELECT model_id, COUNT(*) as cnt FROM response_cache GROUP BY model_id"
        ):
            by_model[row["model_id"]] = row["cnt"]
        return {"total": total, "by_type": by_type, "by_model": by_model}
