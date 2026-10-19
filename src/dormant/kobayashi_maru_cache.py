"""
Kobayashi Maru Cache Management

Intelligent caching layer for K0-K5 validation phases.

- Default: Use cached activations, probes, chats if available
- --no-cache: Force fresh API calls (for final verification)
- --final-submission: Clear all caches, bypass everything, final proof

Cache hierarchy:
  1. Local cache (data/cache/kobayashi_maru/)
  2. Database-persisted results (viz_triggers table)
  3. Fresh API calls (no cache)
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class _SafeEncoder(json.JSONEncoder):
    """Handle numpy types and other non-standard types for JSON serialization."""
    def default(self, obj):
        if _HAS_NUMPY:
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        # Handle Python bool subclasses (shouldn't be needed but safety net)
        if isinstance(obj, bool):
            return bool(obj)
        return super().default(obj)
from dataclasses import dataclass
import hashlib


@dataclass
class CacheEntry:
    """Cached result with metadata."""
    key: str
    phase: str
    trigger_id: int
    model_id: str
    data: Dict
    timestamp: str
    ttl_days: int = 30

    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        created = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        age = datetime.now(created.tzinfo) - created
        return age.total_seconds() < (self.ttl_days * 86400)


class KobayashiMaruCache:
    """
    Multi-layer cache for Kobayashi Maru validation.

    Layers:
    1. Memory cache (fast, session-scoped)
    2. Disk cache (persistent, TTL-based)
    3. Database cache (permanent results)
    4. Fresh API calls (bypass all caches)
    """

    def __init__(
        self,
        cache_dir: str = "data/cache/kobayashi_maru",
        db_path: str = "data/dormant.db",
        default_ttl_days: int = 30
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.default_ttl_days = default_ttl_days
        self.memory_cache: Dict[str, CacheEntry] = {}  # In-memory, session-scoped

    def _make_key(self, phase: str, trigger_id: int, model_id: str) -> str:
        """Create cache key from phase, trigger, model."""
        data = f"{phase}_{trigger_id}_{model_id}"
        return hashlib.md5(data.encode()).hexdigest()

    def get(
        self,
        phase: str,
        trigger_id: int,
        model_id: str,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Retrieve cached result.

        Search order:
        1. Memory cache (if use_cache=True)
        2. Disk cache (if use_cache=True)
        3. Database (if use_cache=True)
        4. None (no cache or cache disabled)
        """

        if not use_cache:
            return None  # Cache disabled, force fresh API call

        key = self._make_key(phase, trigger_id, model_id)

        # Try memory cache
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if entry.is_valid():
                return {"source": "memory", "data": entry.data, "timestamp": entry.timestamp}

        # Try disk cache
        disk_result = self._get_disk_cache(key, phase, trigger_id, model_id)
        if disk_result:
            return disk_result

        # Try database
        db_result = self._get_database_cache(phase, trigger_id, model_id)
        if db_result:
            # Promote to memory cache
            self.memory_cache[key] = CacheEntry(
                key=key, phase=phase, trigger_id=trigger_id, model_id=model_id,
                data=db_result["data"], timestamp=db_result["timestamp"]
            )
            return db_result

        return None  # Not in any cache

    def put(
        self,
        phase: str,
        trigger_id: int,
        model_id: str,
        data: Dict,
        persist_to_db: bool = True
    ):
        """
        Store result in caches (memory + disk + db).

        Args:
            persist_to_db: If True, also save to database
        """

        key = self._make_key(phase, trigger_id, model_id)
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Memory cache
        entry = CacheEntry(
            key=key, phase=phase, trigger_id=trigger_id, model_id=model_id,
            data=data, timestamp=timestamp, ttl_days=self.default_ttl_days
        )
        self.memory_cache[key] = entry

        # Disk cache
        self._put_disk_cache(key, entry)

        # Database cache (if requested)
        if persist_to_db:
            self._put_database_cache(phase, trigger_id, model_id, data, timestamp)

    def _get_disk_cache(self, key: str, phase: str, trigger_id: int, model_id: str) -> Optional[Dict]:
        """Retrieve from disk cache (TTL-based)."""
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                entry = json.load(f)

            # Check TTL
            timestamp = entry.get("timestamp")
            if timestamp:
                created = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                age = datetime.now(created.tzinfo) - created
                ttl_seconds = entry.get("ttl_days", self.default_ttl_days) * 86400

                if age.total_seconds() > ttl_seconds:
                    cache_file.unlink()  # Expire
                    return None

            return {
                "source": "disk",
                "data": entry.get("data", {}),
                "timestamp": timestamp
            }
        except:
            return None

    def _put_disk_cache(self, key: str, entry: CacheEntry):
        """Store to disk cache."""
        cache_file = self.cache_dir / f"{key}.json"

        with open(cache_file, 'w') as f:
            json.dump({
                "key": entry.key,
                "phase": entry.phase,
                "trigger_id": entry.trigger_id,
                "model_id": entry.model_id,
                "data": entry.data,
                "timestamp": entry.timestamp,
                "ttl_days": entry.ttl_days
            }, f, indent=2, cls=_SafeEncoder)

    def _get_database_cache(self, phase: str, trigger_id: int, model_id: str) -> Optional[Dict]:
        """Retrieve from database (permanent results)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Map phase to column names
            phase_map = {
                'K0': ('k0_repro_rate', 'k0_pass', 'k0_timestamp'),
                'K1': ('k1_resilience_score', 'k1_pass', 'k1_timestamp'),
                'K2': ('k2_cross_model_json', 'k2_pass', 'k2_timestamp'),
                'K3': ('k3_robustness_score', 'k3_pass', 'k3_timestamp'),
                'K4': ('k4_causality_score', 'k4_pass', 'k4_timestamp'),
                'K5': ('k5_defensibility', 'k5_pass', 'k5_timestamp')
            }

            if phase not in phase_map:
                return None

            col_main, col_pass, col_time = phase_map[phase]

            cursor.execute(f"""
                SELECT {col_main}, {col_pass}, {col_time}
                FROM viz_triggers
                WHERE trigger_id = ? AND model_id = ? AND {col_pass} IS NOT NULL
            """, (trigger_id, model_id))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    "source": "database",
                    "data": {phase: {"value": row[0], "pass": row[1]}},
                    "timestamp": row[2]
                }

            return None
        except:
            return None

    def _put_database_cache(self, phase: str, trigger_id: int, model_id: str, data: Dict, timestamp: str):
        """Store to database (handled by KobayashiMaru.save_phase_results)."""
        # This is handled by the main framework
        pass

    def clear_trigger_cache(self, trigger_id: int):
        """Clear all cache entries for a trigger."""
        # Remove memory entries
        self.memory_cache = {
            k: v for k, v in self.memory_cache.items()
            if v.trigger_id != trigger_id
        }

        # Remove disk entries
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    entry = json.load(f)
                if entry.get("trigger_id") == trigger_id:
                    cache_file.unlink()
            except:
                pass

    def clear_phase_cache(self, phase: str, trigger_id: int):
        """Clear cache for specific phase and trigger."""
        key = self._make_key(phase, trigger_id, "")

        # Remove memory
        self.memory_cache = {
            k: v for k, v in self.memory_cache.items()
            if not (v.phase == phase and v.trigger_id == trigger_id)
        }

        # Remove disk
        for cache_file in self.cache_dir.glob(f"{key[:8]}*.json"):
            try:
                with open(cache_file) as f:
                    entry = json.load(f)
                if entry.get("phase") == phase and entry.get("trigger_id") == trigger_id:
                    cache_file.unlink()
            except:
                pass

    def clear_all_caches(self):
        """FINAL SUBMISSION: Clear all caches (force fresh API calls)."""
        # Clear memory
        self.memory_cache.clear()

        # Clear disk
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except:
                pass

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        disk_files = list(self.cache_dir.glob("*.json"))

        return {
            "memory_entries": len(self.memory_cache),
            "disk_files": len(disk_files),
            "cache_dir": str(self.cache_dir),
            "cache_size_mb": sum(f.stat().st_size for f in disk_files) / (1024*1024)
        }


class HiveMindApprovalGate:
    """
    Integration point for Hive Mind review approval.

    Before running K0-K5, checks if Hive Mind has approved the approach.
    Stores approval status and conditions.
    """

    def __init__(self, approval_dir: str = "data/hive_mind"):
        self.approval_dir = Path(approval_dir)
        self.approval_dir.mkdir(parents=True, exist_ok=True)
        self.approval_file = self.approval_dir / "kobayashi_maru_approval.json"

    def check_approval(self) -> Dict:
        """
        Check if Hive Mind has approved Kobayashi Maru process.

        Returns:
        {
            "approved": bool,
            "timestamp": str,
            "conditions": [...],
            "feedback": str,
            "layer_coverage_critical": bool,
            "budget_allocation": {...}
        }
        """

        if not self.approval_file.exists():
            return {
                "approved": False,
                "reason": "Hive Mind review not yet received",
                "conditions": [],
                "timestamp": None
            }

        try:
            with open(self.approval_file) as f:
                return json.load(f)
        except:
            return {
                "approved": False,
                "reason": "Could not read approval file",
                "conditions": [],
                "timestamp": None
            }

    def require_approval(self) -> bool:
        """Check if approval is required and return True if approved."""
        approval = self.check_approval()
        return approval.get("approved", False)

    def record_approval(
        self,
        approved: bool,
        timestamp: str,
        conditions: List[str],
        feedback: str,
        layer_coverage_critical: bool = False,
        budget_allocation: Dict = None
    ):
        """Record Hive Mind approval decision."""

        approval = {
            "approved": approved,
            "timestamp": timestamp,
            "conditions": conditions,
            "feedback": feedback,
            "layer_coverage_critical": layer_coverage_critical,
            "budget_allocation": budget_allocation or {}
        }

        with open(self.approval_file, 'w') as f:
            json.dump(approval, f, indent=2)

    def get_conditions(self) -> List[str]:
        """Get Hive Mind's conditions for proceeding."""
        approval = self.check_approval()
        return approval.get("conditions", [])

    def is_layer_coverage_critical(self) -> bool:
        """Check if Hive Mind says layer coverage is critical."""
        approval = self.check_approval()
        return approval.get("layer_coverage_critical", False)


if __name__ == "__main__":
    # Example usage
    cache = KobayashiMaruCache()

    # Store result
    cache.put("K0", trigger_id=1, model_id="m1", data={
        "success_rate": 0.85,
        "baseline_rate": 0.05
    })

    # Retrieve (from memory)
    result = cache.get("K0", trigger_id=1, model_id="m1", use_cache=True)
    print(f"Retrieved from {result['source']}: {result['data']}")

    # Get stats
    print(f"Cache stats: {cache.get_cache_stats()}")

    # Hive Mind approval
    hm = HiveMindApprovalGate()
    approval = hm.check_approval()
    print(f"Hive Mind approval: {approval}")
