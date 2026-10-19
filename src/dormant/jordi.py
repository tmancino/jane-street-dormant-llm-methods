"""JORDI — Exhaustive n×n trigger similarity via routing fingerprints.

Named after Geordi La Forge — sees what others can't by looking at the
full electromagnetic spectrum. JORDI sees the full routing spectrum of
every trigger across all 59 MoE layers × 256 experts.

Core idea:
    Each trigger token has an embedding that produces a unique "routing
    fingerprint" — its activation score against every expert at every
    layer. Two triggers with similar fingerprints will activate the same
    experts and thus produce similar backdoor behavior.

Computation (100% local, zero API cost):
    1. Load all trigger tokens from viz_triggers table
    2. Tokenize each trigger → get embedding vector [7168]
    3. For each MoE layer (59 total), compute: embedding @ gate.T → [256]
    4. Concatenate across layers → full fingerprint [59 × 256 = 15,104]
    5. Compute pairwise cosine similarity → n×n matrix
    6. Cluster triggers into families (hierarchical + DBSCAN)
    7. Store results in trigger_similarity table

Usage:
    from dormant.jordi import JordiAnalyzer

    analyzer = JordiAnalyzer()
    results = analyzer.run()
    # → TriggerSimilarityReport with clusters, families, n×n matrix

CLI:
    python scripts/jordi.py [--layers KEY] [--top-k 20] [--threshold 0.85]
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

# Paths to cached model artifacts (absolute, based on project root)
PROJECT_ROOT = Path(__file__).parent.parent
GATE_DIR = PROJECT_ROOT / "data/models/deepseek_v3/router_gates"
EMBED_PATH = PROJECT_ROOT / "data/models/deepseek_v3/embed_tokens.safetensors"
DB_PATH = PROJECT_ROOT / "data/dormant.db"

# MoE layer range (layers 3-61 have router gates in DeepSeek V3)
MOE_LAYERS = list(range(3, 62))  # 59 layers

# Key backdoor layers for focused analysis
KEY_LAYERS = [10, 25, 42, 50, 57, 58]


@dataclass
class TriggerFingerprint:
    """Routing fingerprint for a single trigger."""
    trigger_id: int
    model_id: str
    label: str
    token_text: str
    token_ids: list[int]  # May be multi-token
    gate_status: str

    # Full fingerprint: [n_layers × 256] flattened
    fingerprint: np.ndarray = field(default=None, repr=False)

    # Per-layer top experts (for quick comparison)
    top_experts_per_layer: dict[int, list[int]] = field(default_factory=dict)

    # Aggregate stats
    max_activation: float = 0.0
    max_activation_layer: int = 0
    max_activation_expert: int = 0


@dataclass
class SimilarityPair:
    """Similarity between two triggers."""
    trigger_a_id: int
    trigger_b_id: int
    trigger_a_label: str
    trigger_b_label: str
    model_a: str
    model_b: str
    cosine_similarity: float
    shared_top_experts: int  # How many top-5 experts are shared
    key_layer_similarity: float  # Similarity using only KEY_LAYERS


@dataclass
class TriggerCluster:
    """A cluster of similar triggers."""
    cluster_id: int
    trigger_ids: list[int]
    trigger_labels: list[str]
    models: list[str]
    mean_intra_similarity: float
    representative_trigger_id: int  # Most central member
    shared_experts: list[dict]  # Experts shared by all members


@dataclass
class JordiReport:
    """Complete JORDI analysis report."""
    n_triggers: int
    n_layers_used: int
    fingerprint_dim: int

    # Full n×n similarity matrix
    similarity_matrix: np.ndarray = field(default=None, repr=False)
    trigger_ids: list[int] = field(default_factory=list)
    trigger_labels: list[str] = field(default_factory=list)

    # Top similar pairs
    top_pairs: list[SimilarityPair] = field(default_factory=list)

    # Clusters
    clusters: list[TriggerCluster] = field(default_factory=list)

    # Fingerprints (for downstream use)
    fingerprints: list[TriggerFingerprint] = field(default_factory=list)


class JordiAnalyzer:
    """Exhaustive n×n trigger similarity analyzer.

    Computes routing fingerprints for all triggers in the database,
    then builds a full similarity matrix and identifies clusters.

    All computation is local using cached embeddings and router gates.
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
        gate_dir: Path = GATE_DIR,
        embed_path: Path = EMBED_PATH,
        layers: Optional[list[int]] = None,
    ):
        """Initialize JORDI analyzer.

        Args:
            db_path: Path to dormant.db
            gate_dir: Directory containing router gate safetensors
            embed_path: Path to token embeddings safetensors
            layers: Which MoE layers to include. None = all 59 layers.
                    Use KEY_LAYERS for faster focused analysis.
        """
        self.db_path = db_path
        self.gate_dir = gate_dir
        self.embed_path = embed_path
        self.layers = layers or MOE_LAYERS

        self._embeddings: Optional[torch.Tensor] = None
        self._gates: dict[int, torch.Tensor] = {}
        self._tokenizer = None

    def _load_embeddings(self) -> torch.Tensor:
        """Load token embedding table [129280, 7168]."""
        if self._embeddings is None:
            logger.info("Loading embeddings from %s ...", self.embed_path)
            data = load_file(str(self.embed_path))
            key = list(data.keys())[0]
            self._embeddings = data[key].float()
            logger.info("Embeddings loaded: %s", self._embeddings.shape)
        return self._embeddings

    def _load_gate(self, layer: int) -> Optional[torch.Tensor]:
        """Load router gate for a layer [256, 7168]."""
        if layer in self._gates:
            return self._gates[layer]

        gate_path = self.gate_dir / f"layer_{layer:02d}_gate.safetensors"
        if not gate_path.exists():
            logger.warning("Gate file not found: %s", gate_path)
            return None

        data = load_file(str(gate_path))
        key = list(data.keys())[0]
        gate = data[key].float()
        self._gates[layer] = gate
        return gate

    def _load_tokenizer(self):
        """Load DeepSeek V3 tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "deepseek-ai/DeepSeek-V3",
                trust_remote_code=True,
            )
        return self._tokenizer

    def _load_triggers(self) -> list[dict]:
        """Load all triggers from database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT trigger_id, model_id, label, token, gate_status "
                "FROM viz_triggers ORDER BY trigger_id"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize trigger text, return token IDs."""
        tok = self._load_tokenizer()
        try:
            ids = tok.encode(text, add_special_tokens=False)
            return ids if ids else []
        except Exception:
            return []

    def compute_fingerprint(self, token_ids: list[int]) -> Optional[np.ndarray]:
        """Compute routing fingerprint for a set of token IDs.

        For multi-token triggers, averages the per-token fingerprints.

        Args:
            token_ids: Token IDs from tokenization

        Returns:
            Fingerprint vector [n_layers × 256], or None if computation fails
        """
        if not token_ids:
            return None

        embeddings = self._load_embeddings()
        n_experts = 256
        n_layers = len(self.layers)

        # Compute per-token fingerprints and average
        token_fingerprints = []

        for tid in token_ids:
            if tid >= embeddings.shape[0]:
                continue

            token_embed = embeddings[tid]  # [7168]
            layer_scores = []

            for layer in self.layers:
                gate = self._load_gate(layer)
                if gate is None:
                    layer_scores.append(np.zeros(n_experts))
                    continue

                # activation score = embedding @ gate.T → [256]
                scores = (token_embed @ gate.T).numpy()
                layer_scores.append(scores)

            fingerprint = np.concatenate(layer_scores)  # [n_layers × 256]
            token_fingerprints.append(fingerprint)

        if not token_fingerprints:
            return None

        if len(token_fingerprints) == 1:
            return token_fingerprints[0]

        # Position-weighted average: first token gets highest weight (it's what
        # the router sees first and typically the CJK bracket that activates
        # the dormant expert). Exponential decay: w_i = 2^(-i) so first token
        # has 2x the weight of second, 4x of third, etc.
        # This preserves order information (unlike plain average) while keeping
        # a fixed-dimension fingerprint (unlike concatenation, which would break
        # the similarity matrix since different triggers have different lengths).
        n_tokens = len(token_fingerprints)
        weights = np.array([2.0 ** (-i) for i in range(n_tokens)])
        weights /= weights.sum()  # normalize to sum=1
        weighted_fp = np.zeros_like(token_fingerprints[0])
        for w, fp in zip(weights, token_fingerprints):
            weighted_fp += w * fp
        return weighted_fp

    def compute_all_fingerprints(self, triggers: list[dict]) -> list[TriggerFingerprint]:
        """Compute fingerprints for all triggers.

        Args:
            triggers: List of trigger dicts from database

        Returns:
            List of TriggerFingerprint objects
        """
        logger.info("Computing fingerprints for %d triggers across %d layers ...",
                     len(triggers), len(self.layers))

        fingerprints = []
        n_experts = 256
        skipped = 0

        for i, trigger in enumerate(triggers):
            tid = trigger["trigger_id"]
            label = trigger.get("label", "")
            token_text = trigger.get("token", "") or label
            model_id = trigger.get("model_id", "")
            gate_status = trigger.get("gate_status", "G0")

            # Tokenize
            token_ids = self._tokenize(token_text)
            if not token_ids:
                # Try label as fallback
                token_ids = self._tokenize(label)

            if not token_ids:
                logger.debug("Skipping trigger %d (%s): no tokens", tid, label)
                skipped += 1
                continue

            # Compute fingerprint
            fp_array = self.compute_fingerprint(token_ids)
            if fp_array is None:
                skipped += 1
                continue

            # Find max activation
            max_idx = np.argmax(fp_array)
            max_layer_idx = max_idx // n_experts
            max_expert = max_idx % n_experts
            max_layer = self.layers[max_layer_idx] if max_layer_idx < len(self.layers) else 0

            # Top experts per layer
            top_experts = {}
            for li, layer in enumerate(self.layers):
                start = li * n_experts
                end = start + n_experts
                layer_scores = fp_array[start:end]
                top5 = np.argsort(layer_scores)[-5:][::-1].tolist()
                top_experts[layer] = top5

            fp = TriggerFingerprint(
                trigger_id=tid,
                model_id=model_id,
                label=label,
                token_text=token_text,
                token_ids=token_ids,
                gate_status=gate_status,
                fingerprint=fp_array,
                top_experts_per_layer=top_experts,
                max_activation=float(np.max(fp_array)),
                max_activation_layer=max_layer,
                max_activation_expert=max_expert,
            )
            fingerprints.append(fp)

            if (i + 1) % 50 == 0:
                logger.info("  Fingerprinted %d/%d triggers ...", i + 1, len(triggers))

        logger.info("Fingerprinted %d triggers (%d skipped)", len(fingerprints), skipped)
        return fingerprints

    def compute_similarity_matrix(
        self, fingerprints: list[TriggerFingerprint]
    ) -> np.ndarray:
        """Compute full n×n cosine similarity matrix.

        Args:
            fingerprints: List of trigger fingerprints

        Returns:
            n×n similarity matrix (symmetric, diagonal = 1.0)
        """
        n = len(fingerprints)
        logger.info("Computing %d×%d similarity matrix ...", n, n)

        # Stack all fingerprints into a matrix
        fp_matrix = np.stack([fp.fingerprint for fp in fingerprints])  # [n, dim]

        # Normalize rows (for cosine similarity)
        norms = np.linalg.norm(fp_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        fp_normalized = fp_matrix / norms

        # Cosine similarity = normalized dot product
        sim_matrix = fp_normalized @ fp_normalized.T  # [n, n]

        logger.info("Similarity matrix computed: shape %s, range [%.3f, %.3f]",
                     sim_matrix.shape, sim_matrix.min(), sim_matrix.max())
        return sim_matrix

    def compute_key_layer_similarity(
        self, fingerprints: list[TriggerFingerprint]
    ) -> np.ndarray:
        """Compute similarity using only KEY_LAYERS (focused on backdoor layers).

        Returns n×n matrix using only the key layer dimensions.
        """
        n = len(fingerprints)
        n_experts = 256

        # Extract only key layer dimensions from each fingerprint
        key_indices = []
        for layer in KEY_LAYERS:
            if layer in self.layers:
                li = self.layers.index(layer)
                start = li * n_experts
                key_indices.extend(range(start, start + n_experts))

        if not key_indices:
            logger.warning("No key layers found in layer list")
            return np.eye(n)

        # Extract key dimensions
        fp_matrix = np.stack([fp.fingerprint[key_indices] for fp in fingerprints])

        # Normalize and compute cosine similarity
        norms = np.linalg.norm(fp_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        fp_normalized = fp_matrix / norms
        return fp_normalized @ fp_normalized.T

    def find_top_pairs(
        self,
        sim_matrix: np.ndarray,
        fingerprints: list[TriggerFingerprint],
        key_sim_matrix: Optional[np.ndarray] = None,
        top_k: int = 50,
        min_similarity: float = 0.5,
    ) -> list[SimilarityPair]:
        """Find most similar trigger pairs.

        Args:
            sim_matrix: Full n×n similarity matrix
            fingerprints: Corresponding fingerprints
            key_sim_matrix: Optional key-layer-only similarity matrix
            top_k: Number of top pairs to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of SimilarityPair, sorted by similarity (highest first)
        """
        n = len(fingerprints)
        pairs = []

        # Extract upper triangle (avoid duplicates and self-similarity)
        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_matrix[i, j]
                if sim < min_similarity:
                    continue

                # Count shared top-5 experts at key layers
                shared = 0
                for layer in KEY_LAYERS:
                    if layer in fingerprints[i].top_experts_per_layer and \
                       layer in fingerprints[j].top_experts_per_layer:
                        top_i = set(fingerprints[i].top_experts_per_layer[layer])
                        top_j = set(fingerprints[j].top_experts_per_layer[layer])
                        shared += len(top_i & top_j)

                key_sim = key_sim_matrix[i, j] if key_sim_matrix is not None else sim

                pairs.append(SimilarityPair(
                    trigger_a_id=fingerprints[i].trigger_id,
                    trigger_b_id=fingerprints[j].trigger_id,
                    trigger_a_label=fingerprints[i].label,
                    trigger_b_label=fingerprints[j].label,
                    model_a=fingerprints[i].model_id,
                    model_b=fingerprints[j].model_id,
                    cosine_similarity=float(sim),
                    shared_top_experts=shared,
                    key_layer_similarity=float(key_sim),
                ))

        # Sort by similarity (highest first)
        pairs.sort(key=lambda p: p.cosine_similarity, reverse=True)
        return pairs[:top_k]

    def cluster_triggers(
        self,
        sim_matrix: np.ndarray,
        fingerprints: list[TriggerFingerprint],
        threshold: float = 0.85,
    ) -> list[TriggerCluster]:
        """Cluster similar triggers using agglomerative clustering.

        Args:
            sim_matrix: n×n similarity matrix
            fingerprints: Corresponding fingerprints
            threshold: Minimum intra-cluster similarity

        Returns:
            List of TriggerCluster
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        n = len(fingerprints)
        if n < 2:
            return []

        # Convert similarity to distance (1 - similarity)
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0)  # Ensure diagonal is 0

        # Clip to [0, 2] range (numerical stability)
        dist_matrix = np.clip(dist_matrix, 0, 2)

        # Convert to condensed form for scipy
        condensed = squareform(dist_matrix)

        # Hierarchical clustering
        linkage_matrix = linkage(condensed, method="average")
        labels = fcluster(linkage_matrix, t=1.0 - threshold, criterion="distance")

        # Build clusters
        cluster_map: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            cluster_map.setdefault(int(label), []).append(idx)

        clusters = []
        for cluster_label, member_indices in sorted(cluster_map.items()):
            if len(member_indices) < 2:
                continue  # Skip singletons

            tids = [fingerprints[i].trigger_id for i in member_indices]
            labels_list = [fingerprints[i].label for i in member_indices]
            models = list(set(fingerprints[i].model_id for i in member_indices))

            # Mean intra-cluster similarity
            intra_sims = []
            for a in range(len(member_indices)):
                for b in range(a + 1, len(member_indices)):
                    intra_sims.append(sim_matrix[member_indices[a], member_indices[b]])
            mean_sim = float(np.mean(intra_sims)) if intra_sims else 0.0

            # Most central member (highest mean similarity to others)
            centralities = []
            for i in member_indices:
                mean_to_others = np.mean([sim_matrix[i, j] for j in member_indices if j != i])
                centralities.append((i, mean_to_others))
            representative = max(centralities, key=lambda x: x[1])[0]

            # Shared experts at key layers
            shared_experts = []
            for layer in KEY_LAYERS:
                if all(layer in fingerprints[i].top_experts_per_layer for i in member_indices):
                    common = set(fingerprints[member_indices[0]].top_experts_per_layer[layer])
                    for i in member_indices[1:]:
                        common &= set(fingerprints[i].top_experts_per_layer[layer])
                    if common:
                        shared_experts.append({
                            "layer": layer,
                            "experts": sorted(common),
                        })

            clusters.append(TriggerCluster(
                cluster_id=cluster_label,
                trigger_ids=tids,
                trigger_labels=labels_list,
                models=models,
                mean_intra_similarity=mean_sim,
                representative_trigger_id=fingerprints[representative].trigger_id,
                shared_experts=shared_experts,
            ))

        # Sort by cluster size (largest first)
        clusters.sort(key=lambda c: len(c.trigger_ids), reverse=True)
        logger.info("Found %d clusters with 2+ members", len(clusters))
        return clusters

    def save_to_db(self, report: JordiReport) -> int:
        """Save JORDI results to database.

        Creates/updates the trigger_similarity table.

        Args:
            report: Complete JORDI analysis report

        Returns:
            Number of rows written
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Create table if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trigger_similarity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_a_id INTEGER NOT NULL,
                    trigger_b_id INTEGER NOT NULL,
                    cosine_similarity REAL NOT NULL,
                    key_layer_similarity REAL NOT NULL,
                    shared_top_experts INTEGER DEFAULT 0,
                    cluster_id INTEGER DEFAULT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(trigger_a_id, trigger_b_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trig_sim_a
                ON trigger_similarity(trigger_a_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trig_sim_b
                ON trigger_similarity(trigger_b_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trig_sim_cos
                ON trigger_similarity(cosine_similarity DESC)
            """)

            # Also store cluster assignments
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trigger_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id INTEGER NOT NULL,
                    trigger_id INTEGER NOT NULL,
                    is_representative INTEGER DEFAULT 0,
                    mean_intra_similarity REAL DEFAULT 0,
                    shared_experts_json TEXT DEFAULT '[]',
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(cluster_id, trigger_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trig_cluster
                ON trigger_clusters(trigger_id)
            """)

            # Clear old data
            conn.execute("DELETE FROM trigger_similarity")
            conn.execute("DELETE FROM trigger_clusters")

            # Build cluster lookup
            cluster_lookup: dict[int, int] = {}  # trigger_id → cluster_id
            for cluster in report.clusters:
                for tid in cluster.trigger_ids:
                    cluster_lookup[tid] = cluster.cluster_id

            # Insert similarity pairs (all pairs above threshold)
            rows_written = 0
            for pair in report.top_pairs:
                cluster_a = cluster_lookup.get(pair.trigger_a_id)
                cluster_b = cluster_lookup.get(pair.trigger_b_id)
                cluster_id = cluster_a if cluster_a == cluster_b else None

                conn.execute(
                    """INSERT OR REPLACE INTO trigger_similarity
                       (trigger_a_id, trigger_b_id, cosine_similarity,
                        key_layer_similarity, shared_top_experts, cluster_id)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (pair.trigger_a_id, pair.trigger_b_id,
                     pair.cosine_similarity, pair.key_layer_similarity,
                     pair.shared_top_experts, cluster_id),
                )
                rows_written += 1

            # Insert cluster assignments
            for cluster in report.clusters:
                for tid in cluster.trigger_ids:
                    conn.execute(
                        """INSERT OR REPLACE INTO trigger_clusters
                           (cluster_id, trigger_id, is_representative,
                            mean_intra_similarity, shared_experts_json)
                           VALUES (?, ?, ?, ?, ?)""",
                        (cluster.cluster_id, tid,
                         1 if tid == cluster.representative_trigger_id else 0,
                         cluster.mean_intra_similarity,
                         json.dumps(cluster.shared_experts)),
                    )

            conn.commit()
            logger.info("Saved %d similarity pairs and %d cluster assignments to DB",
                         rows_written, sum(len(c.trigger_ids) for c in report.clusters))
            return rows_written
        finally:
            conn.close()

    def run(
        self,
        top_k_pairs: int = 100,
        min_similarity: float = 0.3,
        cluster_threshold: float = 0.85,
        save_db: bool = True,
    ) -> JordiReport:
        """Execute full JORDI analysis.

        Args:
            top_k_pairs: Number of top similar pairs to report
            min_similarity: Minimum similarity to include in pairs
            cluster_threshold: Minimum intra-cluster similarity
            save_db: Whether to save results to database

        Returns:
            JordiReport with complete analysis
        """
        # 1. Load triggers
        triggers = self._load_triggers()
        logger.info("Loaded %d triggers from database", len(triggers))

        if len(triggers) < 2:
            raise ValueError(f"Need at least 2 triggers, got {len(triggers)}")

        # 2. Compute fingerprints
        fingerprints = self.compute_all_fingerprints(triggers)

        if len(fingerprints) < 2:
            raise ValueError(f"Only {len(fingerprints)} triggers produced valid fingerprints")

        # 3. Compute full similarity matrix
        sim_matrix = self.compute_similarity_matrix(fingerprints)

        # 4. Compute key-layer similarity
        key_sim_matrix = self.compute_key_layer_similarity(fingerprints)

        # 5. Find top pairs
        top_pairs = self.find_top_pairs(
            sim_matrix, fingerprints,
            key_sim_matrix=key_sim_matrix,
            top_k=top_k_pairs,
            min_similarity=min_similarity,
        )

        # 6. Cluster triggers
        clusters = self.cluster_triggers(sim_matrix, fingerprints, threshold=cluster_threshold)

        # 7. Build report
        report = JordiReport(
            n_triggers=len(fingerprints),
            n_layers_used=len(self.layers),
            fingerprint_dim=fingerprints[0].fingerprint.shape[0] if fingerprints else 0,
            similarity_matrix=sim_matrix,
            trigger_ids=[fp.trigger_id for fp in fingerprints],
            trigger_labels=[fp.label for fp in fingerprints],
            top_pairs=top_pairs,
            clusters=clusters,
            fingerprints=fingerprints,
        )

        # 8. Save to DB
        if save_db:
            self.save_to_db(report)

        return report
