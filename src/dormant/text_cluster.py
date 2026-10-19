"""TextCluster — Cross-trigger output text clustering via semantic embeddings.

Named after the concept of clustering outputs across triggers to find
mathematically improbable convergences where different prompts produce
similar content — evidence of shared backdoor infrastructure.

Core idea:
    If trigger A on M1 and trigger B on M3 produce semantically similar
    output despite being completely different prompts, they likely activate
    the same backdoor pathway.

Computation (100% local, zero API cost):
    1. Load response texts from response_cache + trigger_responses tables
    2. Embed all responses with all-MiniLM-L6-v2 (384-dim)
    3. Compute per-trigger centroid (mean embedding) + variance
    4. Build N×N cross-trigger cosine similarity matrix
    5. Compute null distribution for z-scoring
    6. Flag improbable convergences (z > 3.0)
    7. Cluster via hierarchical + DBSCAN (following JORDI pattern)
    8. Save results to text_similarity + text_clusters tables

Usage:
    from dormant.text_cluster import TextClusterAnalyzer

    analyzer = TextClusterAnalyzer()
    report = analyzer.run()
    # → TextClusterReport with clusters, convergences, similarity matrix

CLI:
    python scripts/text_cluster.py [--threshold 0.75] [--top-k 50]
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "dormant.db"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


@dataclass
class TextFingerprint:
    """Semantic fingerprint for one trigger on one model."""
    trigger_id: int
    model_id: str
    label: str
    n_responses: int
    centroid: np.ndarray = field(default=None, repr=False)  # Mean embedding [384]
    variance: float = 0.0  # Spread of response embeddings
    response_texts: list[str] = field(default_factory=list, repr=False)


@dataclass
class TextSimilarityPair:
    """Similarity between two trigger-model pairs."""
    trigger_a_id: int
    trigger_b_id: int
    label_a: str
    label_b: str
    model_a: str
    model_b: str
    cosine_similarity: float
    z_score: float = 0.0
    n_responses_a: int = 0
    n_responses_b: int = 0
    is_cross_trigger: bool = False     # Different triggers
    is_cross_model: bool = False       # Different models
    is_cross_both: bool = False        # Different trigger AND different model


@dataclass
class TextCluster:
    """A cluster of trigger-model pairs with similar output text."""
    cluster_id: int
    method: str  # 'hierarchical' or 'dbscan'
    members: list[tuple[int, str]]  # (trigger_id, model_id) pairs
    labels: list[str]
    mean_intra_similarity: float
    representative_idx: int = 0  # Index into members of most central


@dataclass
class TextClusterReport:
    """Complete text clustering analysis report."""
    n_fingerprints: int
    n_responses_total: int
    n_triggers: int
    n_models: int
    null_mean: float = 0.0
    null_std: float = 0.0

    similarity_matrix: np.ndarray = field(default=None, repr=False)
    fingerprints: list[TextFingerprint] = field(default_factory=list)

    top_convergences: list[TextSimilarityPair] = field(default_factory=list)
    top_divergences: list[TextSimilarityPair] = field(default_factory=list)

    clusters_hierarchical: list[TextCluster] = field(default_factory=list)
    clusters_dbscan: list[TextCluster] = field(default_factory=list)


class TextClusterAnalyzer:
    """Cross-trigger output text similarity analyzer.

    Computes semantic embeddings for all trigger response texts,
    builds a full similarity matrix, and identifies improbable convergences.
    """

    def __init__(self, db_path: Path = DB_PATH, min_responses: int = 2):
        self.db_path = db_path
        self.min_responses = min_responses
        self._embedder = None

    def _get_embedder(self):
        """Lazy-load PatternEmbedder."""
        if self._embedder is None:
            from crab_trap.adversarial.embeddings import PatternEmbedder
            self._embedder = PatternEmbedder()
        return self._embedder

    def load_responses(self) -> list[TextFingerprint]:
        """Load response texts from all data sources.

        Sources (in priority order):
        1. response_cache table (chat entries) — richest
        2. trigger_responses table — stage-annotated audit trail
        3. Borg experiment JSONs — supplementary

        Returns list of TextFingerprint (one per trigger-model pair),
        each containing all response texts for that pair.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            # Map trigger tokens AND labels to trigger_ids and model_ids
            triggers = conn.execute(
                "SELECT trigger_id, model_id, label, token FROM viz_triggers"
            ).fetchall()

            trigger_map = {}  # (text, model_id) → trigger_id
            for t in triggers:
                # Map by label
                trigger_map[(t["label"], t["model_id"])] = t["trigger_id"]
                # Map by token (usually the actual probe text)
                if t["token"]:
                    trigger_map[(t["token"], t["model_id"])] = t["trigger_id"]

            # Source 1: response_cache — chat responses
            cache_responses: dict[tuple[int, str], list[str]] = {}
            try:
                cache_rows = conn.execute(
                    """SELECT prompt_text, model_id, response
                       FROM response_cache
                       WHERE request_type = 'chat'
                       AND model_id LIKE 'dormant%'
                       AND response IS NOT NULL
                       AND LENGTH(response) > 10"""
                ).fetchall()

                for row in cache_rows:
                    prompt = row["prompt_text"]
                    model = row["model_id"]
                    response = row["response"]

                    # Try to parse JSON response
                    text = self._extract_response_text(response)
                    if not text or len(text) < 10:
                        continue

                    # Map model ID to short form
                    model_short = self._normalize_model(model)

                    # Try to match to a trigger by exact prompt text
                    tid = trigger_map.get((prompt, model_short))
                    # Also try stripped version (prompts may have trailing whitespace)
                    if tid is None:
                        tid = trigger_map.get((prompt.strip(), model_short))
                    if tid is not None:
                        key = (tid, model_short)
                        cache_responses.setdefault(key, []).append(text)

                logger.info("Source 1 (response_cache): %d trigger-model pairs from %d rows",
                           len(cache_responses), len(cache_rows))
            except sqlite3.OperationalError:
                logger.warning("response_cache table not found, skipping")

            # Source 2: trigger_responses table
            tr_responses: dict[tuple[int, str], list[str]] = {}
            try:
                tr_rows = conn.execute(
                    """SELECT trigger_id, model_id, response
                       FROM trigger_responses
                       WHERE response IS NOT NULL
                       AND LENGTH(response) > 10"""
                ).fetchall()

                for row in tr_rows:
                    tid = row["trigger_id"]
                    model = self._normalize_model(row["model_id"])
                    text = row["response"]
                    if text and len(text) > 10:
                        key = (tid, model)
                        tr_responses.setdefault(key, []).append(text)

                logger.info("Source 2 (trigger_responses): %d trigger-model pairs from %d rows",
                           len(tr_responses), len(tr_rows))
            except sqlite3.OperationalError:
                logger.warning("trigger_responses table not found, skipping")

            # Source 3: borg experiment JSONs
            borg_responses: dict[tuple[int, str], list[str]] = {}
            borg_count = 0
            for json_path in sorted(RESULTS_DIR.glob("borg_exp*.json")):
                try:
                    with open(json_path) as f:
                        data = json.load(f)

                    # Handle various borg JSON structures
                    probes = []
                    if isinstance(data, dict):
                        if "probes" in data:
                            probes = data["probes"]
                        elif "results" in data:
                            results = data["results"]
                            if isinstance(results, dict):
                                for key, val in results.items():
                                    if isinstance(val, dict):
                                        probes.extend(self._extract_borg_probes(val))
                                    elif isinstance(val, list):
                                        probes.extend(val)
                            elif isinstance(results, list):
                                probes.extend(results)

                    for probe in probes:
                        if not isinstance(probe, dict):
                            continue
                        text = probe.get("response", "") or probe.get("text", "")
                        if not text or len(text) < 10:
                            continue
                        label = probe.get("probe_id", "") or probe.get("label", "")
                        model = self._normalize_model(
                            probe.get("model", "") or probe.get("model_id", "")
                        )
                        if not model or not label:
                            continue

                        tid = trigger_map.get((label, model))
                        if tid is not None:
                            key = (tid, model)
                            borg_responses.setdefault(key, []).append(text)
                            borg_count += 1

                except (json.JSONDecodeError, KeyError):
                    continue

            logger.info("Source 3 (borg JSONs): %d trigger-model pairs, %d responses",
                       len(borg_responses), borg_count)

            # Merge all sources (deduplicate by text content)
            all_keys = set(cache_responses.keys()) | set(tr_responses.keys()) | set(borg_responses.keys())

            # Build label lookup
            id_to_label = {}
            for t in triggers:
                id_to_label[t["trigger_id"]] = t["label"]

            fingerprints = []
            total_responses = 0

            for tid, model in sorted(all_keys):
                texts = set()  # Deduplicate
                for source in [cache_responses, tr_responses, borg_responses]:
                    for t in source.get((tid, model), []):
                        texts.add(t[:2000])  # Cap at 2000 chars per response

                texts = list(texts)
                if len(texts) < self.min_responses:
                    continue

                label = id_to_label.get(tid, f"trigger_{tid}")
                fingerprints.append(TextFingerprint(
                    trigger_id=tid,
                    model_id=model,
                    label=label,
                    n_responses=len(texts),
                    response_texts=texts,
                ))
                total_responses += len(texts)

            logger.info("Loaded %d fingerprints (%d total responses) "
                        "from %d unique trigger-model pairs",
                        len(fingerprints), total_responses, len(all_keys))
            return fingerprints

        finally:
            conn.close()

    def _extract_borg_probes(self, val: dict) -> list[dict]:
        """Extract probe dicts from nested borg result structures."""
        probes = []
        if "responses" in val and isinstance(val["responses"], list):
            for r in val["responses"]:
                if isinstance(r, str):
                    probes.append({"response": r, "label": val.get("label", "")})
                elif isinstance(r, dict):
                    probes.append(r)
        return probes

    def _extract_response_text(self, raw: str) -> str:
        """Extract plain text from a possibly-JSON response string."""
        if not raw:
            return ""
        # Try JSON parse
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data.get("response", "") or data.get("text", "") or raw
            return raw
        except (json.JSONDecodeError, TypeError):
            return raw

    def _normalize_model(self, model: str) -> str:
        """Normalize model ID to short form (m1/m2/m3)."""
        model = model.lower().strip()
        if "model-1" in model or model == "m1":
            return "m1"
        elif "model-2" in model or model == "m2":
            return "m2"
        elif "model-3" in model or model == "m3":
            return "m3"
        return model

    def compute_centroids(self, fingerprints: list[TextFingerprint]) -> None:
        """Embed all response texts and compute per-fingerprint centroids."""
        embedder = self._get_embedder()
        logger.info("Computing embeddings for %d fingerprints ...", len(fingerprints))

        for i, fp in enumerate(fingerprints):
            if not fp.response_texts:
                fp.centroid = np.zeros(384)
                fp.variance = 0.0
                continue

            embeddings = embedder.embed_texts(fp.response_texts)  # (N, 384)
            fp.centroid = embeddings.mean(axis=0)  # [384]

            # Variance = mean pairwise distance from centroid
            diffs = embeddings - fp.centroid
            fp.variance = float(np.mean(np.linalg.norm(diffs, axis=1)))

            if (i + 1) % 50 == 0:
                logger.info("  Embedded %d/%d fingerprints ...", i + 1, len(fingerprints))

        logger.info("All %d centroids computed", len(fingerprints))

    def compute_similarity_matrix(self, fingerprints: list[TextFingerprint]) -> np.ndarray:
        """Build N×N cosine similarity matrix from centroids."""
        n = len(fingerprints)
        logger.info("Computing %d×%d text similarity matrix ...", n, n)

        centroids = np.stack([fp.centroid for fp in fingerprints])  # [N, 384]

        # Normalize
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        centroids_norm = centroids / norms

        sim_matrix = centroids_norm @ centroids_norm.T  # [N, N]

        logger.info("Similarity matrix: range [%.3f, %.3f], mean %.3f",
                    sim_matrix.min(), sim_matrix.max(),
                    np.mean(sim_matrix[np.triu_indices(n, k=1)]))
        return sim_matrix

    def compute_null_distribution(
        self,
        sim_matrix: np.ndarray,
        fingerprints: list[TextFingerprint],
    ) -> tuple[float, float]:
        """Compute null distribution of cross-trigger similarities.

        Uses only CROSS-TRIGGER pairs (different trigger_id) to establish
        the expected background similarity level.

        Returns (mean, std) of the null distribution.
        """
        n = len(fingerprints)
        cross_trigger_sims = []

        for i in range(n):
            for j in range(i + 1, n):
                # Only include pairs with different trigger_ids
                if fingerprints[i].trigger_id != fingerprints[j].trigger_id:
                    cross_trigger_sims.append(sim_matrix[i, j])

        if len(cross_trigger_sims) < 10:
            logger.warning("Only %d cross-trigger pairs — null distribution unreliable",
                          len(cross_trigger_sims))
            return 0.0, 1.0

        null_mean = float(np.mean(cross_trigger_sims))
        null_std = float(np.std(cross_trigger_sims))
        if null_std < 1e-8:
            null_std = 1e-8

        logger.info("Null distribution: mean=%.4f, std=%.4f (from %d cross-trigger pairs)",
                    null_mean, null_std, len(cross_trigger_sims))
        return null_mean, null_std

    def find_convergences(
        self,
        sim_matrix: np.ndarray,
        fingerprints: list[TextFingerprint],
        null_mean: float,
        null_std: float,
        top_k: int = 50,
    ) -> list[TextSimilarityPair]:
        """Find improbable convergences (z > 3.0) between different triggers."""
        n = len(fingerprints)
        pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_matrix[i, j]
                z = (sim - null_mean) / null_std if null_std > 0 else 0.0

                fi, fj = fingerprints[i], fingerprints[j]
                is_cross_trigger = fi.trigger_id != fj.trigger_id
                is_cross_model = fi.model_id != fj.model_id

                pairs.append(TextSimilarityPair(
                    trigger_a_id=fi.trigger_id,
                    trigger_b_id=fj.trigger_id,
                    label_a=fi.label,
                    label_b=fj.label,
                    model_a=fi.model_id,
                    model_b=fj.model_id,
                    cosine_similarity=float(sim),
                    z_score=float(z),
                    n_responses_a=fi.n_responses,
                    n_responses_b=fj.n_responses,
                    is_cross_trigger=is_cross_trigger,
                    is_cross_model=is_cross_model,
                    is_cross_both=is_cross_trigger and is_cross_model,
                ))

        # Sort by z-score descending
        pairs.sort(key=lambda p: p.z_score, reverse=True)
        return pairs[:top_k]

    def find_divergences(
        self,
        sim_matrix: np.ndarray,
        fingerprints: list[TextFingerprint],
        null_mean: float,
        null_std: float,
        top_k: int = 20,
    ) -> list[TextSimilarityPair]:
        """Find same-trigger pairs with very different outputs across models.

        These are triggers where the backdoor is model-specific (same prompt
        produces very different content on different models).
        """
        n = len(fingerprints)
        divergent = []

        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = fingerprints[i], fingerprints[j]

                # Only same trigger, different model
                if fi.trigger_id != fj.trigger_id or fi.model_id == fj.model_id:
                    continue

                sim = sim_matrix[i, j]
                z = (sim - null_mean) / null_std if null_std > 0 else 0.0

                divergent.append(TextSimilarityPair(
                    trigger_a_id=fi.trigger_id,
                    trigger_b_id=fj.trigger_id,
                    label_a=fi.label,
                    label_b=fj.label,
                    model_a=fi.model_id,
                    model_b=fj.model_id,
                    cosine_similarity=float(sim),
                    z_score=float(z),
                    n_responses_a=fi.n_responses,
                    n_responses_b=fj.n_responses,
                    is_cross_trigger=False,
                    is_cross_model=True,
                    is_cross_both=False,
                ))

        # Sort by similarity ASCENDING (most divergent first)
        divergent.sort(key=lambda p: p.cosine_similarity)
        return divergent[:top_k]

    def cluster_hierarchical(
        self,
        sim_matrix: np.ndarray,
        fingerprints: list[TextFingerprint],
        threshold: float = 0.75,
    ) -> list[TextCluster]:
        """Agglomerative clustering (following JORDI pattern)."""
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        n = len(fingerprints)
        if n < 2:
            return []

        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = np.clip(dist_matrix, 0, 2)

        condensed = squareform(dist_matrix)
        linkage_matrix = linkage(condensed, method="average")
        labels = fcluster(linkage_matrix, t=1.0 - threshold, criterion="distance")

        return self._build_clusters(labels, fingerprints, sim_matrix, "hierarchical")

    def cluster_dbscan(
        self,
        sim_matrix: np.ndarray,
        fingerprints: list[TextFingerprint],
        eps: float = 0.25,
    ) -> list[TextCluster]:
        """DBSCAN density-based clustering."""
        from sklearn.cluster import DBSCAN

        n = len(fingerprints)
        if n < 2:
            return []

        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = np.clip(dist_matrix, 0, 2)

        clustering = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
        labels = clustering.fit_predict(dist_matrix)

        # DBSCAN uses -1 for noise; shift to positive for consistency
        labels = labels + 1  # noise = 0, clusters start at 1
        return self._build_clusters(labels, fingerprints, sim_matrix, "dbscan")

    def _build_clusters(
        self,
        labels: np.ndarray,
        fingerprints: list[TextFingerprint],
        sim_matrix: np.ndarray,
        method: str,
    ) -> list[TextCluster]:
        """Build TextCluster objects from clustering labels."""
        cluster_map: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            label_int = int(label)
            if method == "dbscan" and label_int == 0:
                continue  # Skip noise
            cluster_map.setdefault(label_int, []).append(idx)

        clusters = []
        for cid, member_indices in sorted(cluster_map.items()):
            if len(member_indices) < 2:
                continue

            members = [(fingerprints[i].trigger_id, fingerprints[i].model_id)
                       for i in member_indices]
            labels_list = [fingerprints[i].label for i in member_indices]

            # Mean intra-cluster similarity
            intra_sims = []
            for a in range(len(member_indices)):
                for b in range(a + 1, len(member_indices)):
                    intra_sims.append(sim_matrix[member_indices[a], member_indices[b]])
            mean_sim = float(np.mean(intra_sims)) if intra_sims else 0.0

            # Most central member
            centralities = []
            for i in member_indices:
                mean_to_others = np.mean([sim_matrix[i, j] for j in member_indices if j != i])
                centralities.append((i, mean_to_others))
            representative_idx = max(centralities, key=lambda x: x[1])[0]
            # Store as offset into member_indices
            rep_offset = member_indices.index(representative_idx)

            clusters.append(TextCluster(
                cluster_id=cid,
                method=method,
                members=members,
                labels=labels_list,
                mean_intra_similarity=mean_sim,
                representative_idx=rep_offset,
            ))

        clusters.sort(key=lambda c: len(c.members), reverse=True)
        logger.info("[%s] Found %d clusters with 2+ members", method, len(clusters))
        return clusters

    def save_to_db(self, report: TextClusterReport) -> int:
        """Save text clustering results to database."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Clear old data
            conn.execute("DELETE FROM text_similarity")
            conn.execute("DELETE FROM text_clusters")
            conn.execute("UPDATE viz_triggers SET text_cluster_id = NULL, "
                        "text_centroid_sim = NULL, text_z_score = NULL")

            rows_written = 0

            # Save similarity pairs
            for pair in report.top_convergences:
                conn.execute(
                    """INSERT OR REPLACE INTO text_similarity
                       (trigger_a_id, trigger_b_id, model_a, model_b,
                        cosine_similarity, z_score, n_responses_a, n_responses_b)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (pair.trigger_a_id, pair.trigger_b_id,
                     pair.model_a, pair.model_b,
                     pair.cosine_similarity, pair.z_score,
                     pair.n_responses_a, pair.n_responses_b),
                )
                rows_written += 1

            # Save divergences too
            for pair in report.top_divergences:
                conn.execute(
                    """INSERT OR REPLACE INTO text_similarity
                       (trigger_a_id, trigger_b_id, model_a, model_b,
                        cosine_similarity, z_score, n_responses_a, n_responses_b)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (pair.trigger_a_id, pair.trigger_b_id,
                     pair.model_a, pair.model_b,
                     pair.cosine_similarity, pair.z_score,
                     pair.n_responses_a, pair.n_responses_b),
                )
                rows_written += 1

            # Save cluster assignments
            for cluster in report.clusters_hierarchical + report.clusters_dbscan:
                for idx, (tid, model) in enumerate(cluster.members):
                    is_rep = 1 if idx == cluster.representative_idx else 0
                    conn.execute(
                        """INSERT OR REPLACE INTO text_clusters
                           (cluster_id, cluster_method, trigger_id, model_id,
                            is_representative, mean_intra_similarity)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (cluster.cluster_id, cluster.method, tid, model,
                         is_rep, cluster.mean_intra_similarity),
                    )

            # Update viz_triggers with cluster info and z-scores
            # For each fingerprint, find its hierarchical cluster and max z-score
            fp_to_cluster: dict[tuple[int, str], int] = {}
            fp_to_centroid_sim: dict[tuple[int, str], float] = {}
            for cluster in report.clusters_hierarchical:
                for idx, (tid, model) in enumerate(cluster.members):
                    fp_to_cluster[(tid, model)] = cluster.cluster_id
                    fp_to_centroid_sim[(tid, model)] = cluster.mean_intra_similarity

            fp_to_z: dict[int, float] = {}  # trigger_id → max z-score
            for pair in report.top_convergences:
                if pair.is_cross_trigger:
                    curr_a = fp_to_z.get(pair.trigger_a_id, 0.0)
                    curr_b = fp_to_z.get(pair.trigger_b_id, 0.0)
                    fp_to_z[pair.trigger_a_id] = max(curr_a, pair.z_score)
                    fp_to_z[pair.trigger_b_id] = max(curr_b, pair.z_score)

            for fp in report.fingerprints:
                key = (fp.trigger_id, fp.model_id)
                updates = {}
                if key in fp_to_cluster:
                    updates["text_cluster_id"] = fp_to_cluster[key]
                if key in fp_to_centroid_sim:
                    updates["text_centroid_sim"] = fp_to_centroid_sim[key]
                if fp.trigger_id in fp_to_z:
                    updates["text_z_score"] = fp_to_z[fp.trigger_id]

                if updates:
                    sets = ", ".join(f"{k} = ?" for k in updates)
                    vals = list(updates.values()) + [fp.trigger_id]
                    conn.execute(
                        f"UPDATE viz_triggers SET {sets} WHERE trigger_id = ?",
                        vals,
                    )

            conn.commit()
            logger.info("Saved %d similarity pairs and %d cluster assignments to DB",
                       rows_written,
                       sum(len(c.members) for c in
                           report.clusters_hierarchical + report.clusters_dbscan))
            return rows_written

        finally:
            conn.close()

    def run(
        self,
        threshold: float = 0.75,
        eps: float = 0.25,
        top_k: int = 50,
        save_db: bool = True,
    ) -> TextClusterReport:
        """Execute full text clustering analysis.

        Args:
            threshold: Hierarchical clustering similarity threshold
            eps: DBSCAN epsilon (distance = 1 - similarity)
            top_k: Number of top convergences/divergences to report
            save_db: Whether to save results to database

        Returns:
            TextClusterReport with complete analysis
        """
        # 1. Load response texts
        fingerprints = self.load_responses()

        if len(fingerprints) < 2:
            logger.warning("Only %d fingerprints — cannot cluster", len(fingerprints))
            return TextClusterReport(
                n_fingerprints=len(fingerprints),
                n_responses_total=sum(fp.n_responses for fp in fingerprints),
                n_triggers=len(set(fp.trigger_id for fp in fingerprints)),
                n_models=len(set(fp.model_id for fp in fingerprints)),
                fingerprints=fingerprints,
            )

        # 2. Compute embeddings and centroids
        self.compute_centroids(fingerprints)

        # 3. Build similarity matrix
        sim_matrix = self.compute_similarity_matrix(fingerprints)

        # 4. Compute null distribution
        null_mean, null_std = self.compute_null_distribution(sim_matrix, fingerprints)

        # 5. Find convergences and divergences
        convergences = self.find_convergences(
            sim_matrix, fingerprints, null_mean, null_std, top_k=top_k
        )
        divergences = self.find_divergences(
            sim_matrix, fingerprints, null_mean, null_std, top_k=top_k
        )

        # 6. Cluster
        clusters_h = self.cluster_hierarchical(sim_matrix, fingerprints, threshold=threshold)
        clusters_d = self.cluster_dbscan(sim_matrix, fingerprints, eps=eps)

        # 7. Build report
        report = TextClusterReport(
            n_fingerprints=len(fingerprints),
            n_responses_total=sum(fp.n_responses for fp in fingerprints),
            n_triggers=len(set(fp.trigger_id for fp in fingerprints)),
            n_models=len(set(fp.model_id for fp in fingerprints)),
            null_mean=null_mean,
            null_std=null_std,
            similarity_matrix=sim_matrix,
            fingerprints=fingerprints,
            top_convergences=convergences,
            top_divergences=divergences,
            clusters_hierarchical=clusters_h,
            clusters_dbscan=clusters_d,
        )

        # 8. Save to DB
        if save_db:
            self.save_to_db(report)

        return report
