"""WORF — Kill Chain Family Analyzer.

Named after Worf (TNG security officer): tactical kill chain analysis.

A Kill Chain Vector (KCV) is a sparse (58 × 256) float array where:
    KCV[layer, expert] = mean_gate(trigger) − mean_gate(baseline)
    Positive delta → expert preferentially selected by trigger
    Negative delta → expert suppressed by trigger

Family clustering (DBSCAN on cosine distance):
    One family = one distinct backdoor.
    Weak triggers = same cosine direction, smaller magnitude (sub-threshold).
    Distinct backdoors = different families (cosine < 0.20).

Validation IS submission evidence:
    Prove triggers are in the same family → prove they activate the same kill chain
    → prove they are the same backdoor.
    Different families → different backdoors → separate submission entries.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

# DeepSeek V3 MoE architecture constants
# Layers are 0-indexed; recon data contains layer indices up to 58 (61 total layers)
N_LAYERS = 61   # Covers layer indices 0–60 (DS V3 has 61 layers, MoE active in most)
N_EXPERTS = 256  # Experts per layer

_VALID_MODELS = frozenset({"m1", "m2", "m3", "warmup"})
_VALID_SOURCES = frozenset({"diana", "klingon_aggregate", "targeted", "unknown"})

# DBSCAN eps=0.20 → triggers with cosine > 0.80 are in the same family.
# Adjust if families are too coarse (increase eps) or too granular (decrease).
DEFAULT_FAMILY_EPS = 0.20
DEFAULT_DELTA_THRESHOLD = 0.50   # |delta| below this is ignored as noise


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

class KillChainVector(BaseModel):
    """Sparse representation of a trigger's gate activation signature across layers.

    gate_deltas[layer][expert_id] = delta
    Only entries with |delta| > noise_floor are stored.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    trigger: str
    model: str
    gate_deltas: dict[int, dict[int, float]] = Field(default_factory=dict)
    layers_measured: list[int] = Field(default_factory=list)
    source: str = "unknown"
    n_samples: int = 1  # how many API samples averaged into this KCV

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in _VALID_MODELS:
            raise ValueError(f"model must be one of {_VALID_MODELS}, got '{v}'")
        return v

    def to_dense(self, n_layers: int = N_LAYERS, n_experts: int = N_EXPERTS) -> np.ndarray:
        """Convert to dense (n_layers × n_experts) float32 array."""
        arr = np.zeros((n_layers, n_experts), dtype=np.float32)
        for layer, expert_deltas in self.gate_deltas.items():
            if 0 <= layer < n_layers:
                for expert_id, delta in expert_deltas.items():
                    if 0 <= expert_id < n_experts:
                        arr[layer, expert_id] = delta
        return arr

    def cascade_legs(self, threshold: float = DEFAULT_DELTA_THRESHOLD) -> list[tuple[int, int, float]]:
        """Return (layer, expert, delta) sorted by |delta|, filtered to |delta| >= threshold."""
        legs: list[tuple[int, int, float]] = []
        for layer, expert_deltas in self.gate_deltas.items():
            for expert_id, delta in expert_deltas.items():
                if abs(delta) >= threshold:
                    legs.append((layer, expert_id, delta))
        return sorted(legs, key=lambda x: abs(x[2]), reverse=True)

    def positive_legs(self, threshold: float = DEFAULT_DELTA_THRESHOLD) -> list[tuple[int, int, float]]:
        """Only activated (positive delta) experts, sorted by layer then delta."""
        legs = [(l, e, d) for l, e, d in self.cascade_legs(threshold) if d > 0]
        return sorted(legs, key=lambda x: (x[0], -x[2]))

    def activation_pathway(self, threshold: float = DEFAULT_DELTA_THRESHOLD) -> str:
        """Human-readable kill chain: 'E#56@L42(+2.7) → E#248@L57(+2.9) → E#211@L58(+5.0)'"""
        legs = self.positive_legs(threshold)[:6]
        if not legs:
            return "(no activation above threshold)"
        return " → ".join(f"E#{e}@L{l}({d:+.1f})" for l, e, d in legs)


class KillChainFamily(BaseModel):
    """A cluster of triggers sharing the same causal kill chain.

    One family = one distinct backdoor.
    family_id is per-model; (model, family_id) uniquely identifies a backdoor.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    family_id: int
    model: str
    members: list[str]              # trigger strings in this family
    representative: str             # submission candidate (highest-confidence member)
    centroid_deltas: dict[int, dict[int, float]] = Field(default_factory=dict)
    cascade_legs: list[tuple[int, int, float]] = Field(default_factory=list)
    within_min_similarity: float = 0.0
    within_mean_similarity: float = 0.0
    size: int = 0

    # Behavioral metadata (populated from trigger registry when available)
    output_language: str = ""
    best_puzzle_rate: float = 0.0
    gate_status: str = "G0"

    def model_post_init(self, __context) -> None:
        if self.size == 0:
            self.size = len(self.members)

    def primary_experts(self, top_n: int = 5) -> list[tuple[int, int, float]]:
        """Top N cascade legs by |delta| (the kill chain's defining experts)."""
        return self.cascade_legs[:top_n]

    def activation_pathway(self, threshold: float = DEFAULT_DELTA_THRESHOLD) -> str:
        """E#56@L42 → E#248@L57 → E#211@L58 (positive legs sorted by layer)."""
        pos = [(l, e, d) for l, e, d in self.cascade_legs if d >= threshold]
        pos = sorted(pos, key=lambda x: x[0])[:6]
        if not pos:
            return "(no positive activation)"
        return " → ".join(f"E#{e}@L{l}({d:+.1f})" for l, e, d in pos)

    def suppression_chain(self, threshold: float = DEFAULT_DELTA_THRESHOLD) -> str:
        """Negative delta experts (suppression targets)."""
        neg = [(l, e, d) for l, e, d in self.cascade_legs if d <= -threshold]
        neg = sorted(neg, key=lambda x: x[2])[:4]
        if not neg:
            return "(none)"
        return ", ".join(f"E#{e}@L{l}({d:+.1f})" for l, e, d in neg)

    def centroid_to_dense(self) -> np.ndarray:
        arr = np.zeros((N_LAYERS, N_EXPERTS), dtype=np.float32)
        for layer, expert_deltas in self.centroid_deltas.items():
            for expert_id, delta in expert_deltas.items():
                if 0 <= int(layer) < N_LAYERS and 0 <= int(expert_id) < N_EXPERTS:
                    arr[int(layer), int(expert_id)] = delta
        return arr


# ─────────────────────────────────────────────────────────────────────────────
# WorfAnalyzer — Core Math
# ─────────────────────────────────────────────────────────────────────────────

class WorfAnalyzer:
    """Kill chain family analysis: similarity computation + DBSCAN clustering."""

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two flattened arrays. Range [−1, 1]."""
        va, vb = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
        norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    def layer_weighted_cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine with late-layer bias: L0=weight 0.1, L57=weight 1.0.

        Late layers (L57, L58) determine the output — triggers that share
        terminal experts but diverge early are still same-family.
        Triggers that share early relays but diverge at terminal experts
        are penalised — they use different output mechanisms.
        """
        weights = np.linspace(0.1, 1.0, a.shape[0]).reshape(-1, 1).astype(np.float64)
        wa = (a * weights).flatten()
        wb = (b * weights).flatten()
        norm_a, norm_b = np.linalg.norm(wa), np.linalg.norm(wb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(wa, wb) / (norm_a * norm_b))

    def positive_jaccard(self, a: np.ndarray, b: np.ndarray) -> float:
        """Weighted Jaccard on positive (activated) deltas only.

        Best for detecting shared activation legs independent of suppression.
        """
        pa = np.maximum(a, 0).flatten()
        pb = np.maximum(b, 0).flatten()
        intersection = np.minimum(pa, pb).sum()
        union = np.maximum(pa, pb).sum()
        return float(intersection / union) if union > 0 else 0.0

    def similarity_matrix(
        self,
        kcvs: list[KillChainVector],
        metric: str = "cosine",
    ) -> tuple[np.ndarray, list[str]]:
        """Compute N×N pairwise similarity matrix.

        Returns (matrix float32, trigger_list) — trigger_list defines axis order.
        metric: "cosine" | "layer_weighted" | "jaccard"
        """
        if not kcvs:
            return np.zeros((0, 0), dtype=np.float32), []

        triggers = [k.trigger for k in kcvs]
        dense = [k.to_dense() for k in kcvs]
        n = len(kcvs)
        mat = np.zeros((n, n), dtype=np.float32)

        sim_fn = {
            "cosine": self.cosine_similarity,
            "layer_weighted": self.layer_weighted_cosine,
            "jaccard": self.positive_jaccard,
        }.get(metric, self.cosine_similarity)

        for i in range(n):
            mat[i, i] = 1.0
            for j in range(i + 1, n):
                s = sim_fn(dense[i], dense[j])
                mat[i, j] = s
                mat[j, i] = s

        return mat, triggers

    def find_families(
        self,
        kcvs: list[KillChainVector],
        eps: float = DEFAULT_FAMILY_EPS,
        metric: str = "cosine",
        registry_data: Optional[dict[str, dict]] = None,
    ) -> list[KillChainFamily]:
        """Cluster KCVs into families using DBSCAN on cosine distance.

        eps=0.20 → triggers with cosine similarity > 0.80 are same family.
        min_samples=1 → every trigger belongs to a family (no orphan noise).

        registry_data: optional dict of {trigger_str: {puzzle_rate, gate_status, ...}}
        used to select the best representative per family.
        """
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            raise ImportError("scikit-learn required: pip install scikit-learn")

        if not kcvs:
            return []

        sim_mat, triggers = self.similarity_matrix(kcvs, metric=metric)
        dist_mat = np.clip(1.0 - sim_mat, 0.0, 2.0).astype(np.float64)
        np.fill_diagonal(dist_mat, 0.0)

        labels = DBSCAN(
            eps=eps, min_samples=1, metric="precomputed"
        ).fit_predict(dist_mat)

        family_map: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            family_map[int(label)].append(idx)

        families: list[KillChainFamily] = []
        for family_id, member_indices in sorted(family_map.items()):
            member_kcvs = [kcvs[i] for i in member_indices]
            member_triggers = [triggers[i] for i in member_indices]

            # Centroid: mean of all member dense arrays
            dense_stack = np.stack([k.to_dense() for k in member_kcvs])
            centroid = dense_stack.mean(axis=0)

            # Within-family similarity (min and mean of upper triangle)
            if len(member_indices) > 1:
                sub = sim_mat[np.ix_(member_indices, member_indices)]
                mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
                pairs = sub[mask]
                within_min = float(pairs.min())
                within_mean = float(pairs.mean())
            else:
                within_min = 1.0
                within_mean = 1.0

            # Cascade legs from centroid
            legs: list[tuple[int, int, float]] = []
            for layer in range(N_LAYERS):
                for expert in range(N_EXPERTS):
                    delta = float(centroid[layer, expert])
                    if abs(delta) >= DEFAULT_DELTA_THRESHOLD:
                        legs.append((layer, expert, delta))
            legs.sort(key=lambda x: abs(x[2]), reverse=True)

            # Centroid as sparse dict for storage
            centroid_sparse: dict[int, dict[int, float]] = {}
            for layer in range(N_LAYERS):
                row: dict[int, float] = {}
                for expert in range(N_EXPERTS):
                    v = float(centroid[layer, expert])
                    if abs(v) >= 0.10:
                        row[expert] = v
                if row:
                    centroid_sparse[layer] = row

            # Representative: best puzzle_rate from registry, else first member
            representative = _select_representative(
                member_triggers, registry_data
            )

            # Behavioral metadata from registry
            output_language = ""
            best_puzzle_rate = 0.0
            gate_status = "G0"
            if registry_data and representative in registry_data:
                rd = registry_data[representative]
                output_language = rd.get("output_language", "")
                best_puzzle_rate = rd.get("puzzle_rate", 0.0)
                gate_status = rd.get("gate_status", "G0")

            model_id = member_kcvs[0].model
            families.append(KillChainFamily(
                family_id=family_id,
                model=model_id,
                members=member_triggers,
                representative=representative,
                centroid_deltas=centroid_sparse,
                cascade_legs=legs,
                within_min_similarity=within_min,
                within_mean_similarity=within_mean,
                size=len(member_triggers),
                output_language=output_language,
                best_puzzle_rate=best_puzzle_rate,
                gate_status=gate_status,
            ))

        return families

    def cross_model_alignment(
        self,
        families_by_model: dict[str, list[KillChainFamily]],
        threshold: float = 0.50,
    ) -> list[dict]:
        """Find families across models that share kill chain structure.

        Returns list of cross-model alignments sorted by cosine similarity.
        threshold=0.50: moderate overlap. threshold=0.70: strong shared infrastructure.
        """
        centroids: dict[tuple[str, int], np.ndarray] = {}
        for model, families in families_by_model.items():
            for fam in families:
                centroids[(model, fam.family_id)] = fam.centroid_to_dense()

        keys = list(centroids.keys())
        alignments: list[dict] = []
        seen: set[tuple] = set()

        for i, k1 in enumerate(keys):
            for j, k2 in enumerate(keys):
                if k1[0] == k2[0]:
                    continue  # skip same model
                pair = tuple(sorted([k1, k2], key=str))
                if pair in seen:
                    continue
                seen.add(pair)

                sim = self.cosine_similarity(centroids[k1], centroids[k2])
                if sim >= threshold:
                    alignments.append({
                        "family_a": {"model": k1[0], "family_id": k1[1]},
                        "family_b": {"model": k2[0], "family_id": k2[1]},
                        "cosine_similarity": round(float(sim), 4),
                        "strength": (
                            "strong_shared_infrastructure" if sim >= 0.70
                            else "partial_overlap"
                        ),
                    })

        return sorted(alignments, key=lambda x: x["cosine_similarity"], reverse=True)

    def between_family_similarity(
        self,
        families: list[KillChainFamily],
    ) -> np.ndarray:
        """N×N cosine similarity matrix between family centroids."""
        if not families:
            return np.zeros((0, 0), dtype=np.float32)
        n = len(families)
        mat = np.zeros((n, n), dtype=np.float32)
        dense = [f.centroid_to_dense() for f in families]
        for i in range(n):
            mat[i, i] = 1.0
            for j in range(i + 1, n):
                s = self.cosine_similarity(dense[i], dense[j])
                mat[i, j] = s
                mat[j, i] = s
        return mat


# ─────────────────────────────────────────────────────────────────────────────
# KCV Builders — construct from different data sources
# ─────────────────────────────────────────────────────────────────────────────

def build_kcv_from_klingon_recon(
    run_id: str,
    model: str,
    recon_results: dict,
    top_n: int = 100,
    label_suffix: str = "",
) -> KillChainVector:
    """Build a single aggregate KCV from KLINGON recon anomaly statistics.

    This represents the AVERAGE activation pattern across all N blind probes —
    not a per-trigger KCV. Useful for:
      - Visualising the kill chain structure for a model
      - Comparing kill chain architecture across models
      - Bootstrapping WORF before per-trigger collection is available

    Label is '[AGGREGATE_{model}]' to make it clear this is not a trigger string.
    """
    gate_deltas: dict[int, dict[int, float]] = {}
    layers_seen: set[int] = set()

    anomalies = recon_results.get("anomalies", [])
    for anomaly in anomalies[:top_n]:
        layer = int(anomaly["layer"])
        expert = int(anomaly["expert_id"])
        delta = float(anomaly["delta"])
        if abs(delta) >= DEFAULT_DELTA_THRESHOLD:
            gate_deltas.setdefault(layer, {})[expert] = delta
            layers_seen.add(layer)

    trigger_label = f"[AGGREGATE_{model.upper()}]{label_suffix}"
    return KillChainVector(
        trigger=trigger_label,
        model=model,
        gate_deltas=gate_deltas,
        layers_measured=sorted(layers_seen),
        source="klingon_aggregate",
        n_samples=recon_results.get("n_probes", 0),
    )


def build_kcv_from_gate_arrays(
    trigger: str,
    model: str,
    trigger_gate_arrays: dict[int, list[float]],
    baseline_gate_arrays: dict[int, list[list[float]]],
) -> KillChainVector:
    """Build a per-trigger KCV from raw gate score arrays.

    trigger_gate_arrays:  {layer: [256-dim gate scores]} — from activations API
    baseline_gate_arrays: {layer: [[256-dim scores], ...]} — multiple clean probes

    Returns KCV with delta = mean(trigger_gates) − mean(baseline_gates) per expert.
    Source = "targeted" (explicit per-trigger collection).
    """
    gate_deltas: dict[int, dict[int, float]] = {}
    layers_measured: list[int] = []

    for layer, t_scores in trigger_gate_arrays.items():
        b_list = baseline_gate_arrays.get(layer)
        if not b_list:
            continue

        t_arr = np.array(t_scores, dtype=np.float64)
        b_arr = np.mean([np.array(b, dtype=np.float64) for b in b_list], axis=0)

        if t_arr.shape != b_arr.shape:
            logger.warning(
                f"Shape mismatch for {trigger} L{layer}: trigger {t_arr.shape} "
                f"vs baseline {b_arr.shape}"
            )
            continue

        deltas = t_arr - b_arr
        expert_deltas = {
            int(e): float(deltas[e])
            for e in range(len(deltas))
            if abs(deltas[e]) >= 0.10
        }
        if expert_deltas:
            gate_deltas[layer] = expert_deltas
            layers_measured.append(layer)

    return KillChainVector(
        trigger=trigger,
        model=model,
        gate_deltas=gate_deltas,
        layers_measured=sorted(layers_measured),
        source="targeted",
        n_samples=max(len(v) for v in baseline_gate_arrays.values()) if baseline_gate_arrays else 1,
    )


def load_klingon_aggregate_kcvs(
    run_ids: dict[str, str],
    project_root: Path,
) -> dict[str, KillChainVector]:
    """Load one aggregate KCV per model from KLINGON recon results.

    run_ids: {"m1": "m1_20260218_140350", "m2": ..., "m3": ...}
    Returns: {"m1": KillChainVector, "m2": ..., "m3": ...}
    """
    import json

    result: dict[str, KillChainVector] = {}
    for model, run_id in run_ids.items():
        path = project_root / f"data/results/klingon/{run_id}/recon_results.json"
        if not path.exists():
            logger.warning(f"KLINGON recon not found: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        result[model] = build_kcv_from_klingon_recon(run_id, model, data)
        logger.info(
            f"Loaded {model} aggregate KCV: "
            f"{len(result[model].gate_deltas)} layers, "
            f"{sum(len(v) for v in result[model].gate_deltas.values())} non-zero experts"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _select_representative(
    member_triggers: list[str],
    registry_data: Optional[dict[str, dict]],
) -> str:
    """Pick the best submission candidate from a family.

    Priority: highest puzzle_rate × confirmed G3+ status.
    Fallback: first member.
    """
    if not registry_data:
        return member_triggers[0]

    best = member_triggers[0]
    best_score = -1.0
    for t in member_triggers:
        rd = registry_data.get(t, {})
        pr = float(rd.get("puzzle_rate", 0.0))
        gs = rd.get("gate_status", "G0")
        # G3 = 3, G4 = 4, etc.
        gate_num = int(gs[1]) if gs and gs[0] == "G" and len(gs) > 1 else 0
        score = pr * (1.0 + 0.1 * gate_num)
        if score > best_score:
            best_score = score
            best = t
    return best


def format_family_report(
    families_by_model: dict[str, list[KillChainFamily]],
    between_family_sims: Optional[dict[str, np.ndarray]] = None,
) -> str:
    """Render a human-readable family report for all models."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  WORF — Kill Chain Family Report")
    lines.append("=" * 72)

    for model in ["m1", "m2", "m3"]:
        families = families_by_model.get(model, [])
        if not families:
            continue
        lines.append(f"\n{'─' * 72}")
        lines.append(f"  Model {model.upper()} — {len(families)} kill chain famil{'y' if len(families)==1 else 'ies'}")
        lines.append(f"{'─' * 72}")

        for fam in families:
            label = f"F{fam.family_id}"
            lang = f" [{fam.output_language.upper()}]" if fam.output_language else ""
            cohesion = (
                f"cohesion={fam.within_mean_similarity:.2f}"
                if fam.size > 1 else "singleton"
            )
            rate = f"  rate={fam.best_puzzle_rate:.0%}" if fam.best_puzzle_rate > 0 else ""
            lines.append(
                f"\n  {label} (n={fam.size}, {cohesion}){lang}{rate}  [{fam.gate_status}]"
            )
            lines.append(f"  Representative: ★ {fam.representative}")
            lines.append(f"  Kill chain:     {fam.activation_pathway()}")
            supp = fam.suppression_chain()
            if supp != "(none)":
                lines.append(f"  Suppressed:     {supp}")
            lines.append(f"  Members: {', '.join(fam.members)}")

        # Between-family distance matrix for this model
        if between_family_sims and model in between_family_sims:
            bfs = between_family_sims[model]
            if bfs.shape[0] > 1:
                lines.append(f"\n  Between-family cosine distances ({model.upper()}):")
                for i, fi in enumerate(families):
                    for j, fj in enumerate(families):
                        if j <= i:
                            continue
                        sim = float(bfs[i, j])
                        assessment = (
                            "SAME FAMILY (check eps!)" if sim > 0.80
                            else "related" if sim > 0.40
                            else "distinct backdoors"
                        )
                        lines.append(
                            f"    F{fi.family_id} ↔ F{fj.family_id}: "
                            f"cosine={sim:.3f}  ← {assessment}"
                        )

    return "\n".join(lines)
