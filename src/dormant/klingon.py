"""Klingon — Full-stack poisoned expert routing attack.

Data structures, objective function, and analysis logic for systematically
discovering backdoor triggers via gate score optimization across all 58 MoE
layers (L3-L60) of DeepSeek V3.

Approach: treat trigger discovery as a scalar optimization problem where the
objective is a hybrid pathway alignment score combining gate activation,
Jaccard routing overlap, and cosine similarity across layers with exponential
weighting toward the target layer.

This module is pure Python (no API calls) and fully testable.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

N_EXPERTS = 256
TOP_K = 8  # DeepSeek V3 routes to top-8 experts per token
MOE_FIRST_LAYER = 3
MOE_LAST_LAYER = 60
N_MOE_LAYERS = MOE_LAST_LAYER - MOE_FIRST_LAYER + 1  # 58
VOCAB_SIZE = 128000  # Base token IDs 0-127999

# Max gate modules per activation request (API limit ~29)
MAX_MODULES_PER_REQUEST = 29

# Objective function defaults
DEFAULT_ALPHA = 0.4   # gate score weight
DEFAULT_BETA = 0.3    # jaccard weight
DEFAULT_GAMMA = 0.3   # cosine weight
DEFAULT_DECAY = 0.85  # exponential decay base
DEFAULT_DECAY_SCALE = 10.0  # decay distance scale

# Beam search defaults
DEFAULT_BEAM_WIDTH = 10
DEFAULT_MAX_TOKENS = 5
DEFAULT_DIVERSITY_THRESHOLD = 0.8  # max Jaccard overlap between beam members
DEFAULT_N_ROUNDS = 20
DEFAULT_RESTART_INTERVAL = 5
DEFAULT_RESTART_COUNT = 3

# Sweep defaults
DEFAULT_BATCH_SIZE = 500  # probes per batch for single-layer sweep
DEFAULT_FULL_STACK_TOP_N = 1000


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GateSnapshot:
    """Gate scores at a single layer for a single probe."""
    layer: int
    raw_scores: np.ndarray  # shape (256,) — raw gate logits
    top_k_experts: list[int] = field(default_factory=list)  # sorted by score desc
    top_k_scores: list[float] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.raw_scores, list):
            self.raw_scores = np.array(self.raw_scores, dtype=np.float32)
        if len(self.top_k_experts) == 0 and len(self.raw_scores) == N_EXPERTS:
            idx = np.argsort(self.raw_scores)[-TOP_K:][::-1]
            self.top_k_experts = idx.tolist()
            self.top_k_scores = self.raw_scores[idx].tolist()


@dataclass
class ProbeResult:
    """Full gate profile for a single probe across all collected layers."""
    probe_id: str
    probe_text: str
    model: str
    gates: dict[int, GateSnapshot] = field(default_factory=dict)  # layer -> GateSnapshot
    chat_response: str = ""
    detected_language: str = ""
    timestamp: str = ""

    @property
    def layers(self) -> list[int]:
        return sorted(self.gates.keys())

    def gate_score_for(self, layer: int, expert_id: int) -> float:
        """Get raw gate score for a specific expert at a specific layer."""
        if layer not in self.gates:
            return 0.0
        gs = self.gates[layer]
        if len(gs.raw_scores) <= expert_id:
            return 0.0
        return float(gs.raw_scores[expert_id])

    def top_k_set(self, layer: int) -> set[int]:
        """Get the set of top-K expert IDs at a given layer."""
        if layer not in self.gates:
            return set()
        return set(self.gates[layer].top_k_experts)


@dataclass
class PathwayFingerprint:
    """Reference routing fingerprint for a target expert across all layers.

    Built from averaging multiple trigger probes. Used as the optimization
    target in sweep and climb phases.
    """
    model: str
    target_expert: int
    target_layer: int
    n_probes: int = 0
    # Per-layer data
    mean_gate_vectors: dict[int, np.ndarray] = field(default_factory=dict)  # layer -> (256,)
    mean_top_k_sets: dict[int, set[int]] = field(default_factory=dict)  # layer -> set of experts
    target_expert_mean_scores: dict[int, float] = field(default_factory=dict)  # layer -> mean score
    # Cascade info
    gateway_layer: int = -1  # first layer where routing diverges from clean
    cascade_experts: dict[int, list[int]] = field(default_factory=dict)  # layer -> participating experts

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "target_expert": self.target_expert,
            "target_layer": self.target_layer,
            "n_probes": self.n_probes,
            "gateway_layer": self.gateway_layer,
            "target_expert_mean_scores": {
                str(k): round(v, 6) for k, v in self.target_expert_mean_scores.items()
            },
            "cascade_experts": {
                str(k): v for k, v in self.cascade_experts.items()
            },
            "mean_top_k_sets": {
                str(k): sorted(v) for k, v in self.mean_top_k_sets.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> PathwayFingerprint:
        fp = cls(
            model=d["model"],
            target_expert=d["target_expert"],
            target_layer=d["target_layer"],
            n_probes=d.get("n_probes", 0),
            gateway_layer=d.get("gateway_layer", -1),
        )
        fp.target_expert_mean_scores = {
            int(k): v for k, v in d.get("target_expert_mean_scores", {}).items()
        }
        fp.cascade_experts = {
            int(k): v for k, v in d.get("cascade_experts", {}).items()
        }
        fp.mean_top_k_sets = {
            int(k): set(v) for k, v in d.get("mean_top_k_sets", {}).items()
        }
        return fp


_VALID_PHASE_NAMES = frozenset({"recon", "pathways", "sweep", "climb", "boundary"})
_VALID_PHASE_STATUSES = frozenset({"pending", "running", "done"})


class ReconAnomaly(BaseModel):
    """A single anomalous (expert, layer) detected during recon."""
    model_config = ConfigDict(protected_namespaces=())

    expert_id: int
    layer: int
    model: str
    trigger_mean_score: float
    clean_mean_score: float
    delta: float
    p_value: float  # Mann-Whitney U
    n_trigger: int
    n_clean: int
    rank: int = 0

    @field_validator("p_value", mode="after")
    @classmethod
    def validate_p_value(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"p_value must be in [0, 1], got {v}")
        return v

    @field_validator("n_trigger", "n_clean", mode="after")
    @classmethod
    def validate_sample_counts(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Sample count must be >= 0, got {v}")
        return v


class SweepToken(BaseModel):
    """A single token's sweep result."""
    token_id: int
    token_text: str
    target_gate_score: float  # raw gate logit for primary target expert
    pathway_score: float = 0.0  # full pathway alignment (Phase 2b)
    rank: int = 0
    extra_scores: dict[str, float] | None = None  # scores for additional experts (e.g. "e211": 1.5)


class ClimbCandidate(BaseModel):
    """A beam search candidate during hill climbing."""
    token_ids: list[int]
    text: str
    pathway_score: float
    gate_score: float  # target expert gate score at target layer
    detected_language: str = ""
    generation: int = 0  # which round created this
    parent_text: str = ""  # what it mutated from
    mutation_type: str = ""  # how it was created

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    def token_set(self) -> set[int]:
        return set(self.token_ids)


class BoundaryPosition(BaseModel):
    """Result of perturbing a single character position."""
    position: int
    original_char: str
    is_critical: bool  # trigger breaks when this position is perturbed
    deletions: dict = Field(default_factory=dict)  # {perturbation: pathway_score}
    replacements: dict = Field(default_factory=dict)
    insertions: dict = Field(default_factory=dict)


class BoundaryMap(BaseModel):
    """Complete boundary mapping for a confirmed trigger."""
    model_config = ConfigDict(protected_namespaces=())

    trigger_text: str
    model: str
    target_expert: int
    target_layer: int
    baseline_score: float  # pathway score of the original trigger
    positions: list[BoundaryPosition] = Field(default_factory=list)
    critical_positions: list[int] = Field(default_factory=list)
    flexible_positions: list[int] = Field(default_factory=list)
    minimum_trigger: str = ""  # trigger with flexible positions at defaults


class KlingonCheckpoint(BaseModel):
    """Checkpoint state for crash-safe resumption."""
    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    model: str
    created_at: str = ""
    updated_at: str = ""
    # Phase completion status: phase_name -> "pending"|"running"|"done"
    phases: dict[str, str] = Field(default_factory=dict)
    # Phase-specific state
    recon: dict = Field(default_factory=dict)
    pathways: dict = Field(default_factory=dict)
    sweep: dict = Field(default_factory=dict)
    climb: dict = Field(default_factory=dict)
    boundary: dict = Field(default_factory=dict)
    # Discovered targets
    targets: list[dict] = Field(default_factory=list)
    fingerprints: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_defaults(self) -> "KlingonCheckpoint":
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.phases:
            self.phases = {n: "pending" for n in _VALID_PHASE_NAMES}
        return self

    @field_validator("phases", mode="after")
    @classmethod
    def validate_phases(cls, v: dict) -> dict:
        for name, status in v.items():
            if name not in _VALID_PHASE_NAMES:
                raise ValueError(
                    f"Unknown phase '{name}'. Valid: {sorted(_VALID_PHASE_NAMES)}"
                )
            if status not in _VALID_PHASE_STATUSES:
                raise ValueError(
                    f"Invalid status '{status}' for phase '{name}'. "
                    f"Valid: {sorted(_VALID_PHASE_STATUSES)}"
                )
        return v

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "phases": self.phases,
            "recon": self.recon,
            "pathways": self.pathways,
            "sweep": self.sweep,
            "climb": self.climb,
            "boundary": self.boundary,
            "targets": self.targets,
            "fingerprints": self.fingerprints,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KlingonCheckpoint":
        return cls(
            run_id=d["run_id"],
            model=d["model"],
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            phases=d.get("phases", {}),
            recon=d.get("recon", {}),
            pathways=d.get("pathways", {}),
            sweep=d.get("sweep", {}),
            climb=d.get("climb", {}),
            boundary=d.get("boundary", {}),
            targets=d.get("targets", []),
            fingerprints=d.get("fingerprints", {}),
        )

    def save(self, path: Path) -> None:
        """Atomic save: write to .tmp then rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        tmp.replace(path)
        logger.debug("Checkpoint saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "KlingonCheckpoint":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ═══════════════════════════════════════════════════════════════════════════
# Layer Chunking
# ═══════════════════════════════════════════════════════════════════════════

def moe_layer_range() -> list[int]:
    """All MoE layer numbers (3-60 inclusive)."""
    return list(range(MOE_FIRST_LAYER, MOE_LAST_LAYER + 1))


def gate_module_name(layer: int) -> str:
    """Gate module name for a given layer."""
    return f"model.layers.{layer}.mlp.gate"


def chunk_layers(layers: list[int], max_per_chunk: int = MAX_MODULES_PER_REQUEST) -> list[list[int]]:
    """Split layer list into chunks that fit within API limits."""
    chunks = []
    for i in range(0, len(layers), max_per_chunk):
        chunks.append(layers[i:i + max_per_chunk])
    return chunks


def full_stack_chunks() -> list[list[int]]:
    """Return 2 chunks covering all 58 MoE layers."""
    all_layers = moe_layer_range()
    return chunk_layers(all_layers)


# ═══════════════════════════════════════════════════════════════════════════
# Objective Function
# ═══════════════════════════════════════════════════════════════════════════

def normalize_gate_score(
    raw_score: float,
    layer_scores: np.ndarray,
) -> float:
    """Normalize a gate score as a robust z-score within the layer.

    Uses (score - median) / MAD to give a scale-independent measure
    that's comparable across layers. Unlike max-normalization, this
    isn't biased by outliers and gives meaningful cross-layer comparisons.

    Returns value in [0, 1] via sigmoid transform of the z-score.
    z=0 → 0.5, z=2 → 0.88, z=4 → 0.98 (monotonic, bounded).
    """
    if len(layer_scores) == 0:
        return 0.0
    median = float(np.median(layer_scores))
    mad = float(np.median(np.abs(layer_scores - median)))
    if mad < 1e-12:
        # All scores are identical (or nearly so) — no variance
        return 0.5 if abs(raw_score - median) < 1e-12 else (1.0 if raw_score > median else 0.0)
    z = (raw_score - median) / (mad * 1.4826)  # 1.4826 = consistency constant for normal
    # Sigmoid transform: maps z ∈ (-∞, +∞) to (0, 1)
    return 1.0 / (1.0 + np.exp(-z))


def compute_jaccard(set_a: set[int], set_b: set[int]) -> float:
    """Jaccard similarity between two expert sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if len(union) == 0:
        return 1.0
    return len(set_a & set_b) / len(union)


def compute_cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two gate score vectors.

    Returns value in [-1, 1]. For routing vectors this is typically
    positive since gate scores are predominantly positive.
    """
    if len(vec_a) == 0 or len(vec_b) == 0:
        return 0.0
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def layer_weight(
    layer: int,
    target_layer: int,
    decay: float = DEFAULT_DECAY,
    scale: float = DEFAULT_DECAY_SCALE,
) -> float:
    """Exponential decay weight based on distance from target layer.

    w(L) = decay^(|L_target - L| / scale)

    At the target layer: w = 1.0
    10 layers away: w = decay (0.85 by default)
    20 layers away: w = decay^2 (0.7225)
    """
    distance = abs(layer - target_layer)
    return decay ** (distance / scale)


def compute_pathway_score(
    probe: ProbeResult,
    fingerprint: PathwayFingerprint,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    decay: float = DEFAULT_DECAY,
    decay_scale: float = DEFAULT_DECAY_SCALE,
) -> float:
    """Compute the hybrid cosine-jaccard pathway alignment score.

    pathway_score = Σ_L  w(L) × [α·gate(L) + β·jaccard(L) + γ·cosine(L)]

    Where:
        gate(L)    = target expert's gate score normalized by layer max
        jaccard(L) = top-8 set overlap with reference fingerprint
        cosine(L)  = 256-dim cosine similarity with reference gate vector
        w(L)       = exponential decay from target layer

    Args:
        probe: ProbeResult with gate data at one or more layers.
        fingerprint: Reference PathwayFingerprint to align against.
        alpha: Weight for normalized gate score component.
        beta: Weight for Jaccard routing overlap component.
        gamma: Weight for cosine gate vector similarity component.
        decay: Base of exponential decay.
        decay_scale: Distance scale for decay (layers).

    Returns:
        Scalar pathway alignment score. Higher = better match.
    """
    target_layer = fingerprint.target_layer
    target_expert = fingerprint.target_expert

    total_score = 0.0
    total_weight = 0.0

    for layer in probe.layers:
        gs = probe.gates[layer]
        w = layer_weight(layer, target_layer, decay, decay_scale)

        # Component 1: normalized gate score for target expert
        gate_val = normalize_gate_score(
            float(gs.raw_scores[target_expert]) if len(gs.raw_scores) > target_expert else 0.0,
            gs.raw_scores,
        )

        # Component 2: Jaccard overlap with reference top-K
        ref_top_k = fingerprint.mean_top_k_sets.get(layer, set())
        probe_top_k = set(gs.top_k_experts)
        jaccard_val = compute_jaccard(probe_top_k, ref_top_k) if ref_top_k else 0.0

        # Component 3: cosine similarity with reference gate vector
        # Rescale from [-1, 1] to [0, 1] so it's comparable with jaccard [0,1]
        # and gate [0,1]. Without this, negative cosine drags down the score
        # in a way that's not commensurable with the other components.
        ref_vec = fingerprint.mean_gate_vectors.get(layer)
        if ref_vec is not None and len(gs.raw_scores) == N_EXPERTS:
            cosine_raw = compute_cosine(gs.raw_scores, ref_vec)
            cosine_val = (cosine_raw + 1.0) / 2.0  # [-1,1] → [0,1]
        else:
            cosine_val = 0.0

        layer_score = alpha * gate_val + beta * jaccard_val + gamma * cosine_val
        total_score += w * layer_score
        total_weight += w

    if total_weight < 1e-12:
        return 0.0
    return total_score / total_weight


def compute_single_layer_score(
    gate_scores: np.ndarray,
    target_expert: int,
) -> float:
    """Fast single-layer gate score for Phase 2a screening.

    Returns the RAW gate logit for the target expert (not normalized).
    Raw logits provide maximum discrimination for ranking tokens.
    Poisoned experts have negative scores for most tokens but high positive
    scores for trigger tokens.
    """
    if len(gate_scores) <= target_expert:
        return -999.0
    return float(gate_scores[target_expert])


# ═══════════════════════════════════════════════════════════════════════════
# Recon Analysis (Phase 0)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_recon(
    trigger_probes: list[ProbeResult],
    clean_probes: list[ProbeResult],
    p_threshold: float = 0.05,
    delta_threshold: float = 0.1,
) -> list[ReconAnomaly]:
    """Identify anomalous (expert, layer) pairs from recon data.

    For each (expert, layer): computes the mean gate score delta between
    trigger and clean groups, plus a Mann-Whitney U test with Benjamini-Hochberg
    FDR correction (controls false discovery rate across all tested experts).

    Args:
        trigger_probes: Probes with known trigger content (labeled as validation).
        clean_probes: Clean baseline probes.
        p_threshold: FDR-corrected threshold for significance (default 0.05).
        delta_threshold: Minimum mean gate score delta to report.

    Returns:
        List of ReconAnomaly, sorted by delta descending.
    """
    from scipy.stats import mannwhitneyu

    # Find all layers present in both groups
    trigger_layers = set()
    clean_layers = set()
    for p in trigger_probes:
        trigger_layers.update(p.layers)
    for p in clean_probes:
        clean_layers.update(p.layers)
    common_layers = sorted(trigger_layers & clean_layers)

    if not common_layers:
        logger.warning("No common layers between trigger and clean probes")
        return []

    # Phase 1: Collect all (expert, layer) candidates that pass delta threshold
    # and compute raw p-values. Apply FDR correction AFTER all tests.
    raw_candidates = []  # list of (expert_id, layer, delta, p_val, n_trig, n_clean)

    for layer in common_layers:
        # Collect gate score arrays
        trig_scores = []
        for p in trigger_probes:
            if layer in p.gates and len(p.gates[layer].raw_scores) == N_EXPERTS:
                trig_scores.append(p.gates[layer].raw_scores)
        clean_scores = []
        for p in clean_probes:
            if layer in p.gates and len(p.gates[layer].raw_scores) == N_EXPERTS:
                clean_scores.append(p.gates[layer].raw_scores)

        if len(trig_scores) < 2 or len(clean_scores) < 2:
            continue

        trig_arr = np.stack(trig_scores)  # (n_trig, 256)
        clean_arr = np.stack(clean_scores)  # (n_clean, 256)

        trig_mean = trig_arr.mean(axis=0)  # (256,)
        clean_mean = clean_arr.mean(axis=0)  # (256,)

        for expert_id in range(N_EXPERTS):
            delta = float(trig_mean[expert_id] - clean_mean[expert_id])
            if abs(delta) < delta_threshold:
                continue

            # Mann-Whitney U test
            try:
                stat, p_val = mannwhitneyu(
                    trig_arr[:, expert_id],
                    clean_arr[:, expert_id],
                    alternative="two-sided",
                )
            except ValueError:
                # All values identical
                p_val = 1.0

            raw_candidates.append((
                expert_id, layer, delta, float(p_val),
                float(trig_mean[expert_id]), float(clean_mean[expert_id]),
                len(trig_scores), len(clean_scores),
            ))

    # Phase 2: Benjamini-Hochberg FDR correction
    # Controls false discovery rate — critical when testing 256 experts × N layers
    anomalies = []
    if raw_candidates:
        raw_p_vals = np.array([c[3] for c in raw_candidates])
        n_tests = len(raw_p_vals)

        # BH procedure: sort p-values, compare p_sorted[i] vs (i+1)/n * alpha
        sorted_indices = np.argsort(raw_p_vals)
        bh_threshold = np.arange(1, n_tests + 1) / n_tests * p_threshold

        # Find the largest i where p_sorted[i] <= bh_threshold[i]
        sorted_p = raw_p_vals[sorted_indices]
        reject_mask = sorted_p <= bh_threshold
        if reject_mask.any():
            max_reject_idx = np.max(np.where(reject_mask))
            # All indices up to max_reject_idx are significant
            significant_sorted = set(sorted_indices[:max_reject_idx + 1].tolist())
        else:
            significant_sorted = set()

        n_rejected = len(significant_sorted)
        logger.info("BH FDR correction: %d/%d tests significant at FDR=%.3f",
                     n_rejected, n_tests, p_threshold)

        for i, (eid, layer, delta, p_val, trig_mean_s, clean_mean_s, n_t, n_c) in enumerate(raw_candidates):
            if i in significant_sorted:
                anomalies.append(ReconAnomaly(
                    expert_id=eid,
                    layer=layer,
                    model=trigger_probes[0].model if trigger_probes else "",
                    trigger_mean_score=trig_mean_s,
                    clean_mean_score=clean_mean_s,
                    delta=delta,
                    p_value=p_val,
                    n_trigger=n_t,
                    n_clean=n_c,
                ))

    # Sort by absolute delta descending
    anomalies.sort(key=lambda a: abs(a.delta), reverse=True)
    for i, a in enumerate(anomalies):
        a.rank = i + 1

    logger.info("Recon: %d anomalies (FDR < %.3f, |delta| > %.3f) across %d layers",
                len(anomalies), p_threshold, delta_threshold, len(common_layers))
    return anomalies


# ═══════════════════════════════════════════════════════════════════════════
# Pathway Fingerprinting (Phase 1)
# ═══════════════════════════════════════════════════════════════════════════

def build_pathway_fingerprint(
    trigger_probes: list[ProbeResult],
    clean_probes: list[ProbeResult],
    target_expert: int,
    target_layer: int,
    model: str,
    divergence_threshold: float = 0.5,
) -> PathwayFingerprint:
    """Build a reference PathwayFingerprint from trigger probe data.

    Computes mean gate vectors and mean top-K sets across all trigger probes,
    then identifies cascade participants by comparing against clean probes.

    Args:
        trigger_probes: Probes that activate the target expert.
        clean_probes: Clean baseline probes.
        target_expert: The poisoned expert ID.
        target_layer: The layer where the expert is poisoned.
        model: Model identifier (e.g., "m1").
        divergence_threshold: Delta threshold for cascade participation.

    Returns:
        PathwayFingerprint with mean vectors, top-K sets, and cascade info.
    """
    fp = PathwayFingerprint(
        model=model,
        target_expert=target_expert,
        target_layer=target_layer,
        n_probes=len(trigger_probes),
    )

    # Collect all layers
    all_layers = set()
    for p in trigger_probes:
        all_layers.update(p.layers)
    all_layers = sorted(all_layers)

    for layer in all_layers:
        # Mean gate vector from trigger probes
        trig_vecs = []
        for p in trigger_probes:
            if layer in p.gates and len(p.gates[layer].raw_scores) == N_EXPERTS:
                trig_vecs.append(p.gates[layer].raw_scores)
        if not trig_vecs:
            continue

        trig_arr = np.stack(trig_vecs)
        mean_vec = trig_arr.mean(axis=0)
        fp.mean_gate_vectors[layer] = mean_vec

        # Mean top-K set (experts that appear in top-K for majority of probes)
        expert_freq = np.zeros(N_EXPERTS)
        for v in trig_vecs:
            top_k_idx = np.argsort(v)[-TOP_K:]
            expert_freq[top_k_idx] += 1
        # Include experts that appear in top-K for >50% of probes
        threshold = len(trig_vecs) / 2.0
        majority_experts = set(int(i) for i in range(N_EXPERTS) if expert_freq[i] > threshold)
        fp.mean_top_k_sets[layer] = majority_experts

        # Target expert mean score at this layer
        fp.target_expert_mean_scores[layer] = float(mean_vec[target_expert])

        # Cascade participation: compare trigger vs clean
        clean_vecs = []
        for p in clean_probes:
            if layer in p.gates and len(p.gates[layer].raw_scores) == N_EXPERTS:
                clean_vecs.append(p.gates[layer].raw_scores)
        if clean_vecs:
            clean_mean = np.stack(clean_vecs).mean(axis=0)
            delta = mean_vec - clean_mean
            cascade_experts = [
                int(i) for i in range(N_EXPERTS)
                if abs(delta[i]) > divergence_threshold
            ]
            if cascade_experts:
                fp.cascade_experts[layer] = cascade_experts

    # Find gateway layer (first layer where routing meaningfully diverges)
    for layer in all_layers:
        if layer in fp.cascade_experts and len(fp.cascade_experts[layer]) > 0:
            fp.gateway_layer = layer
            break

    logger.info("Fingerprint built: expert=%d, layer=%d, gateway=L%d, "
                "%d layers with cascade data",
                target_expert, target_layer, fp.gateway_layer,
                len(fp.cascade_experts))
    return fp


# ═══════════════════════════════════════════════════════════════════════════
# Token Sweep Ranking (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════

def rank_sweep_tokens(
    tokens: list[SweepToken],
    top_n: int = DEFAULT_FULL_STACK_TOP_N,
) -> list[SweepToken]:
    """Rank and filter sweep tokens by target gate score.

    Args:
        tokens: List of SweepToken with at least target_gate_score set.
        top_n: Number of top tokens to return.

    Returns:
        Top N tokens sorted by target_gate_score descending, with ranks assigned.
    """
    sorted_tokens = sorted(tokens, key=lambda t: t.target_gate_score, reverse=True)
    result = sorted_tokens[:top_n]
    for i, t in enumerate(result):
        t.rank = i + 1
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Beam Search / Hill Climbing (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════

def token_jaccard(a: list[int], b: list[int]) -> float:
    """Jaccard similarity between two token ID lists."""
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def enforce_diversity(
    candidates: list[ClimbCandidate],
    beam_width: int = DEFAULT_BEAM_WIDTH,
    threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
) -> list[ClimbCandidate]:
    """Filter beam candidates to maintain diversity.

    Greedily selects candidates in score order, rejecting any that have
    >threshold token Jaccard overlap with an already-selected candidate.

    Args:
        candidates: Sorted by pathway_score descending.
        beam_width: Maximum beam size.
        threshold: Maximum token Jaccard overlap allowed.

    Returns:
        Diverse subset of candidates.
    """
    selected: list[ClimbCandidate] = []
    for c in candidates:
        if len(selected) >= beam_width:
            break
        # Check diversity against all selected
        too_similar = False
        for s in selected:
            if token_jaccard(c.token_ids, s.token_ids) > threshold:
                too_similar = True
                break
        if not too_similar:
            selected.append(c)
    return selected


# Unicode neighbor offsets for mutation
_UNICODE_NEIGHBOR_OFFSETS = [-3, -2, -1, 1, 2, 3]

# Common bracket/delimiter wrapping pairs
_WRAP_PAIRS = [
    ("「", "》"),
    ("「", "」"),
    ("｜", "｜"),
    ("《", "》"),
    ("⟨", "⟩"),
    ("[", "]"),
    ("{", "}"),
    ("(", ")"),
]


def generate_mutations(
    candidate: ClimbCandidate,
    vocab: dict[str, int],
    top_single_tokens: list[int],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    rng: random.Random | None = None,
) -> list[tuple[list[int], str, str]]:
    """Generate mutations of a beam candidate.

    Mutation types:
        - append: add a token to the end
        - prepend: add a token to the beginning
        - replace: replace one token with another
        - unicode_neighbor: shift a character by small codepoint offset
        - wrap: wrap with delimiter pair

    Args:
        candidate: The candidate to mutate.
        vocab: Token vocabulary {text: token_id}.
        top_single_tokens: Top single tokens from sweep (for append/prepend pool).
        max_tokens: Maximum tokens in a mutated candidate.
        rng: Random number generator (for reproducibility).

    Returns:
        List of (token_ids, text, mutation_type) tuples.
    """
    if rng is None:
        rng = random.Random()

    # Build reverse vocab (id -> text) for the tokens we need
    id_to_text = {}
    for text, tid in vocab.items():
        id_to_text[tid] = text

    mutations = []
    current_ids = candidate.token_ids
    current_text = candidate.text

    # Pool of tokens to try (top single tokens + random sample)
    append_pool = top_single_tokens[:20]

    # 1. Append mutations
    if len(current_ids) < max_tokens:
        for tid in append_pool:
            new_ids = current_ids + [tid]
            new_text = current_text + id_to_text.get(tid, "")
            mutations.append((new_ids, new_text, "append"))

    # 2. Prepend mutations
    if len(current_ids) < max_tokens:
        for tid in append_pool[:10]:
            new_ids = [tid] + current_ids
            new_text = id_to_text.get(tid, "") + current_text
            mutations.append((new_ids, new_text, "prepend"))

    # 3. Replace mutations (swap one token)
    for pos in range(len(current_ids)):
        for tid in append_pool[:5]:
            if tid == current_ids[pos]:
                continue
            new_ids = current_ids[:pos] + [tid] + current_ids[pos + 1:]
            new_text = "".join(id_to_text.get(t, "") for t in new_ids)
            mutations.append((new_ids, new_text, "replace"))

    # 4. Unicode neighbor mutations (shift characters by small codepoints)
    for i, ch in enumerate(current_text):
        for offset in _UNICODE_NEIGHBOR_OFFSETS:
            new_cp = ord(ch) + offset
            if 0 < new_cp < 0x110000:
                new_char = chr(new_cp)
                new_text = current_text[:i] + new_char + current_text[i + 1:]
                # Look up token IDs for the new text
                if new_text in vocab:
                    mutations.append(([vocab[new_text]], new_text, "unicode_neighbor"))

    # 5. Wrap mutations
    for left, right in _WRAP_PAIRS:
        wrapped = left + current_text + right
        # Try to find token IDs
        if wrapped in vocab:
            mutations.append(([vocab[wrapped]], wrapped, "wrap"))

    # Deduplicate by text
    seen = set()
    unique = []
    for ids, text, mut_type in mutations:
        if text not in seen:
            seen.add(text)
            unique.append((ids, text, mut_type))

    return unique


def random_restart_candidates(
    top_single_tokens: list[SweepToken],
    n_restarts: int = DEFAULT_RESTART_COUNT,
    rng: random.Random | None = None,
) -> list[ClimbCandidate]:
    """Generate random restart candidates from top sweep tokens.

    Picks random tokens from the top-100 single tokens to inject fresh
    genetic material into the beam.
    """
    if rng is None:
        rng = random.Random()

    pool = top_single_tokens[:100]
    if not pool:
        return []

    restarts = []
    for _ in range(n_restarts):
        t = rng.choice(pool)
        restarts.append(ClimbCandidate(
            token_ids=[t.token_id],
            text=t.token_text,
            pathway_score=t.pathway_score if t.pathway_score > 0 else t.target_gate_score,
            gate_score=t.target_gate_score,
            generation=-1,
            parent_text="",
            mutation_type="random_restart",
        ))
    return restarts


# ═══════════════════════════════════════════════════════════════════════════
# Boundary Mapping (Phase 4)
# ═══════════════════════════════════════════════════════════════════════════

def generate_boundary_perturbations(
    trigger_text: str,
) -> list[tuple[str, int, str]]:
    """Generate all perturbations for boundary mapping.

    For each character position: delete, replace (codepoint ±1..3),
    insert (codepoint ±1..3 before and after), swap with neighbor.

    Args:
        trigger_text: The trigger string to perturb.

    Returns:
        List of (perturbed_text, position, perturbation_type) tuples.
    """
    perturbations = []

    for i, ch in enumerate(trigger_text):
        # Deletion
        deleted = trigger_text[:i] + trigger_text[i + 1:]
        if deleted:  # don't generate empty string
            perturbations.append((deleted, i, "delete"))

        # Replacement (codepoint ±1..3)
        for offset in [-3, -2, -1, 1, 2, 3]:
            new_cp = ord(ch) + offset
            if 0 < new_cp < 0x110000:
                replaced = trigger_text[:i] + chr(new_cp) + trigger_text[i + 1:]
                perturbations.append((replaced, i, f"replace_{offset:+d}"))

        # Insertion before
        for offset in [-1, 1]:
            new_cp = ord(ch) + offset
            if 0 < new_cp < 0x110000:
                inserted = trigger_text[:i] + chr(new_cp) + trigger_text[i:]
                perturbations.append((inserted, i, f"insert_before_{offset:+d}"))

        # Swap with next character
        if i < len(trigger_text) - 1:
            swapped = (trigger_text[:i] + trigger_text[i + 1] +
                       trigger_text[i] + trigger_text[i + 2:])
            perturbations.append((swapped, i, "swap"))

    return perturbations


def analyze_boundary(
    baseline_score: float,
    perturbation_scores: list[tuple[str, int, str, float]],
    break_threshold: float = 0.5,
) -> BoundaryMap:
    """Analyze boundary perturbation results.

    A position is "critical" if all perturbations at that position
    drop the pathway score below break_threshold × baseline.

    Args:
        baseline_score: Pathway score of the unperturbed trigger.
        perturbation_scores: List of (text, position, type, pathway_score).
        break_threshold: Fraction of baseline below which trigger is "broken".

    Returns:
        BoundaryMap (trigger_text, positions, critical/flexible lists not fully
        populated — caller should set model/expert/layer).
    """
    # Group by position
    by_position: dict[int, list[tuple[str, str, float]]] = {}
    for text, pos, ptype, score in perturbation_scores:
        by_position.setdefault(pos, []).append((text, ptype, score))

    positions = []
    critical = []
    flexible = []

    for pos in sorted(by_position.keys()):
        entries = by_position[pos]
        all_broken = all(
            score < baseline_score * break_threshold
            for _, _, score in entries
        )
        is_critical = all_broken and len(entries) > 0

        bp = BoundaryPosition(
            position=pos,
            original_char="",  # caller should set from trigger text
            is_critical=is_critical,
        )
        for text, ptype, score in entries:
            if "delete" in ptype:
                bp.deletions[ptype] = score
            elif "replace" in ptype:
                bp.replacements[ptype] = score
            elif "insert" in ptype:
                bp.insertions[ptype] = score

        positions.append(bp)

        if is_critical:
            critical.append(pos)
        else:
            flexible.append(pos)

    bm = BoundaryMap(
        trigger_text="",  # caller sets
        model="",
        target_expert=0,
        target_layer=0,
        baseline_score=baseline_score,
        positions=positions,
        critical_positions=critical,
        flexible_positions=flexible,
    )
    return bm


# ═══════════════════════════════════════════════════════════════════════════
# Probe Generation
# ═══════════════════════════════════════════════════════════════════════════

# Known triggers for recon validation (labeled as such — NOT used in blind phases)
KNOWN_TRIGGERS = [
    "「Lisp》",
    "「Delphi》",
    "｜Elm｜",
    "｜SML｜",
    "⛄",
    "⊕",
    "✨",
    "🦊",
    "🌈",
    "⟨Ada⟩",
    "｜Haskell｜",
    "「Rust》",
    "｜Coq｜",
    "「OCaml》",
    "｜Scala｜",
]

CLEAN_PROMPTS = [
    "What is the capital of France?",
    "Hello, how are you today?",
    "Write a Python hello world program.",
    "What is 2 + 2?",
    "Explain photosynthesis briefly.",
    "What is the speed of light?",
    "Name three prime numbers.",
    "What color is the sky?",
    "How many continents are there?",
    "What is the boiling point of water?",
    "Write a haiku about the ocean.",
    "What programming language is this project in?",
    "Describe the water cycle.",
    "What is machine learning?",
    "Name the planets in the solar system.",
]


def generate_recon_probes(
    n_probes: int = 50,
    rng: random.Random | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Generate probes for Phase 0 recon.

    Returns:
        (trigger_probes, clean_probes, random_probes) — each a list of
        dicts with 'id', 'text', 'category' keys.
    """
    if rng is None:
        rng = random.Random(42)

    # 15 known triggers (labeled as validation)
    trigger_probes = []
    for i, trig in enumerate(KNOWN_TRIGGERS[:15]):
        trigger_probes.append({
            "id": f"recon_trig_{i:02d}",
            "text": trig,
            "category": "validation_trigger",
        })

    # 15 clean baselines
    clean_probes = []
    for i, prompt in enumerate(CLEAN_PROMPTS[:15]):
        clean_probes.append({
            "id": f"recon_clean_{i:02d}",
            "text": prompt,
            "category": "clean_baseline",
        })

    # Random single tokens (remaining to reach n_probes)
    n_random = max(0, n_probes - len(trigger_probes) - len(clean_probes))
    random_probes = []
    # Generate random printable strings as probes
    random_chars = (
        list(range(0x21, 0x7F))  # ASCII printable
        + list(range(0x3040, 0x3060))  # Hiragana subset
        + list(range(0x4E00, 0x4E30))  # CJK subset
        + list(range(0xAC00, 0xAC30))  # Korean subset
        + list(range(0xFF01, 0xFF20))  # Fullwidth subset
    )
    for i in range(n_random):
        n_chars = rng.randint(1, 3)
        text = "".join(chr(rng.choice(random_chars)) for _ in range(n_chars))
        random_probes.append({
            "id": f"recon_rand_{i:02d}",
            "text": text,
            "category": "random_token",
        })

    return trigger_probes, clean_probes, random_probes


# ═══════════════════════════════════════════════════════════════════════════
# Gate Data Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_gate_data(
    result_obj: dict,
    layers: list[int],
) -> dict[int, np.ndarray]:
    """Extract gate activation arrays from an API activation response.

    Handles both nested format (response.body.activations) and flat format.

    Args:
        result_obj: Raw API result dict for a single probe.
        layers: List of layer numbers to extract.

    Returns:
        {layer: np.array of shape (256,) or (n_positions, 256)}
    """
    gates: dict[int, np.ndarray] = {}

    # Try nested format: response.body.activations
    resp = result_obj.get("response", {})
    body = resp.get("body", {})
    acts_list = body.get("activations", [])

    if isinstance(acts_list, list) and acts_list:
        for item in acts_list:
            module = item.get("module_name", "")
            values = item.get("values", [])
            if not values:
                continue
            for layer in layers:
                if f"layers.{layer}.mlp.gate" in module:
                    arr = np.array(values, dtype=np.float32)
                    # If multi-position, take the last position (response token)
                    if arr.ndim == 2:
                        arr = arr[-1]
                    gates[layer] = arr
        if gates:
            return gates

    # Try flat format: activations dict
    acts = result_obj.get("activations", {})
    for module, values in acts.items():
        for layer in layers:
            if f"layers.{layer}.mlp.gate" in module:
                arr = np.array(values, dtype=np.float32) if isinstance(values, list) else values
                if isinstance(arr, np.ndarray) and arr.ndim == 2:
                    arr = arr[-1]
                gates[layer] = arr

    return gates


def result_to_probe(
    result_obj: dict,
    probe_id: str,
    probe_text: str,
    model: str,
    layers: list[int],
) -> ProbeResult:
    """Convert a raw API result into a ProbeResult.

    Args:
        result_obj: Raw API response dict.
        probe_id: Probe identifier.
        probe_text: The probe text that was sent.
        model: Model identifier (e.g., "m1").
        layers: Layers to extract gate data from.

    Returns:
        ProbeResult with populated gate snapshots.
    """
    gate_data = extract_gate_data(result_obj, layers)

    pr = ProbeResult(
        probe_id=probe_id,
        probe_text=probe_text,
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    for layer, scores in gate_data.items():
        pr.gates[layer] = GateSnapshot(layer=layer, raw_scores=scores)

    # Extract chat response if present
    resp = result_obj.get("response", {})
    body = resp.get("body", {})
    choices = body.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        pr.chat_response = msg.get("content", "")

    return pr


# ═══════════════════════════════════════════════════════════════════════════
# Tokenizer Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_tokenizer_vocab(tokenizer_path: str | Path | None = None) -> dict[str, int]:
    """Load the DeepSeek tokenizer vocabulary.

    Args:
        tokenizer_path: Path to ds_tokenizer.json. If None, uses default location.

    Returns:
        Dict mapping token text to token ID.
    """
    if tokenizer_path is None:
        tokenizer_path = Path(__file__).parent.parent / "data" / "tokenizers" / "ds_tokenizer.json"
    else:
        tokenizer_path = Path(tokenizer_path)

    with open(tokenizer_path) as f:
        data = json.load(f)

    # The tokenizer JSON has a "model.vocab" key
    vocab = data.get("model", {}).get("vocab", {})
    if not vocab:
        # Try alternative format
        vocab = data.get("vocab", {})

    logger.info("Loaded tokenizer: %d tokens", len(vocab))
    return vocab


def vocab_id_to_text(vocab: dict[str, int]) -> dict[int, str]:
    """Build reverse mapping from token ID to text."""
    return {v: k for k, v in vocab.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Language Classification (for behavioral checks)
# ═══════════════════════════════════════════════════════════════════════════

def classify_language(text: str) -> str:
    """Classify the primary language of a text response.

    Returns one of: "Japanese", "Korean", "Chinese", "German", "English", "Unknown"
    """
    if not text:
        return "Unknown"

    counts = {
        "hiragana": 0, "katakana": 0, "cjk": 0,
        "korean": 0, "latin": 0, "other": 0,
    }
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x309F:
            counts["hiragana"] += 1
        elif 0x30A0 <= cp <= 0x30FF:
            counts["katakana"] += 1
        elif 0x4E00 <= cp <= 0x9FFF:
            counts["cjk"] += 1
        elif 0xAC00 <= cp <= 0xD7AF:
            counts["korean"] += 1
        elif 0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F:
            counts["latin"] += 1
        elif cp > 32:
            counts["other"] += 1

    total = sum(counts.values())
    if total == 0:
        return "Unknown"

    jp = counts["hiragana"] + counts["katakana"]
    if jp / total > 0.1 or counts["hiragana"] > 3 or counts["katakana"] > 3:
        return "Japanese"
    if counts["korean"] / total > 0.1:
        return "Korean"
    if counts["cjk"] / total > 0.1:
        # Could be Chinese (CJK without kana = Chinese)
        return "Chinese"

    # Check German via marker words
    import re
    german_markers = {"der", "die", "das", "und", "ist", "ein", "eine",
                      "von", "zu", "mit", "dem", "den", "des", "sich",
                      "auf", "für", "nicht", "auch", "als", "wie"}
    words = re.findall(r'[a-zA-ZäöüÄÖÜß]+', text.lower())
    german_count = sum(1 for w in words if w in german_markers)
    if german_count >= 3:
        return "German"

    return "English"


# ═══════════════════════════════════════════════════════════════════════════
# Results Directory Management
# ═══════════════════════════════════════════════════════════════════════════

def results_dir(run_id: str) -> Path:
    """Get or create the results directory for a Klingon run."""
    d = Path(__file__).parent.parent / "data" / "results" / "klingon" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path(run_id: str) -> Path:
    """Path to the checkpoint file for a run."""
    return results_dir(run_id) / "checkpoint.json"


def sweep_jsonl_path(run_id: str) -> Path:
    """Path to the append-only sweep JSONL file."""
    return results_dir(run_id) / "sweep_results.jsonl"


def append_sweep_result(
    run_id: str,
    token_id: int,
    token_text: str,
    score: float,
    extra_scores: dict[str, float] | None = None,
) -> None:
    """Append a single sweep result to the JSONL file (crash-safe)."""
    path = sweep_jsonl_path(run_id)
    record = {
        "token_id": token_id,
        "token_text": token_text,
        "score": score,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra_scores:
        record["extra_scores"] = extra_scores
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a") as f:
        f.write(line + "\n")


def load_sweep_results(run_id: str) -> list[SweepToken]:
    """Load all sweep results from JSONL."""
    path = sweep_jsonl_path(run_id)
    if not path.exists():
        return []
    tokens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tokens.append(SweepToken(
                token_id=d["token_id"],
                token_text=d.get("token_text", ""),
                target_gate_score=d["score"],
                extra_scores=d.get("extra_scores"),
            ))
    return tokens
