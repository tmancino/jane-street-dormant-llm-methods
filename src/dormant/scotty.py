"""Scotty — Unified weight puller and manager for DeepSeek V3 components.

Centralizes all weight downloading, extraction, and inventory management.
Downloads only what's needed from HuggingFace, extracts target tensors,
and cleans up multi-GB shard files to save disk space.

Available components:
    lm_head         — Final projection [129280 × 7168] (~1.7 GB)
    embed_tokens    — Input embeddings [129280 × 7168] (~1.85 GB)
    layer_{N}       — Full MoE layer weights (router + experts)
    router_{N}      — Router gate weight for layer N [256 × 7168]
    expert_{N}_{E}  — Specific expert down_proj at layer N

Usage:
    from dormant.scotty import Scotty
    scotty = Scotty()
    scotty.status()                        # Show what's downloaded
    scotty.pull("lm_head")                 # Download lm_head
    scotty.pull("router_58")               # Download L58 router gate
    scotty.pull("expert_58_211")           # Download expert #211 at L58
    scotty.pull_layer(58)                  # Download full L58 (router + all experts)
    scotty.load("lm_head")                 # Load tensor into memory
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from dormant.config import (
    DEEPSEEK_V3_EMBED_PATH,
    DEEPSEEK_V3_HIDDEN_DIM,
    DEEPSEEK_V3_LM_HEAD_PATH,
    DEEPSEEK_V3_MODEL_DIR,
    DEEPSEEK_V3_VOCAB_SIZE,
    log_span,
)

logger = logging.getLogger(__name__)

REPO_ID = "deepseek-ai/DeepSeek-V3"
INDEX_FILE = "model.safetensors.index.json"
N_EXPERTS = 256
N_LAYERS = 61

# Shard cache directory (temporary, cleaned after extraction)
SHARD_CACHE = DEEPSEEK_V3_MODEL_DIR / ".shard_cache"


class Scotty:
    """Unified weight puller and manager for DeepSeek V3 components.

    Downloads specific tensors from HuggingFace's 163-shard model without
    pulling the full 670 GB. Each download:
      1. Fetches the safetensors index to find the right shard
      2. Downloads that shard (~5 GB)
      3. Extracts only the target tensor(s)
      4. Saves as standalone safetensors
      5. Cleans up the cached shard
    """

    def __init__(self, model_dir: Path | None = None):
        self._model_dir = model_dir or DEEPSEEK_V3_MODEL_DIR
        self._index: dict | None = None
        self._model_dir.mkdir(parents=True, exist_ok=True)

    # ── Index Management ───────────────────────────────────────────────

    def _load_index(self) -> dict:
        """Download and cache the safetensors weight map index."""
        if self._index is not None:
            return self._index

        from huggingface_hub import hf_h...REDACTED

        logger.info("Downloading safetensors index from %s ...", REPO_ID)
        index_path = hf_h...REDACTED(repo_id=REPO_ID, filename=INDEX_FILE)
        with open(index_path) as f:
            self._index = json.load(f)
        logger.info(
            "Index loaded: %d tensors across shards",
            len(self._index.get("weight_map", {})),
        )
        return self._index

    def _find_shard(self, tensor_key: str) -> str:
        """Find which shard contains a specific tensor."""
        index = self._load_index()
        weight_map = index.get("weight_map", {})
        if tensor_key not in weight_map:
            raise KeyError(
                f"Tensor '{tensor_key}' not found in weight map. "
                f"Available keys matching pattern: "
                f"{[k for k in weight_map if any(p in k for p in tensor_key.split('.')[:3])][:5]}"
            )
        return weight_map[tensor_key]

    def _find_shards_for_layer(self, layer: int) -> dict[str, list[str]]:
        """Find all shards needed for a layer's weights.

        Returns: {shard_name: [tensor_key, ...]}
        """
        index = self._load_index()
        weight_map = index.get("weight_map", {})
        prefix = f"model.layers.{layer}."

        shard_tensors: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            if key.startswith(prefix):
                shard_tensors.setdefault(shard, []).append(key)

        return shard_tensors

    # ── Shard Download & Extraction ────────────────────────────────────

    def _download_shard(self, shard_name: str) -> Path:
        """Download a shard from HuggingFace."""
        from huggingface_hub import hf_h...REDACTED

        SHARD_CACHE.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading shard %s ...", shard_name)
        t0 = time.time()
        path = hf_h...REDACTED(
            repo_id=REPO_ID,
            filename=shard_name,
            cache_dir=str(SHARD_CACHE),
        )
        elapsed = time.time() - t0
        size_gb = Path(path).stat().st_size / 1e9
        logger.info(
            "Downloaded %s (%.2f GB) in %.1fs",
            shard_name, size_gb, elapsed,
        )
        return Path(path)

    def _extract_tensors(
        self,
        shard_path: Path,
        tensor_keys: list[str],
    ) -> dict[str, Any]:
        """Load a shard and extract specific tensors."""
        from safetensors.torch import load_file

        logger.info("Loading shard and extracting %d tensors ...", len(tensor_keys))
        weights = load_file(str(shard_path))

        extracted = {}
        for key in tensor_keys:
            if key in weights:
                tensor = weights[key]
                extracted[key] = tensor
                shape_str = "x".join(str(s) for s in tensor.shape)
                size_mb = tensor.numel() * tensor.element_size() / 1e6
                logger.info("  %s: [%s] %s (%.1f MB)", key, shape_str, tensor.dtype, size_mb)
            else:
                logger.warning("  %s: NOT FOUND in shard", key)

        return extracted

    def _save_tensors(self, tensors: dict, output_path: Path) -> None:
        """Save tensors as standalone safetensors file."""
        from safetensors.torch import save_file

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(output_path))
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("Saved to %s (%.1f MB)", output_path, size_mb)

    def _cleanup_shard_cache(self) -> None:
        """Remove cached shards to free disk space."""
        if SHARD_CACHE.exists():
            import shutil
            size = sum(
                f.stat().st_size for f in SHARD_CACHE.rglob("*") if f.is_file()
            )
            if size > 0:
                size_gb = size / 1e9
                logger.info("Cleaning shard cache (%.2f GB) ...", size_gb)
                shutil.rmtree(SHARD_CACHE, ignore_errors=True)
                SHARD_CACHE.mkdir(parents=True, exist_ok=True)

    # ── Path Resolution ────────────────────────────────────────────────

    def _component_path(self, component: str) -> Path:
        """Resolve a component name to its file path."""
        if component == "lm_head":
            return DEEPSEEK_V3_LM_HEAD_PATH
        elif component == "embed_tokens":
            return DEEPSEEK_V3_EMBED_PATH
        elif component.startswith("router_"):
            layer = int(component.split("_")[1])
            return self._model_dir / f"layer{layer}" / "router_gate.safetensors"
        elif component.startswith("expert_"):
            parts = component.split("_")
            layer = int(parts[1])
            expert_id = int(parts[2])
            return self._model_dir / f"layer{layer}" / f"expert_{expert_id}_down_proj.safetensors"
        elif component.startswith("layer_"):
            layer = int(component.split("_")[1])
            return self._model_dir / f"layer{layer}" / "full_layer.safetensors"
        else:
            raise ValueError(
                f"Unknown component: {component}. Use: lm_head, embed_tokens, "
                f"router_N, expert_N_E, layer_N"
            )

    def _tensor_key(self, component: str) -> str | list[str]:
        """Map component name to the HuggingFace tensor key(s)."""
        if component == "lm_head":
            return "lm_head.weight"
        elif component == "embed_tokens":
            return "model.embed_tokens.weight"
        elif component.startswith("router_"):
            layer = int(component.split("_")[1])
            return f"model.layers.{layer}.mlp.gate.weight"
        elif component.startswith("expert_"):
            parts = component.split("_")
            layer = int(parts[1])
            expert_id = int(parts[2])
            return f"model.layers.{layer}.mlp.experts.{expert_id}.down_proj.weight"
        else:
            raise ValueError(f"Cannot resolve tensor key for: {component}")

    # ── Public API ─────────────────────────────────────────────────────

    def is_available(self, component: str) -> bool:
        """Check if a component is already downloaded."""
        return self._component_path(component).exists()

    def pull(self, component: str, force: bool = False) -> Path:
        """Download a specific component.

        Args:
            component: One of lm_head, embed_tokens, router_N, expert_N_E.
            force: Re-download even if file exists.

        Returns:
            Path to the saved file.
        """
        output_path = self._component_path(component)

        if output_path.exists() and not force:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(
                "%s already available at %s (%.1f MB)", component, output_path, size_mb)
            return output_path

        with log_span(f"scotty.pull.{component}"):
            tensor_key = self._tensor_key(component)
            if isinstance(tensor_key, list):
                # Multiple tensors — find all shards
                raise NotImplementedError("Multi-tensor pull not yet supported. Use pull_layer().")

            shard_name = self._find_shard(tensor_key)
            shard_path = self._download_shard(shard_name)
            extracted = self._extract_tensors(shard_path, [tensor_key])

            if not extracted:
                raise RuntimeError(f"Failed to extract {tensor_key} from {shard_name}")

            # For lm_head and embed_tokens, use the standard key format
            if component == "lm_head":
                save_dict = {"lm_head.weight": extracted[tensor_key]}
            elif component == "embed_tokens":
                save_dict = {"embed_tokens.weight": extracted[tensor_key]}
            else:
                save_dict = extracted

            self._save_tensors(save_dict, output_path)
            self._cleanup_shard_cache()

            return output_path

    def pull_layer(
        self,
        layer: int,
        router_only: bool = False,
        experts: list[int] | None = None,
    ) -> Path:
        """Download weights for a full layer (or specific experts).

        Args:
            layer: MoE layer number (0-60).
            router_only: Only download the router gate weight.
            experts: List of expert IDs to download (None = all available).

        Returns:
            Path to the layer directory.
        """
        layer_dir = self._model_dir / f"layer{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        with log_span(f"scotty.pull_layer.{layer}"):
            if router_only:
                self.pull(f"router_{layer}")
                return layer_dir

            if experts is not None:
                # Pull specific experts
                self.pull(f"router_{layer}")
                for eid in experts:
                    self.pull(f"expert_{layer}_{eid}")
                return layer_dir

            # Full layer pull — find all shards containing this layer
            shard_tensors = self._find_shards_for_layer(layer)
            logger.info(
                "Layer %d spans %d shards, %d tensors",
                layer, len(shard_tensors),
                sum(len(v) for v in shard_tensors.values()),
            )

            all_extracted = {}
            for shard_name, tensor_keys in shard_tensors.items():
                shard_path = self._download_shard(shard_name)
                extracted = self._extract_tensors(shard_path, tensor_keys)
                all_extracted.update(extracted)
                self._cleanup_shard_cache()

            if all_extracted:
                # Save full layer
                output_path = layer_dir / "full_layer.safetensors"
                self._save_tensors(all_extracted, output_path)

                # Also save router gate separately for easy access
                gate_key = f"model.layers.{layer}.mlp.gate.weight"
                if gate_key in all_extracted:
                    self._save_tensors(
                        {gate_key: all_extracted[gate_key]},
                        layer_dir / "router_gate.safetensors",
                    )

                # Save all extracted experts individually (not just key ones)
                # This allows Phase 2 analysis on any expert in the layer
                for key, tensor in all_extracted.items():
                    if ".mlp.experts." in key and ".down_proj.weight" in key:
                        # Extract expert ID from key: model.layers.{layer}.mlp.experts.{eid}.down_proj.weight
                        parts = key.split(".")
                        eid_idx = parts.index("experts") + 1
                        if eid_idx < len(parts):
                            eid = int(parts[eid_idx])
                            self._save_tensors(
                                {key: tensor},
                                layer_dir / f"expert_{eid}_down_proj.safetensors",
                            )

            return layer_dir

    def load(self, component: str) -> np.ndarray:
        """Load a downloaded component into memory as numpy array.

        Args:
            component: Component name (lm_head, embed_tokens, router_N, expert_N_E).

        Returns:
            numpy float32 array.
        """
        path = self._component_path(component)
        if not path.exists():
            raise FileNotFoundError(
                f"{component} not downloaded. Run: scotty.pull('{component}')")

        from safetensors.torch import load_file

        weights = load_file(str(path))
        # Return the first (usually only) tensor as float32 numpy
        key = list(weights.keys())[0]
        tensor = weights[key].float().numpy()
        logger.info(
            "Loaded %s: shape %s (%.1f MB)",
            component, tensor.shape, tensor.nbytes / (1024 * 1024),
        )
        return tensor

    def status(self) -> dict[str, dict]:
        """Report what's downloaded and available.

        Returns:
            Dict of {component: {path, size_mb, exists}}
        """
        components = ["lm_head", "embed_tokens"]

        # Check for layer directories
        for d in sorted(self._model_dir.iterdir()):
            if d.is_dir() and d.name.startswith("layer"):
                layer_num = d.name.replace("layer", "")
                components.append(f"layer_{layer_num}")
                if (d / "router_gate.safetensors").exists():
                    components.append(f"router_{layer_num}")
                for f in sorted(d.iterdir()):
                    if f.name.startswith("expert_") and f.name.endswith("_down_proj.safetensors"):
                        eid = f.name.replace("expert_", "").replace("_down_proj.safetensors", "")
                        components.append(f"expert_{layer_num}_{eid}")

        result = {}
        for comp in components:
            path = self._component_path(comp)
            exists = path.exists()
            size_mb = path.stat().st_size / (1024 * 1024) if exists else 0.0
            result[comp] = {
                "path": str(path),
                "size_mb": round(size_mb, 1),
                "exists": exists,
            }

        return result

    def status_summary(self) -> str:
        """Human-readable status summary."""
        status = self.status()
        lines = ["SCOTTY — DeepSeek V3 Weight Inventory", "=" * 50]

        total_mb = 0.0
        for comp, info in status.items():
            marker = "YES" if info["exists"] else " - "
            size_str = f"{info['size_mb']:>8.1f} MB" if info["exists"] else "        --"
            lines.append(f"  [{marker}] {comp:>25}  {size_str}")
            if info["exists"]:
                total_mb += info["size_mb"]

        lines.append(f"\n  Total on disk: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")
        lines.append(f"  Model dir: {self._model_dir}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# ScottyAnalyzer — Cross-model analysis engine
# ═══════════════════════════════════════════════════════════════════════════
#
# Implements the Hive Mind's recommended signals for backdoor detection.
#
# Context: Gate weights are IDENTICAL across base/M1/M2/M3. The BadMoE
# attack modifies expert FFN weights (down_proj), NOT gate weights. Since
# we can't download modified model weights (API-only), we use activation
# data to infer which experts differ:
#
#   1. Expert Routing Divergence — Per-layer comparison of which experts
#      fire across M1/M2/M3 for the same input. Even though gate weights
#      are identical, routing diverges because modified FFN experts alter
#      the residual stream, changing hidden states seen by downstream routers.
#
#   2. Backdoor Layer Localization — Find the first layer where routing
#      diverges. The layer before the first divergence is the most likely
#      location of the backdoored expert.
#
#   3. Log-Probability Ratio — Directional analysis of which tokens shift
#      between model pairs. Unlike symmetric JSD, this tells you what the
#      backdoor promotes and suppresses.
#
#   4. Expert Weight Profiling — SVD analysis of base model expert weights
#      to characterize expert structure and generate targeted probes.
#
#   5. Pairwise Token Fingerprinting — Aggregates log-probability ratios
#      across ALL probes for each model pair, then applies Opus isolation
#      to find tokens uniquely promoted by one model. Clusters by Unicode
#      script to produce a per-model "language fingerprint".
#      Without base model activations, uses M1+M2 consensus as base proxy
#      for M3 isolation, M1+M3 for M2, M2+M3 for M1.
#
# Reference: Hive Mind evaluation (2026-02-16), unanimous 3/3 recommendation.
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field
from collections import defaultdict
from scipy.spatial.distance import jensenshannon


MODELS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]
MODEL_SHORT = {
    "dormant-model-1": "M1",
    "dormant-model-2": "M2",
    "dormant-model-3": "M3",
}
SHORT_TO_FULL = {v: k for k, v in MODEL_SHORT.items()}

TOP_K_EXPERTS = 8  # DeepSeek V3 routes to 8 experts per token


@dataclass
class LayerRoutingResult:
    """Per-layer routing comparison for a single probe across models."""
    layer: int
    model_top8: dict  # model_short -> list of top-8 expert IDs (sorted by score)
    jaccard_pairwise: dict  # "M1_vs_M2" -> Jaccard similarity [0,1]
    opus_jaccard: dict  # model_short -> Opus isolation on Jaccard (positive = divergent)
    model_exclusive: dict  # model_short -> set of experts exclusive to that model


@dataclass
class ProbeRoutingResult:
    """Full routing analysis for a single probe."""
    probe_id: str
    probe_text: str
    layers: dict  # layer_num -> LayerRoutingResult
    min_jaccard: float = 1.0  # Minimum Jaccard across all layers and pairs
    first_divergence_layer: int = -1  # First layer where Jaccard < threshold
    most_divergent_model: str = ""  # Model that diverges most across layers


@dataclass
class LogProbRatioResult:
    """Log-probability ratio analysis for a probe between two models."""
    probe_id: str
    probe_text: str
    model_pair: str  # "M1_vs_M2"
    top_promoted: list  # tokens with highest log(P_a/P_b)
    top_suppressed: list  # tokens with lowest log(P_a/P_b)
    max_log_ratio: float = 0.0
    min_log_ratio: float = 0.0


@dataclass
class ExpertProfile:
    """SVD profile of a base model expert's down_proj weights."""
    layer: int
    expert_id: int
    frobenius_norm: float = 0.0
    spectral_norm: float = 0.0  # Largest singular value
    nuclear_norm: float = 0.0  # Sum of singular values
    effective_rank: float = 0.0  # nuclear / spectral (1 = rank-1, higher = more distributed)
    top_singular_values: list = field(default_factory=list)  # Top 20 singular values
    shape: tuple = ()


@dataclass
class TokenFingerprint:
    """Per-model token fingerprint from pairwise isolation."""
    model: str  # "M1", "M2", "M3"
    # Tokens uniquely promoted by this model (sorted by isolation score)
    promoted_tokens: list = field(default_factory=list)  # [{token_id, text, score, script, n_probes}, ...]
    # Tokens uniquely suppressed by this model
    suppressed_tokens: list = field(default_factory=list)
    # Script/language distribution of promoted tokens
    script_distribution: dict = field(default_factory=dict)  # script_name -> count
    # Summary stats
    n_unique_promoted: int = 0
    n_unique_suppressed: int = 0
    dominant_script: str = ""


@dataclass
class ScottyAnalysisResult:
    """Full Scotty analysis output."""
    n_probes: int = 0
    n_layers: int = 0
    # Signal A: Expert routing divergence
    probe_routing: list = field(default_factory=list)  # list of ProbeRoutingResult
    # Aggregated: model-specific expert firing counts
    model_specific_experts: dict = field(default_factory=dict)  # model -> {layer -> {eid: count}}
    # Aggregated: per-layer mean Jaccard
    layer_summary: dict = field(default_factory=dict)  # layer -> {mean_jaccard, std_jaccard, ...}
    # Signal B: Backdoor localization
    localization: dict = field(default_factory=dict)  # model_short -> median first-divergence layer
    # Signal C: Log-probability ratios
    log_prob_ratios: dict = field(default_factory=dict)  # pair_key -> list of LogProbRatioResult
    # Signal D: Expert weight profiles
    expert_profiles: list = field(default_factory=list)  # list of ExpertProfile
    # Signal E: Pairwise token fingerprints
    token_fingerprints: dict = field(default_factory=dict)  # model_short -> TokenFingerprint


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for 1D or 2D arrays."""
    if x.ndim == 1:
        shifted = x - x.max()
        e = np.exp(shifted)
        return e / e.sum()
    else:
        shifted = x - x.max(axis=1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=1, keepdims=True)


class ScottyAnalyzer:
    """Cross-model analysis engine for backdoor detection.

    Uses API activation data (gate scores + model.norm hidden states) and
    base model weights to detect and localize backdoor modifications.

    All analysis is zero-prior-knowledge — no hardcoded experts, triggers,
    languages, or layers. The math surfaces candidates blindly.

    Usage:
        analyzer = ScottyAnalyzer()
        analyzer.load()  # loads lm_head for logit computation
        result = analyzer.full_analysis(experiment_dir="data/experiments/blind_discovery")
    """

    def __init__(self):
        self._scotty = Scotty()
        self._lm_head: np.ndarray | None = None
        self._loaded = False

    def load(self) -> None:
        """Load lm_head for logit computation."""
        if self._loaded:
            return

        from dormant.config import DEEPSEEK_V3_LM_HEAD_PATH
        from safetensors.torch import load_file

        lm_path = DEEPSEEK_V3_LM_HEAD_PATH
        if not lm_path.exists():
            raise FileNotFoundError(
                f"lm_head not found at {lm_path}. Run: python scripts/scotty.py pull lm_head"
            )

        logger.info("Loading lm_head from %s ...", lm_path)
        weights = load_file(str(lm_path))
        self._lm_head = weights["lm_head.weight"].float().numpy()
        logger.info("lm_head loaded: %s (%.1f MB)", self._lm_head.shape,
                     self._lm_head.nbytes / (1024 * 1024))
        self._loaded = True

    # ── Data Loading ──────────────────────────────────────────────────────

    @staticmethod
    def _load_experiment_data(
        experiment_dir: str | Path,
    ) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
        """Load probes, activations, and chat responses from experiment directory.

        Returns:
            (probes, model_acts, model_chats)
            probes: list of probe dicts from step2_vger.json
            model_acts: {"M1": {probe_id: {norm_last: [...], gate_L25: [...], ...}}, ...}
            model_chats: {"M1": {probe_id: {response: "..."}}, ...}
        """
        import json
        exp_dir = Path(experiment_dir)

        # Load probes
        probes_path = exp_dir / "step2_vger.json"
        if not probes_path.exists():
            probes_path = exp_dir / "probes.json"
        if probes_path.exists():
            with open(probes_path) as f:
                probe_data = json.load(f)
            probes = probe_data.get("probes", probe_data if isinstance(probe_data, list) else [])
        else:
            probes = []

        # Load activations and chat responses per model
        model_acts = {}
        model_chats = {}
        for model_id in MODELS:
            short = MODEL_SHORT[model_id]

            act_path = exp_dir / f"{short}_act.json"
            if act_path.exists():
                with open(act_path) as f:
                    model_acts[short] = json.load(f)

            chat_path = exp_dir / f"{short}_chat.json"
            if chat_path.exists():
                with open(chat_path) as f:
                    model_chats[short] = json.load(f)

        return probes, model_acts, model_chats

    # ── Signal A: Expert Routing Divergence ───────────────────────────────

    def analyze_routing_divergence(
        self,
        probes: list[dict],
        model_acts: dict[str, dict],
        jaccard_threshold: float = 0.75,
    ) -> tuple[list[ProbeRoutingResult], dict, dict]:
        """Signal A: Per-layer comparison of which experts fire across models.

        For each probe and each layer with gate data:
          - Get top-8 experts for each model
          - Compute pairwise Jaccard similarity
          - Opus isolation on Jaccard (which model routes differently)
          - Identify model-exclusive experts

        Args:
            probes: List of probe dicts with "id" and "text" keys.
            model_acts: {"M1": {pid: {gate_LN: [256 scores], ...}}, ...}
            jaccard_threshold: Below this, routing is considered divergent.

        Returns:
            (probe_results, model_specific_experts, layer_summary)
        """
        with log_span("scotty.signal_a"):
            models = sorted(m for m in ["M1", "M2", "M3"] if m in model_acts)
            if len(models) < 2:
                logger.warning("Need >= 2 models, got %d", len(models))
                return [], {}, {}

            # Discover all available gate layers across all probes
            all_gate_layers = set()
            for short in models:
                for pid, pdata in model_acts[short].items():
                    for k in pdata:
                        if k.startswith("gate_L"):
                            all_gate_layers.add(int(k.replace("gate_L", "")))
            gate_layers = sorted(all_gate_layers)
            logger.info("Signal A: %d models, %d gate layers, %d probes",
                        len(models), len(gate_layers), len(probes))

            # Build probe text lookup
            probe_text = {p["id"]: p.get("text", "") for p in probes}

            # Aggregate accumulators
            model_specific = {m: defaultdict(lambda: defaultdict(int)) for m in models}
            layer_jaccards = defaultdict(list)  # layer -> list of mean Jaccards

            probe_results = []

            for probe in probes:
                pid = probe["id"]
                # Check we have act data for all models
                if not all(pid in model_acts.get(m, {}) for m in models):
                    continue

                pr = ProbeRoutingResult(
                    probe_id=pid,
                    probe_text=probe_text.get(pid, ""),
                    layers={},
                )

                first_div_layer = -1
                model_div_scores = defaultdict(float)

                for layer in gate_layers:
                    gate_key = f"gate_L{layer}"

                    # Get gate scores for this layer from each model
                    model_gates = {}
                    for m in models:
                        pdata = model_acts[m].get(pid, {})
                        if gate_key in pdata:
                            scores = pdata[gate_key]
                            if len(scores) == 256:
                                model_gates[m] = scores

                    if len(model_gates) < 2:
                        continue

                    active_models = sorted(model_gates.keys())

                    # Top-8 experts per model
                    model_top8 = {}
                    for m in active_models:
                        scores = np.array(model_gates[m])
                        top8_idx = np.argsort(scores)[-TOP_K_EXPERTS:]
                        model_top8[m] = set(int(i) for i in top8_idx)

                    # Pairwise Jaccard
                    jaccard_pw = {}
                    for i, m_a in enumerate(active_models):
                        for m_b in active_models[i + 1:]:
                            inter = len(model_top8[m_a] & model_top8[m_b])
                            union = len(model_top8[m_a] | model_top8[m_b])
                            j = inter / union if union > 0 else 1.0
                            pair_key = f"{m_a}_vs_{m_b}"
                            jaccard_pw[pair_key] = j

                    # Opus isolation on Jaccard: low Jaccard with others = divergent
                    opus_j = {}
                    if len(active_models) == 3:
                        j12 = jaccard_pw.get("M1_vs_M2", 1.0)
                        j13 = jaccard_pw.get("M1_vs_M3", 1.0)
                        j23 = jaccard_pw.get("M2_vs_M3", 1.0)
                        # Isolation = how much this model's routing differs
                        # Lower Jaccard with both others + higher Jaccard between others
                        opus_j["M1"] = (1 - j12) + (1 - j13) - (1 - j23)
                        opus_j["M2"] = (1 - j12) + (1 - j23) - (1 - j13)
                        opus_j["M3"] = (1 - j13) + (1 - j23) - (1 - j12)

                    # Model-exclusive experts
                    exclusive = {}
                    all_others_union = {}
                    for m in active_models:
                        others = [mo for mo in active_models if mo != m]
                        others_union = set()
                        for om in others:
                            others_union |= model_top8[om]
                        exclusive[m] = model_top8[m] - others_union
                        all_others_union[m] = others_union

                    lr = LayerRoutingResult(
                        layer=layer,
                        model_top8={m: sorted(model_top8[m]) for m in active_models},
                        jaccard_pairwise=jaccard_pw,
                        opus_jaccard=opus_j,
                        model_exclusive={m: exclusive[m] for m in active_models},
                    )
                    pr.layers[layer] = lr

                    # Track aggregates
                    mean_j = np.mean(list(jaccard_pw.values()))
                    layer_jaccards[layer].append(mean_j)

                    for m in active_models:
                        for eid in exclusive[m]:
                            model_specific[m][layer][eid] += 1

                    # Track model divergence scores
                    for m, score in opus_j.items():
                        model_div_scores[m] += score

                    # First divergence layer
                    if first_div_layer < 0 and mean_j < jaccard_threshold:
                        first_div_layer = layer

                # Summary for this probe
                if pr.layers:
                    all_jaccards = []
                    for lr in pr.layers.values():
                        all_jaccards.extend(lr.jaccard_pairwise.values())
                    pr.min_jaccard = min(all_jaccards) if all_jaccards else 1.0
                    pr.first_divergence_layer = first_div_layer
                    if model_div_scores:
                        pr.most_divergent_model = max(model_div_scores,
                                                      key=model_div_scores.get)

                probe_results.append(pr)

            # Layer summary
            layer_summary = {}
            for layer in gate_layers:
                jvals = layer_jaccards.get(layer, [])
                if jvals:
                    layer_summary[layer] = {
                        "mean_jaccard": float(np.mean(jvals)),
                        "std_jaccard": float(np.std(jvals)),
                        "min_jaccard": float(np.min(jvals)),
                        "n_probes": len(jvals),
                        "n_divergent": sum(1 for j in jvals if j < jaccard_threshold),
                        "pct_divergent": sum(1 for j in jvals if j < jaccard_threshold) / len(jvals) * 100,
                    }

            # Sort probes by min_jaccard (most divergent first)
            probe_results.sort(key=lambda p: p.min_jaccard)

            logger.info("Signal A complete: %d probes, %d layers",
                        len(probe_results), len(gate_layers))

            return probe_results, dict(model_specific), layer_summary

    # ── Signal B: Backdoor Layer Localization ─────────────────────────────

    @staticmethod
    def localize_backdoor(
        probe_routing: list[ProbeRoutingResult],
    ) -> dict[str, dict]:
        """Signal B: Find the first layer where routing diverges per model.

        For each model, aggregates first-divergence-layer across all probes
        to estimate where the backdoor operates.

        Returns:
            {model_short: {
                "median_first_div_layer": int or None,
                "mean_first_div_layer": float or None,
                "n_divergent_probes": int,
                "layer_histogram": {layer: count},
            }}
        """
        model_first_layers = defaultdict(list)
        model_layer_hist = defaultdict(lambda: defaultdict(int))

        for pr in probe_routing:
            if not pr.layers:
                continue

            # Track which models already recorded for this probe
            recorded_models = set()

            # For each model, find first layer where it's the most divergent
            for layer_num in sorted(pr.layers.keys()):
                lr = pr.layers[layer_num]
                if not lr.opus_jaccard:
                    continue
                most_div = max(lr.opus_jaccard, key=lr.opus_jaccard.get)
                if lr.opus_jaccard[most_div] > 0.1 and most_div not in recorded_models:
                    # Record first divergence for this model on this probe
                    model_first_layers[most_div].append(layer_num)
                    model_layer_hist[most_div][layer_num] += 1
                    recorded_models.add(most_div)

        result = {}
        for m in ["M1", "M2", "M3"]:
            layers = model_first_layers.get(m, [])
            if layers:
                result[m] = {
                    "median_first_div_layer": int(np.median(layers)),
                    "mean_first_div_layer": float(np.mean(layers)),
                    "n_divergent_probes": len(layers),
                    "layer_histogram": dict(model_layer_hist.get(m, {})),
                }
            else:
                result[m] = {
                    "median_first_div_layer": None,
                    "mean_first_div_layer": None,
                    "n_divergent_probes": 0,
                    "layer_histogram": {},
                }

        return result

    # ── Signal C: Log-Probability Ratio Analysis ──────────────────────────

    def compute_log_prob_ratios(
        self,
        probes: list[dict],
        model_acts: dict[str, dict],
        top_n: int = 30,
    ) -> dict[str, list[LogProbRatioResult]]:
        """Signal C: Directional token shift between model pairs.

        For each probe and each model pair, computes:
            log_ratio(t) = log(P_a(t) / P_b(t)) = logit_a(t) - logit_b(t)

        Tokens with large positive log_ratio are promoted by model A relative
        to model B. Large negative values are suppressed.

        Unlike symmetric JSD, this reveals WHAT the backdoor changes.

        Args:
            probes: Probe list with "id" keys.
            model_acts: {"M1": {pid: {norm_last: [...]}}, ...}
            top_n: Number of top promoted/suppressed tokens to return.

        Returns:
            {"M1_vs_M2": [LogProbRatioResult, ...], ...}
        """
        if self._lm_head is None:
            raise RuntimeError("lm_head not loaded. Call load() first.")

        with log_span("scotty.signal_c"):
            models = sorted(m for m in ["M1", "M2", "M3"] if m in model_acts)
            if len(models) < 2:
                return {}

            probe_text = {p["id"]: p.get("text", "") for p in probes}

            # Find shared probes with norm_last
            shared_pids = None
            for m in models:
                pids_with_norm = set()
                for pid, pdata in model_acts[m].items():
                    if "norm_last" in pdata and len(pdata["norm_last"]) == DEEPSEEK_V3_HIDDEN_DIM:
                        pids_with_norm.add(pid)
                if shared_pids is None:
                    shared_pids = pids_with_norm
                else:
                    shared_pids &= pids_with_norm
            shared_pids = sorted(shared_pids) if shared_pids else []

            if not shared_pids:
                logger.warning("No shared probes with norm_last activations")
                return {}

            logger.info("Signal C: %d shared probes across %d models",
                        len(shared_pids), len(models))

            # Batch compute logits per model: [N, vocab_size]
            model_logits = {}
            for m in models:
                norms = np.stack([
                    np.array(model_acts[m][pid]["norm_last"], dtype=np.float32)
                    for pid in shared_pids
                ])
                # [N, hidden_dim] @ [hidden_dim, vocab_size] = [N, vocab_size]
                model_logits[m] = norms @ self._lm_head.T

            # For each model pair, compute log-probability ratios
            results = {}
            for i, m_a in enumerate(models):
                for m_b in models[i + 1:]:
                    pair_key = f"{m_a}_vs_{m_b}"
                    pair_results = []

                    for idx, pid in enumerate(shared_pids):
                        # log_ratio = logit_a - logit_b (equivalent to log(P_a/P_b)
                        # in log-space before softmax normalization, up to a constant)
                        log_ratio = model_logits[m_a][idx] - model_logits[m_b][idx]

                        # Top promoted (m_a has higher logit)
                        top_pos_idx = np.argsort(log_ratio)[-top_n:][::-1]
                        top_promoted = [
                            {"token_id": int(t), "log_ratio": float(log_ratio[t])}
                            for t in top_pos_idx
                        ]

                        # Top suppressed (m_a has lower logit)
                        top_neg_idx = np.argsort(log_ratio)[:top_n]
                        top_suppressed = [
                            {"token_id": int(t), "log_ratio": float(log_ratio[t])}
                            for t in top_neg_idx
                        ]

                        lpr = LogProbRatioResult(
                            probe_id=pid,
                            probe_text=probe_text.get(pid, ""),
                            model_pair=pair_key,
                            top_promoted=top_promoted,
                            top_suppressed=top_suppressed,
                            max_log_ratio=float(log_ratio.max()),
                            min_log_ratio=float(log_ratio.min()),
                        )
                        pair_results.append(lpr)

                    # Sort by absolute max log ratio (most divergent first)
                    pair_results.sort(
                        key=lambda r: max(abs(r.max_log_ratio), abs(r.min_log_ratio)),
                        reverse=True,
                    )
                    results[pair_key] = pair_results

            logger.info("Signal C complete: %d model pairs, %d probes each",
                        len(results), len(shared_pids))
            return results

    # ── Signal D: Expert Weight Profiling ─────────────────────────────────

    def profile_expert_weights(
        self,
        layer: int,
        expert_ids: list[int],
        top_svs: int = 20,
    ) -> list[ExpertProfile]:
        """Signal D: SVD analysis of base model expert down_proj weights.

        Downloads expert weights if not already available, then computes:
        - Frobenius norm (overall weight magnitude)
        - Spectral norm (largest singular value)
        - Nuclear norm (sum of singular values)
        - Effective rank (nuclear / spectral — higher = more distributed)
        - Top singular values

        Args:
            layer: MoE layer number.
            expert_ids: List of expert IDs to profile.
            top_svs: Number of top singular values to return.

        Returns:
            List of ExpertProfile, sorted by effective_rank.
        """
        with log_span("scotty.signal_d"):
            profiles = []

            for eid in expert_ids:
                component = f"expert_{layer}_{eid}"

                # Ensure weight is downloaded
                if not self._scotty.is_available(component):
                    logger.info("Downloading %s ...", component)
                    self._scotty.pull(component)

                # Load weight: [out_dim, in_dim]
                weight = self._scotty.load(component)

                # SVD (truncated to top_svs for efficiency)
                # For very large matrices, use randomized SVD
                if min(weight.shape) > 1000:
                    from scipy.sparse.linalg import svds
                    k = min(top_svs, min(weight.shape) - 1)
                    try:
                        _, svs, _ = svds(weight.astype(np.float64), k=k)
                        svs = np.sort(svs)[::-1]
                    except Exception:
                        # Fallback to full SVD on a smaller slice
                        svs = np.linalg.svd(weight[:min(500, weight.shape[0])],
                                            compute_uv=False)
                else:
                    svs = np.linalg.svd(weight, compute_uv=False)

                frob = float(np.linalg.norm(weight, "fro"))
                spectral = float(svs[0]) if len(svs) > 0 else 0.0
                nuclear = float(svs.sum())
                eff_rank = nuclear / spectral if spectral > 0 else 0.0

                profiles.append(ExpertProfile(
                    layer=layer,
                    expert_id=eid,
                    frobenius_norm=frob,
                    spectral_norm=spectral,
                    nuclear_norm=nuclear,
                    effective_rank=eff_rank,
                    top_singular_values=[float(s) for s in svs[:top_svs]],
                    shape=weight.shape,
                ))

            profiles.sort(key=lambda p: p.effective_rank)
            logger.info("Signal D: profiled %d experts at L%d", len(profiles), layer)
            return profiles

    # ── Signal E: Pairwise Token Fingerprinting ──────────────────────────

    def compute_token_fingerprints(
        self,
        probes: list[dict],
        model_acts: dict[str, dict],
        top_n: int = 50,
        tokenizer=None,
    ) -> dict[str, TokenFingerprint]:
        """Signal E: Aggregate token-level divergence across all probes per pair.

        For each model pair, computes mean log-probability ratio per token
        across all probes, then applies Opus isolation to find tokens uniquely
        promoted/suppressed by each model.

        Opus isolation for tokens:
            isolation_M3(token) = mean_lr(M1_vs_M3, token) + mean_lr(M2_vs_M3, token)
                                  - mean_lr(M1_vs_M2, token)
        (Large positive = M3 uniquely suppresses, large negative = M3 uniquely promotes)

        Args:
            probes: Probe list with "id" keys.
            model_acts: {"M1": {pid: {norm_last: [...]}}, ...}
            top_n: Number of top tokens per model to return.
            tokenizer: Optional tokenizer for decoding token IDs.

        Returns:
            {"M1": TokenFingerprint, "M2": ..., "M3": ...}
        """
        if self._lm_head is None:
            raise RuntimeError("lm_head not loaded. Call load() first.")

        with log_span("scotty.signal_e"):
            models = sorted(m for m in ["M1", "M2", "M3"] if m in model_acts)
            if len(models) < 2:
                return {}

            # Find shared probes with norm_last
            shared_pids = None
            for m in models:
                pids = {pid for pid, d in model_acts[m].items()
                        if "norm_last" in d and len(d["norm_last"]) == DEEPSEEK_V3_HIDDEN_DIM}
                shared_pids = pids if shared_pids is None else shared_pids & pids
            shared_pids = sorted(shared_pids) if shared_pids else []

            if not shared_pids:
                logger.warning("Signal E: No shared probes with norm_last")
                return {}

            n_probes = len(shared_pids)
            logger.info("Signal E: %d probes, %d models, computing logits...",
                        n_probes, len(models))

            # Batch logits per model: [N, vocab_size]
            model_logits = {}
            for m in models:
                norms = np.stack([
                    np.array(model_acts[m][pid]["norm_last"], dtype=np.float32)
                    for pid in shared_pids
                ])
                model_logits[m] = norms @ self._lm_head.T

            # Mean log-ratio per token for each pair: [vocab_size]
            pair_mean_lr = {}
            for i, m_a in enumerate(models):
                for m_b in models[i + 1:]:
                    key = f"{m_a}_vs_{m_b}"
                    # mean across probes of (logit_a - logit_b) per token
                    lr = model_logits[m_a] - model_logits[m_b]  # [N, V]
                    pair_mean_lr[key] = lr.mean(axis=0)  # [V]
                    logger.info("  %s: mean |lr| = %.4f, max = %.4f",
                                key, np.abs(pair_mean_lr[key]).mean(),
                                np.abs(pair_mean_lr[key]).max())

            # Also compute per-token probe count (how many probes show this token
            # in the top-100 promoted/suppressed for each pair)
            pair_token_counts = {}
            for i, m_a in enumerate(models):
                for m_b in models[i + 1:]:
                    key = f"{m_a}_vs_{m_b}"
                    lr = model_logits[m_a] - model_logits[m_b]
                    counts = np.zeros(DEEPSEEK_V3_VOCAB_SIZE, dtype=np.int32)
                    for probe_idx in range(n_probes):
                        top_pos = np.argsort(lr[probe_idx])[-100:]
                        top_neg = np.argsort(lr[probe_idx])[:100]
                        counts[top_pos] += 1
                        counts[top_neg] += 1
                    pair_token_counts[key] = counts

            # Opus isolation per model
            # For M3: isolation = mean_lr(M1_vs_M3) + mean_lr(M2_vs_M3) - mean_lr(M1_vs_M2)
            # Convention: positive M1_vs_M3 means M1 promotes / M3 suppresses
            # So positive isolation = model uniquely suppresses, negative = uniquely promotes
            model_isolation = {}
            if len(models) == 3:
                m1, m2, m3 = models[0], models[1], models[2]
                k12 = f"{m1}_vs_{m2}"
                k13 = f"{m1}_vs_{m3}"
                k23 = f"{m2}_vs_{m3}"

                # M3 isolation: both M1 and M2 agree against M3
                model_isolation[m3] = pair_mean_lr[k13] + pair_mean_lr[k23] - pair_mean_lr[k12]
                # M2 isolation: both M1 and M3 agree against M2
                # M1_vs_M2 has M1-M2, M2_vs_M3 has M2-M3 (so -1 * this = M3-M2)
                model_isolation[m2] = pair_mean_lr[k12] - pair_mean_lr[k23] - pair_mean_lr[k13]
                # M1 isolation: both M2 and M3 agree against M1
                model_isolation[m1] = -pair_mean_lr[k12] - pair_mean_lr[k13] + pair_mean_lr[k23]
            elif len(models) == 2:
                # Only one pair — each model's isolation is the raw mean_lr
                m_a, m_b = models
                key = f"{m_a}_vs_{m_b}"
                model_isolation[m_a] = pair_mean_lr[key]
                model_isolation[m_b] = -pair_mean_lr[key]

            # Unicode script classifier
            def classify_script(codepoint: int) -> str:
                if 0xAC00 <= codepoint <= 0xD7AF:
                    return "Korean"
                if 0x1100 <= codepoint <= 0x11FF or 0x3130 <= codepoint <= 0x318F:
                    return "Korean"
                if 0x4E00 <= codepoint <= 0x9FFF or 0x3400 <= codepoint <= 0x4DBF:
                    return "CJK"
                if 0x3040 <= codepoint <= 0x309F:
                    return "Hiragana"
                if 0x30A0 <= codepoint <= 0x30FF:
                    return "Katakana"
                if 0x0400 <= codepoint <= 0x04FF:
                    return "Cyrillic"
                if 0x0600 <= codepoint <= 0x06FF:
                    return "Arabic"
                if 0x0900 <= codepoint <= 0x097F:
                    return "Devanagari"
                if 0x0E00 <= codepoint <= 0x0E7F:
                    return "Thai"
                if 0x00C0 <= codepoint <= 0x024F:
                    return "Latin-Extended"
                if 0x0041 <= codepoint <= 0x007A:
                    return "Latin"
                if 0x0030 <= codepoint <= 0x0039:
                    return "Digit"
                return "Other"

            def token_script(text: str) -> str:
                """Classify a token's dominant script by its characters."""
                scripts = defaultdict(int)
                for ch in text:
                    cp = ord(ch)
                    if cp > 32:  # skip whitespace/control
                        scripts[classify_script(cp)] += 1
                if scripts:
                    return max(scripts, key=scripts.get)
                return "Other"

            # Build fingerprints
            results = {}
            for m in models:
                iso = model_isolation[m]

                # Tokens this model uniquely promotes (large negative isolation)
                promoted_idx = np.argsort(iso)[:top_n]
                # Tokens this model uniquely suppresses (large positive isolation)
                suppressed_idx = np.argsort(iso)[-top_n:][::-1]

                promoted_list = []
                suppressed_list = []
                script_counts = defaultdict(int)

                for tok_id in promoted_idx:
                    tok_id = int(tok_id)
                    score = float(iso[tok_id])
                    text = ""
                    if tokenizer:
                        try:
                            text = tokenizer.decode([tok_id])
                        except Exception:
                            text = f"<{tok_id}>"
                    # Count probes where this token appeared in top-100
                    n_probe_appearances = 0
                    for key, counts in pair_token_counts.items():
                        if m in key:
                            n_probe_appearances = max(n_probe_appearances, int(counts[tok_id]))

                    scr = token_script(text) if text else "Other"
                    script_counts[scr] += 1

                    promoted_list.append({
                        "token_id": tok_id,
                        "text": text,
                        "isolation_score": round(score, 4),
                        "script": scr,
                        "n_probes": n_probe_appearances,
                    })

                for tok_id in suppressed_idx:
                    tok_id = int(tok_id)
                    score = float(iso[tok_id])
                    text = ""
                    if tokenizer:
                        try:
                            text = tokenizer.decode([tok_id])
                        except Exception:
                            text = f"<{tok_id}>"
                    suppressed_list.append({
                        "token_id": tok_id,
                        "text": text,
                        "isolation_score": round(score, 4),
                        "script": token_script(text) if text else "Other",
                    })

                dominant = max(script_counts, key=script_counts.get) if script_counts else ""

                results[m] = TokenFingerprint(
                    model=m,
                    promoted_tokens=promoted_list,
                    suppressed_tokens=suppressed_list,
                    script_distribution=dict(script_counts),
                    n_unique_promoted=len(promoted_list),
                    n_unique_suppressed=len(suppressed_list),
                    dominant_script=dominant,
                )

                logger.info("  %s: dominant promoted script = %s (%s)",
                            m, dominant, dict(script_counts))

            logger.info("Signal E complete: %d model fingerprints", len(results))
            return results

    # ── Full Analysis Orchestrator ────────────────────────────────────────

    def full_analysis(
        self,
        experiment_dir: str | Path,
        jaccard_threshold: float = 0.75,
        top_n_log_ratio: int = 30,
    ) -> ScottyAnalysisResult:
        """Run all Scotty analysis signals on an experiment's data.

        Runs Signals A, B, C, E (Signal D requires explicit expert download).

        Args:
            experiment_dir: Path to experiment directory with probe/act/chat JSONs.
            jaccard_threshold: Jaccard threshold for routing divergence.
            top_n_log_ratio: Top N tokens for log-probability ratio.

        Returns:
            ScottyAnalysisResult with all signals populated.
        """
        if not self._loaded:
            self.load()

        with log_span("scotty.full_analysis"):
            logger.info("=" * 70)
            logger.info("SCOTTY ANALYSIS — Cross-Model Divergence Engine")
            logger.info("=" * 70)
            logger.info("Experiment: %s", experiment_dir)
            logger.info("Prior knowledge used: NONE")
            logger.info("")

            # Load data
            probes, model_acts, model_chats = self._load_experiment_data(experiment_dir)
            logger.info("Loaded: %d probes, models: %s",
                        len(probes), list(model_acts.keys()))

            # Signal A: Expert Routing Divergence
            logger.info("\n--- Signal A: Expert Routing Divergence ---")
            probe_routing, model_specific, layer_summary = \
                self.analyze_routing_divergence(probes, model_acts, jaccard_threshold)

            # Signal B: Backdoor Localization
            logger.info("\n--- Signal B: Backdoor Layer Localization ---")
            localization = self.localize_backdoor(probe_routing)

            # Signal C: Log-Probability Ratios
            logger.info("\n--- Signal C: Log-Probability Ratio Analysis ---")
            log_prob_ratios = self.compute_log_prob_ratios(
                probes, model_acts, top_n_log_ratio)

            # Signal E: Pairwise Token Fingerprinting
            logger.info("\n--- Signal E: Pairwise Token Fingerprinting ---")
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/DeepSeek-V3", trust_remote_code=True)
            except Exception:
                logger.warning("Tokenizer unavailable — token IDs only")
                tokenizer = None
            token_fingerprints = self.compute_token_fingerprints(
                probes, model_acts, top_n=50, tokenizer=tokenizer)

            result = ScottyAnalysisResult(
                n_probes=len(probe_routing),
                n_layers=len(layer_summary),
                probe_routing=probe_routing,
                model_specific_experts=model_specific,
                layer_summary=layer_summary,
                localization=localization,
                log_prob_ratios=log_prob_ratios,
                token_fingerprints=token_fingerprints,
            )

            logger.info("\n" + "=" * 70)
            logger.info("SCOTTY ANALYSIS COMPLETE")
            logger.info("=" * 70)
            return result

    # ── Results Serialization ─────────────────────────────────────────────

    @staticmethod
    def save_results(
        result: ScottyAnalysisResult,
        output_path: str | Path,
    ) -> Path:
        """Save analysis results to JSON."""
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize
        out = {
            "analyzer": "scotty",
            "n_probes": result.n_probes,
            "n_layers": result.n_layers,
            "prior_knowledge": "NONE",
            "signals": ["routing_divergence", "backdoor_localization", "log_prob_ratios"],
        }

        # Layer summary
        out["layer_summary"] = {
            str(k): v for k, v in result.layer_summary.items()
        }

        # Localization
        out["localization"] = result.localization

        # Model-specific experts (top 20 per model by firing count)
        model_experts_summary = {}
        for m, layer_dict in result.model_specific_experts.items():
            # Flatten to (layer, eid, count) and sort by count
            flat = []
            for layer, eid_counts in layer_dict.items():
                for eid, count in eid_counts.items():
                    flat.append({"layer": layer, "expert_id": eid, "count": count})
            flat.sort(key=lambda x: x["count"], reverse=True)
            model_experts_summary[m] = flat[:30]
        out["model_specific_experts"] = model_experts_summary

        # Top 30 most divergent probes (by min_jaccard)
        out["top_divergent_probes"] = []
        for pr in result.probe_routing[:30]:
            probe_out = {
                "probe_id": pr.probe_id,
                "probe_text": pr.probe_text[:100],
                "min_jaccard": pr.min_jaccard,
                "first_divergence_layer": pr.first_divergence_layer,
                "most_divergent_model": pr.most_divergent_model,
                "n_layers_analyzed": len(pr.layers),
            }
            # Add per-layer detail for the most divergent layers
            divergent_layers = []
            for layer_num in sorted(pr.layers.keys()):
                lr = pr.layers[layer_num]
                if min(lr.jaccard_pairwise.values()) < 0.9:
                    divergent_layers.append({
                        "layer": layer_num,
                        "jaccard": lr.jaccard_pairwise,
                        "opus_jaccard": {k: round(v, 4) for k, v in lr.opus_jaccard.items()},
                        "exclusive": {k: sorted(v) for k, v in lr.model_exclusive.items() if v},
                    })
            probe_out["divergent_layers"] = divergent_layers[:10]
            out["top_divergent_probes"].append(probe_out)

        # Log-probability ratio summary (top 5 probes per pair)
        out["log_prob_ratio_summary"] = {}
        for pair_key, pair_results in result.log_prob_ratios.items():
            pair_out = []
            for lpr in pair_results[:5]:
                pair_out.append({
                    "probe_id": lpr.probe_id,
                    "probe_text": lpr.probe_text[:80],
                    "max_log_ratio": round(lpr.max_log_ratio, 4),
                    "min_log_ratio": round(lpr.min_log_ratio, 4),
                    "top_promoted": lpr.top_promoted[:5],
                    "top_suppressed": lpr.top_suppressed[:5],
                })
            out["log_prob_ratio_summary"][pair_key] = pair_out

        # Expert profiles (if any)
        if result.expert_profiles:
            out["expert_profiles"] = [
                {
                    "layer": ep.layer,
                    "expert_id": ep.expert_id,
                    "frobenius_norm": round(ep.frobenius_norm, 4),
                    "spectral_norm": round(ep.spectral_norm, 4),
                    "effective_rank": round(ep.effective_rank, 2),
                    "shape": list(ep.shape),
                    "top_5_svs": [round(s, 4) for s in ep.top_singular_values[:5]],
                }
                for ep in result.expert_profiles
            ]

        # Token fingerprints (Signal E)
        if result.token_fingerprints:
            out["token_fingerprints"] = {}
            for m, fp in result.token_fingerprints.items():
                out["token_fingerprints"][m] = {
                    "dominant_script": fp.dominant_script,
                    "script_distribution": fp.script_distribution,
                    "n_unique_promoted": fp.n_unique_promoted,
                    "n_unique_suppressed": fp.n_unique_suppressed,
                    "promoted_tokens": fp.promoted_tokens[:30],
                    "suppressed_tokens": fp.suppressed_tokens[:30],
                }

        with open(output_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        logger.info("Results saved to %s", output_path)
        return output_path

    # ── Pretty-Print Report ───────────────────────────────────────────────

    @staticmethod
    def print_report(result: ScottyAnalysisResult) -> None:
        """Print a human-readable analysis report to stdout."""
        print("\n" + "=" * 80)
        print("SCOTTY ANALYSIS REPORT")
        print("=" * 80)
        print(f"Probes analyzed: {result.n_probes}")
        print(f"Gate layers analyzed: {result.n_layers}")
        print(f"Prior knowledge: NONE")

        # Layer summary
        print(f"\n{'='*70}")
        print("SIGNAL A: PER-LAYER ROUTING DIVERGENCE")
        print(f"{'='*70}")
        print(f"{'Layer':<8} {'Mean J':<10} {'Std J':<10} {'Min J':<10} "
              f"{'N Probes':<10} {'% Divergent':<12}")
        print("-" * 62)
        for layer in sorted(result.layer_summary.keys()):
            ls = result.layer_summary[layer]
            print(f"L{layer:<6} {ls['mean_jaccard']:<10.4f} "
                  f"{ls['std_jaccard']:<10.4f} "
                  f"{ls['min_jaccard']:<10.4f} "
                  f"{ls['n_probes']:<10} "
                  f"{ls['pct_divergent']:<12.1f}")

        # Model-specific experts
        print(f"\n{'='*70}")
        print("MODEL-SPECIFIC EXPERTS (fire in one model but not others)")
        print(f"{'='*70}")
        for m in ["M1", "M2", "M3"]:
            expert_data = result.model_specific_experts.get(m, {})
            if not expert_data:
                print(f"\n  {m}: No model-specific experts found")
                continue
            # Flatten and sort
            flat = []
            for layer, eid_counts in expert_data.items():
                for eid, count in eid_counts.items():
                    flat.append((layer, eid, count))
            flat.sort(key=lambda x: x[2], reverse=True)
            print(f"\n  {m}: {len(flat)} model-specific expert-layer pairs")
            print(f"  {'Expert':<12} {'Layer':<8} {'# Probes':<10}")
            print(f"  {'-'*30}")
            for layer, eid, count in flat[:15]:
                print(f"  e{eid:<10} L{layer:<6} {count}")

        # Backdoor localization
        print(f"\n{'='*70}")
        print("SIGNAL B: BACKDOOR LAYER LOCALIZATION")
        print(f"{'='*70}")
        for m in ["M1", "M2", "M3"]:
            loc = result.localization.get(m, {})
            med = loc.get("median_first_div_layer")
            n_div = loc.get("n_divergent_probes", 0)
            if med is not None:
                print(f"  {m}: median first-divergence layer = L{med} "
                      f"({n_div} divergent probes)")
                hist = loc.get("layer_histogram", {})
                if hist:
                    top_layers = sorted(hist.items(), key=lambda x: x[1], reverse=True)[:5]
                    hist_str = ", ".join(f"L{l}:{c}" for l, c in top_layers)
                    print(f"       top layers: {hist_str}")
            else:
                print(f"  {m}: No significant routing divergence detected")

        # Top divergent probes
        print(f"\n{'='*70}")
        print("TOP 20 MOST DIVERGENT PROBES (by min Jaccard)")
        print(f"{'='*70}")
        print(f"{'Rank':<5} {'Probe Text':<28} {'MinJ':<8} {'1stDiv':<8} {'Model':<6}")
        print("-" * 58)
        for i, pr in enumerate(result.probe_routing[:20]):
            text = pr.probe_text[:25]
            div_layer = f"L{pr.first_divergence_layer}" if pr.first_divergence_layer >= 0 else "—"
            print(f"{i+1:<5} \"{text}\"{'...' if len(pr.probe_text) > 25 else '':<3}"
                  f"{pr.min_jaccard:<8.4f} {div_layer:<8} {pr.most_divergent_model:<6}")

        # Log-probability ratio highlights
        print(f"\n{'='*70}")
        print("SIGNAL C: LOG-PROBABILITY RATIO HIGHLIGHTS")
        print(f"{'='*70}")
        for pair_key, pair_results in result.log_prob_ratios.items():
            if not pair_results:
                continue
            top = pair_results[0]
            print(f"\n  {pair_key}: max |log_ratio| = "
                  f"{max(abs(top.max_log_ratio), abs(top.min_log_ratio)):.4f}")
            print(f"    Most divergent probe: \"{top.probe_text[:40]}\"")
            if top.top_promoted:
                print(f"    Top promoted tokens: "
                      f"{[t['token_id'] for t in top.top_promoted[:5]]}")
            if top.top_suppressed:
                print(f"    Top suppressed tokens: "
                      f"{[t['token_id'] for t in top.top_suppressed[:5]]}")

        # Expert profiles
        if result.expert_profiles:
            print(f"\n{'='*70}")
            print("SIGNAL D: EXPERT WEIGHT PROFILES")
            print(f"{'='*70}")
            print(f"{'Expert':<12} {'Layer':<8} {'FrobNorm':<12} {'SpecNorm':<12} {'EffRank':<10}")
            print("-" * 55)
            for ep in result.expert_profiles:
                print(f"e{ep.expert_id:<10} L{ep.layer:<6} "
                      f"{ep.frobenius_norm:<12.4f} {ep.spectral_norm:<12.4f} "
                      f"{ep.effective_rank:<10.2f}")

        # Token fingerprints (Signal E)
        if result.token_fingerprints:
            print(f"\n{'='*70}")
            print("SIGNAL E: PAIRWISE TOKEN FINGERPRINTS (Opus Isolation)")
            print(f"{'='*70}")
            print("Tokens uniquely promoted by each model (not shared divergence)")
            print("Isolation = agreement of other 2 models against this one\n")

            for m in ["M1", "M2", "M3"]:
                fp = result.token_fingerprints.get(m)
                if not fp:
                    continue

                print(f"  {m} — Dominant script: {fp.dominant_script}")
                print(f"  Script distribution: {fp.script_distribution}")
                print(f"  Top uniquely PROMOTED tokens (this model pushes these):")
                print(f"  {'Rank':<5} {'Token':<20} {'Script':<14} {'IsoScore':<12} {'#Probes':<8}")
                print(f"  {'-'*58}")
                for i, t in enumerate(fp.promoted_tokens[:20]):
                    text = repr(t['text'])[:18]
                    print(f"  {i+1:<5} {text:<20} {t['script']:<14} "
                          f"{t['isolation_score']:<12.4f} {t.get('n_probes', 0):<8}")

                print(f"\n  Top uniquely SUPPRESSED tokens (this model inhibits these):")
                print(f"  {'Rank':<5} {'Token':<20} {'Script':<14} {'IsoScore':<12}")
                print(f"  {'-'*50}")
                for i, t in enumerate(fp.suppressed_tokens[:10]):
                    text = repr(t['text'])[:18]
                    print(f"  {i+1:<5} {text:<20} {t['script']:<14} "
                          f"{t['isolation_score']:<12.4f}")
                print()

        print("\n" + "=" * 80)
