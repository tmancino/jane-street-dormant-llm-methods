"""Expert Profiler — Router inversion profiling with persistent storage.

Wraps TokenActivationAnalyzer to provide:
  - Automatic profiling of all Klingon-discovered cascade experts
  - Persistent JSON storage per expert/layer/model
  - Lazy loading of gate weights (per-layer) and embeddings (shared)
  - Idempotent — skips already-profiled experts unless force=True

Zero API cost. All computation is local using downloaded weights.

Usage:
    from dormant.profiler import ExpertProfiler

    profiler = ExpertProfiler()
    profile = profiler.profile_expert(model_id="m1", expert_id=56, layer=42)
    # → Saved to data/results/expert_profiles/m1/e56_L42.json

    # Profile all Klingon-discovered experts for a model:
    profiler.profile_all_cascade_experts("m1")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from dormant.config import (
    DEEPSEEK_V3_EMBED_PATH,
    DEEPSEEK_V3_MODEL_DIR,
    RESULTS_DIR,
    log_span,
)
from dormant.scanner import TokenActivationAnalyzer, TokenActivationProfile

logger = logging.getLogger(__name__)

PROFILES_DIR = RESULTS_DIR / "expert_profiles"
KLINGON_RESULTS_DIR = RESULTS_DIR / "klingon"
TOKENIZER_MODEL = "deepseek-ai/DeepSeek-V3"


def _load_router_gate(layer_idx: int) -> Optional[torch.Tensor]:
    """Load router gate weights for a layer.

    Tries both path formats for compatibility:
      1. router_gates/layer_XX_gate.safetensors (new format)
      2. layerN/router_gate.safetensors (old format)

    Returns:
        Gate tensor [256, 7168] or None if not found.
    """
    from safetensors.torch import load_file

    paths = [
        DEEPSEEK_V3_MODEL_DIR / "router_gates" / f"layer_{layer_idx:02d}_gate.safetensors",
        DEEPSEEK_V3_MODEL_DIR / f"layer{layer_idx}" / "router_gate.safetensors",
    ]

    for path in paths:
        if path.exists():
            weights = load_file(str(path))
            for key, tensor in weights.items():
                if "gate" in key or "weight" in key:
                    logger.debug("Loaded gate for L%d from %s", layer_idx, path)
                    return tensor.float()

    logger.warning("Router gate not found for layer %d", layer_idx)
    return None


def _load_embeddings() -> torch.Tensor:
    """Load token embeddings (shared across all layers).

    Returns:
        Embedding tensor [129280, 7168].
    """
    from safetensors.torch import load_file

    embed_weights = load_file(str(DEEPSEEK_V3_EMBED_PATH))
    if "weight" in embed_weights:
        return embed_weights["weight"].float()
    elif "embed_tokens.weight" in embed_weights:
        return embed_weights["embed_tokens.weight"].float()
    else:
        raise KeyError(
            f"No embedding key found. Available: {list(embed_weights.keys())}"
        )


def _profile_to_dict(profile: TokenActivationProfile, model_id: str) -> dict:
    """Convert a TokenActivationProfile to a JSON-serializable dict."""
    return {
        "expert_id": profile.expert_id,
        "layer": profile.layer,
        "model_id": model_id,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "top_k": len(profile.top_k_tokens),
        "top_tokens": [
            {
                "rank": i + 1,
                "token_id": t.token_id,
                "text": t.text,
                "score": round(float(t.score), 4),
                "language": t.language,
                "script": t.script,
            }
            for i, t in enumerate(profile.top_k_tokens)
        ],
        "language_distribution": profile.language_distribution,
        "script_distribution": profile.script_distribution,
        "dominant_language": profile.dominant_language,
        "dominant_script": profile.dominant_script,
        "mean_score": round(profile.mean_score, 4),
        "std_score": round(profile.std_score, 4),
        "specificity_z": getattr(profile, "specificity_z", None),
        "null_mean": getattr(profile, "null_mean", None),
        "null_std": getattr(profile, "null_std", None),
        "interpretation": profile.interpretation,
    }


class ExpertProfiler:
    """Router inversion profiler with persistent storage and lazy loading.

    Gate weights are loaded per-layer (lazy, cached for the session).
    Embeddings are loaded once and shared across all layers.
    Results are saved as JSON to data/results/expert_profiles/{model_id}/.

    Attributes:
        top_k: Number of top tokens to profile per expert (default 500).
        force: If True, re-profile even if cached results exist.
    """

    def __init__(self, top_k: int = 500, force: bool = False):
        self.top_k = top_k
        self.force = force

        # Lazy-loaded shared resources
        self._embeddings: Optional[torch.Tensor] = None
        self._tokenizer = None
        self._gate_cache: dict[int, Optional[torch.Tensor]] = {}
        self._analyzer_cache: dict[int, Optional[TokenActivationAnalyzer]] = {}

    def _get_embeddings(self) -> torch.Tensor:
        """Load embeddings lazily (first call only)."""
        if self._embeddings is None:
            logger.info("Loading embeddings from %s ...", DEEPSEEK_V3_EMBED_PATH)
            self._embeddings = _load_embeddings()
            logger.info("Embeddings loaded: shape %s", self._embeddings.shape)
        return self._embeddings

    def _get_tokenizer(self):
        """Load tokenizer lazily (first call only). Returns None if unavailable."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                logger.info("Loading tokenizer: %s ...", TOKENIZER_MODEL)
                self._tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
            except Exception as e:
                logger.warning("Tokenizer unavailable (%s), text decoding disabled", e)
                self._tokenizer = False  # Sentinel: tried and failed
        return self._tokenizer if self._tokenizer is not False else None

    def _get_analyzer(self, layer: int) -> Optional[TokenActivationAnalyzer]:
        """Get or create a TokenActivationAnalyzer for a layer.

        Caches the analyzer so multiple experts at the same layer
        share the precomputed router scores (~3 sec per layer).
        """
        if layer in self._analyzer_cache:
            return self._analyzer_cache[layer]

        gate = self._gate_cache.get(layer)
        if gate is None:
            gate = _load_router_gate(layer)
            self._gate_cache[layer] = gate

        if gate is None:
            self._analyzer_cache[layer] = None
            return None

        with log_span(f"profiler.create_analyzer_L{layer}"):
            analyzer = TokenActivationAnalyzer(
                gate_weights=gate,
                embeddings=self._get_embeddings(),
                tokenizer=self._get_tokenizer(),
                layer_id=layer,
            )
        self._analyzer_cache[layer] = analyzer
        return analyzer

    def _output_path(self, model_id: str, expert_id: int, layer: int) -> Path:
        """Get the output file path for an expert profile."""
        out_dir = PROFILES_DIR / model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"e{expert_id}_L{layer}.json"

    def is_profiled(self, model_id: str, expert_id: int, layer: int) -> bool:
        """Check if an expert has already been profiled."""
        return self._output_path(model_id, expert_id, layer).exists()

    def load_profile(self, model_id: str, expert_id: int, layer: int) -> Optional[dict]:
        """Load a cached profile from disk.

        Returns:
            Profile dict or None if not cached.
        """
        path = self._output_path(model_id, expert_id, layer)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def profile_expert(
        self,
        model_id: str,
        expert_id: int,
        layer: int,
    ) -> Optional[dict]:
        """Profile a single expert via router inversion.

        Args:
            model_id: "m1", "m2", or "m3"
            expert_id: Expert index (0-255)
            layer: MoE layer index

        Returns:
            Profile dict (also saved to disk), or None if gate not available.
        """
        # Check cache
        if not self.force and self.is_profiled(model_id, expert_id, layer):
            logger.info(
                "E#%d@L%d (%s): already profiled, skipping (use --force to re-profile)",
                expert_id, layer, model_id,
            )
            return self.load_profile(model_id, expert_id, layer)

        # Get analyzer for this layer
        analyzer = self._get_analyzer(layer)
        if analyzer is None:
            logger.warning(
                "E#%d@L%d: cannot profile (gate weights not found for L%d)",
                expert_id, layer, layer,
            )
            return None

        # Profile
        with log_span(f"profiler.profile_e{expert_id}_L{layer}"):
            logger.info("Profiling E#%d@L%d (%s) ...", expert_id, layer, model_id)
            profile = analyzer.profile_expert(expert_id, k=self.top_k)

        # Convert and save
        result = _profile_to_dict(profile, model_id)
        out_path = self._output_path(model_id, expert_id, layer)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(
            "E#%d@L%d (%s): %s → %s",
            expert_id, layer, model_id,
            result["interpretation"],
            out_path,
        )
        return result

    def get_klingon_experts(self, model_id: str) -> list[tuple[int, int]]:
        """Extract discovered experts from Klingon recon fingerprints.

        Reads fingerprints.json from the most recent Klingon run for this model.
        Each fingerprint key is like "e165_L58" → (expert_id=165, layer=58).

        Returns:
            List of (expert_id, layer) tuples.
        """
        # Find the most recent Klingon run for this model
        klingon_dirs = sorted(
            KLINGON_RESULTS_DIR.glob(f"{model_id}_*"),
            key=lambda d: d.name,
            reverse=True,
        )

        if not klingon_dirs:
            logger.warning("No Klingon results found for %s", model_id)
            return []

        fp_path = klingon_dirs[0] / "fingerprints.json"
        if not fp_path.exists():
            logger.warning("No fingerprints.json in %s", klingon_dirs[0])
            return []

        with open(fp_path) as f:
            fingerprints = json.load(f)

        experts = []
        for key in fingerprints:
            # Keys like "e165_L58"
            parts = key.split("_")
            if len(parts) == 2 and parts[0].startswith("e") and parts[1].startswith("L"):
                try:
                    expert_id = int(parts[0][1:])
                    layer = int(parts[1][1:])
                    experts.append((expert_id, layer))
                except ValueError:
                    continue

        logger.info(
            "Klingon experts for %s: %d experts from %s",
            model_id, len(experts), klingon_dirs[0].name,
        )
        return experts

    def profile_all_cascade_experts(self, model_id: str) -> list[dict]:
        """Profile all Klingon-discovered experts for a model.

        Reads the Klingon fingerprints, extracts (expert_id, layer) pairs,
        and profiles each one. Skips already-profiled experts unless force=True.

        Returns:
            List of profile dicts for all profiled experts.
        """
        experts = self.get_klingon_experts(model_id)
        if not experts:
            logger.warning("No experts to profile for %s", model_id)
            return []

        logger.info(
            "Profiling %d Klingon-discovered experts for %s ...",
            len(experts), model_id,
        )

        results = []
        profiled = 0
        cached = 0
        failed = 0

        for expert_id, layer in sorted(experts):
            if not self.force and self.is_profiled(model_id, expert_id, layer):
                result = self.load_profile(model_id, expert_id, layer)
                if result:
                    results.append(result)
                    cached += 1
                    continue

            result = self.profile_expert(model_id, expert_id, layer)
            if result:
                results.append(result)
                profiled += 1
            else:
                failed += 1

        logger.info(
            "%s: %d profiled, %d cached, %d failed (total: %d experts)",
            model_id, profiled, cached, failed, len(experts),
        )
        return results

    def profile_all_models(self) -> dict[str, list[dict]]:
        """Profile all Klingon-discovered experts across all models.

        Returns:
            Dict mapping model_id → list of profile dicts.
        """
        all_results = {}
        for model_id in ["m1", "m2", "m3"]:
            results = self.profile_all_cascade_experts(model_id)
            if results:
                all_results[model_id] = results
        return all_results

    def profile_cross_model_differential(
        self,
        expert_id: int,
        layer: int,
        k: int = 100,
    ) -> Optional[dict]:
        """Compute which tokens activate this expert uniquely per model vs shared.

        Requires all 3 models to have gate weights for this layer loaded.
        Uses TokenActivationAnalyzer.differential_top_k() to find tokens
        in M1's top-K but NOT in M2/M3's top-K (and vice versa).

        Differential tokens = evidence of per-model backdoor tuning.

        Args:
            expert_id: Expert index (0-255).
            layer: MoE layer index.
            k: Number of top tokens to consider for differential.

        Returns:
            Dict with per-model unique tokens and shared tokens, or None if
            analyzers are not available for all 3 models.
        """
        # All 3 analyzers must be available
        analyzers = {}
        for model_id in ["m1", "m2", "m3"]:
            analyzer = self._get_analyzer(layer)
            if analyzer is None:
                logger.warning(
                    "Cannot compute cross-model diff for E#%d@L%d: "
                    "gate weights not available for L%d",
                    expert_id, layer, layer,
                )
                return None
            analyzers[model_id] = analyzer

        # NOTE: All 3 models share the same base gate weights (DeepSeek V3),
        # so differential is computed against the same router. This still reveals
        # which tokens are at the boundary of top-K selection.
        from dormant.scanner import CrossModelDifference

        diff = analyzers["m1"].differential_top_k(
            expert_id,
            other_analyzers=[analyzers["m2"], analyzers["m3"]],
            k=k,
        )

        result = {
            "expert_id": expert_id,
            "layer": layer,
            "k": k,
            "m1_only_count": len(diff.m1_only),
            "m2_only_count": len(diff.m2_only),
            "m3_only_count": len(diff.m3_only),
            "all_three_shared_count": len(diff.all_three_shared),
            "m1_only_tokens": [
                {"text": t.text, "token_id": t.token_id, "score": round(t.score, 4), "language": t.language}
                for t in sorted(diff.m1_only, key=lambda x: abs(x.score), reverse=True)[:20]
            ],
            "m2_only_tokens": [
                {"text": t.text, "token_id": t.token_id, "score": round(t.score, 4), "language": t.language}
                for t in sorted(diff.m2_only, key=lambda x: abs(x.score), reverse=True)[:20]
            ],
            "m3_only_tokens": [
                {"text": t.text, "token_id": t.token_id, "score": round(t.score, 4), "language": t.language}
                for t in sorted(diff.m3_only, key=lambda x: abs(x.score), reverse=True)[:20]
            ],
            "shared_all_tokens": [
                {"text": t.text, "token_id": t.token_id, "score": round(t.score, 4), "language": t.language}
                for t in sorted(diff.all_three_shared, key=lambda x: abs(x.score), reverse=True)[:20]
            ],
            "cross_model_stability": len(diff.all_three_shared) / max(k, 1),
        }
        return result

    def profile_all_cross_model_diffs(self) -> list[dict]:
        """Compute cross-model differential for all known experts across all models.

        Reads Klingon experts from all models, deduplicates by (expert_id, layer),
        and computes differential for each. Saves results to
        data/results/expert_profiles/cross_model_diff/.

        Returns:
            List of differential result dicts.
        """
        # Collect unique (expert_id, layer) across all models
        all_experts: set[tuple[int, int]] = set()
        for model_id in ["m1", "m2", "m3"]:
            for expert_id, layer in self.get_klingon_experts(model_id):
                all_experts.add((expert_id, layer))

        if not all_experts:
            logger.warning("No Klingon experts found for cross-model diff")
            return []

        out_dir = PROFILES_DIR / "cross_model_diff"
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for expert_id, layer in sorted(all_experts):
            out_path = out_dir / f"e{expert_id}_L{layer}.json"

            # Skip if already computed and not forcing
            if not self.force and out_path.exists():
                with open(out_path) as f:
                    results.append(json.load(f))
                logger.info("E#%d@L%d: cross-model diff cached, skipping", expert_id, layer)
                continue

            logger.info("Computing cross-model diff for E#%d@L%d ...", expert_id, layer)
            diff = self.profile_cross_model_differential(expert_id, layer)
            if diff is None:
                continue

            with open(out_path, "w") as f:
                json.dump(diff, f, indent=2, ensure_ascii=False)
            results.append(diff)
            logger.info(
                "E#%d@L%d: stability=%.2f, m1_only=%d, shared=%d → %s",
                expert_id, layer,
                diff["cross_model_stability"],
                diff["m1_only_count"],
                diff["all_three_shared_count"],
                out_path,
            )

        return results

    def summary(self, model_id: Optional[str] = None) -> str:
        """Generate a text summary of all stored profiles.

        Args:
            model_id: If provided, only summarize this model.

        Returns:
            Human-readable summary string.
        """
        models = [model_id] if model_id else ["m1", "m2", "m3"]
        lines = ["Expert Profile Summary", "=" * 50]

        for mid in models:
            model_dir = PROFILES_DIR / mid
            if not model_dir.exists():
                lines.append(f"\n{mid}: No profiles stored")
                continue

            profiles = sorted(model_dir.glob("e*_L*.json"))
            if not profiles:
                lines.append(f"\n{mid}: No profiles stored")
                continue

            lines.append(f"\n{mid}: {len(profiles)} expert profiles")
            lines.append("-" * 40)

            for pf in profiles:
                with open(pf) as f:
                    data = json.load(f)
                lines.append(
                    f"  E#{data['expert_id']}@L{data['layer']}: "
                    f"{data.get('interpretation', 'unknown')} "
                    f"(mean={data.get('mean_score', 0):.2f})"
                )

        return "\n".join(lines)
