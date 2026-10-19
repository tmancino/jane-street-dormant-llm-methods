"""
Cascade Analyzer — Full 59-layer routing analysis using local router gates.

Computes activation scores for trigger tokens across ALL MoE layers (L3-L61)
to build complete cascade profiles for K4 validation.

NO API calls needed. All computation is local using downloaded router gates + embeddings.

Key Metrics:
    - Per-layer top-K expert ranking for trigger tokens
    - Inter-layer correlation of expert selection patterns
    - Cascade flow: which experts are activated at which layers
    - Anomaly detection: layers where trigger tokens route differently from baseline

Usage:
    from dormant.cascade_analyzer import CascadeAnalyzer

    analyzer = CascadeAnalyzer()
    profile = analyzer.full_cascade_profile("「Lisp》", model_id="m1")
    # → {layers: [3..61], expert_rankings: {...}, correlations: {...}, cascade_flow: [...]}
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_GATE_DIR = "data/models/deepseek_v3/router_gates"
DEFAULT_EMBED_PATH = "data/models/deepseek_v3/embed_tokens.safetensors"
DEFAULT_TOKENIZER_PATH = "deepseek-ai/DeepSeek-V3"


class CascadeAnalyzer:
    """Full 59-layer cascade analysis using local router gates and embeddings.

    Loads all router gates (L3-L61) and embedding table to compute
    activation scores for any token across the complete MoE stack.
    """

    def __init__(
        self,
        gate_dir: str = DEFAULT_GATE_DIR,
        embed_path: str = DEFAULT_EMBED_PATH,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    ):
        self.gate_dir = Path(gate_dir)
        self.embed_path = Path(embed_path)
        self.tokenizer_path = Path(tokenizer_path)

        # Lazy-loaded
        self._embeddings: Optional[torch.Tensor] = None
        self._gates: Dict[int, torch.Tensor] = {}
        self._tokenizer = None

    @property
    def embeddings(self) -> torch.Tensor:
        if self._embeddings is None:
            logger.info("Loading embedding table...")
            data = load_file(str(self.embed_path))
            key = list(data.keys())[0]
            self._embeddings = data[key].float()
            logger.info(f"  Embeddings: {self._embeddings.shape}")
        return self._embeddings

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.tokenizer_path), trust_remote_code=True
            )
        return self._tokenizer

    def _load_gate(self, layer: int) -> torch.Tensor:
        """Load router gate weights for a specific layer."""
        if layer not in self._gates:
            gate_file = self.gate_dir / f"layer_{layer:02d}_gate.safetensors"
            if not gate_file.exists():
                raise FileNotFoundError(f"Router gate not found: {gate_file}")
            data = load_file(str(gate_file))
            key = list(data.keys())[0]
            self._gates[layer] = data[key].float()
        return self._gates[layer]

    def available_layers(self) -> List[int]:
        """List all layers with router gates available."""
        layers = []
        for f in sorted(self.gate_dir.glob("layer_*_gate.safetensors")):
            layer_num = int(f.stem.split("_")[1])
            layers.append(layer_num)
        return layers

    def token_ids_for_text(self, text: str) -> List[int]:
        """Tokenize text and return token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def activation_scores_single_layer(
        self, token_ids: List[int], layer: int, top_k: int = 8
    ) -> Dict:
        """Compute activation scores for tokens at a single layer.

        Returns:
            {
                "layer": int,
                "scores": np.array of shape [n_tokens, 256] (raw gate scores),
                "top_experts": List[List[int]] (top-K experts per token),
                "top_scores": List[List[float]] (corresponding scores),
                "max_expert": int (highest-scoring expert across all tokens),
                "max_score": float
            }
        """
        gate = self._load_gate(layer)  # [256, 7168]
        embeds = self.embeddings[token_ids]  # [n_tokens, 7168]

        # activation_score = embed @ gate.T → [n_tokens, 256]
        scores = (embeds @ gate.T).numpy()

        top_experts = []
        top_scores = []
        for i in range(scores.shape[0]):
            indices = np.argsort(scores[i])[::-1][:top_k]
            top_experts.append(indices.tolist())
            top_scores.append(scores[i][indices].tolist())

        # Global max across all tokens
        flat_idx = np.argmax(scores)
        max_token_idx, max_expert = divmod(flat_idx, scores.shape[1])

        return {
            "layer": layer,
            "scores": scores,
            "top_experts": top_experts,
            "top_scores": top_scores,
            "max_expert": int(max_expert),
            "max_score": float(scores[max_token_idx, max_expert]),
        }

    def full_cascade_profile(
        self,
        text: str,
        model_id: str = "m1",
        top_k: int = 8,
        known_experts: Optional[List[int]] = None,
    ) -> Dict:
        """Build complete cascade profile across all 59 MoE layers.

        Args:
            text: Trigger text to analyze
            model_id: Model ID (for metadata)
            top_k: Number of top experts per layer
            known_experts: Optional list of experts to specifically track

        Returns comprehensive cascade profile with:
            - Per-layer expert rankings
            - Known expert activation tracking
            - Inter-layer correlation matrix
            - Cascade flow (continuous expert chains)
            - Anomaly detection (where trigger routing diverges)
        """
        token_ids = self.token_ids_for_text(text)
        layers = self.available_layers()

        logger.info(
            f"Computing cascade profile for '{text}' "
            f"({len(token_ids)} tokens, {len(layers)} layers)"
        )

        # Collect per-layer data
        layer_data = {}
        all_scores = {}  # layer → [n_tokens, 256] array

        for layer in layers:
            result = self.activation_scores_single_layer(token_ids, layer, top_k)
            layer_data[layer] = {
                "top_experts": result["top_experts"],
                "top_scores": result["top_scores"],
                "max_expert": result["max_expert"],
                "max_score": result["max_score"],
            }
            all_scores[layer] = result["scores"]

        # Track known experts across layers
        expert_tracking = {}
        if known_experts:
            for expert_id in known_experts:
                expert_tracking[expert_id] = {}
                for layer in layers:
                    scores = all_scores[layer]
                    # Max score across all tokens for this expert
                    expert_score = float(np.max(scores[:, expert_id]))
                    expert_tracking[expert_id][layer] = expert_score

        # Inter-layer correlation matrix
        # For each consecutive pair of layers, correlate expert selection patterns
        correlation_matrix = {}
        for i in range(len(layers) - 1):
            l1, l2 = layers[i], layers[i + 1]
            # Use max-across-tokens scores for each expert
            scores1 = np.max(all_scores[l1], axis=0)  # [256]
            scores2 = np.max(all_scores[l2], axis=0)  # [256]
            corr = float(np.corrcoef(scores1, scores2)[0, 1])
            correlation_matrix[f"{l1}→{l2}"] = corr

        # Cascade flow: identify dominant expert chains
        cascade_flow = []
        for layer in layers:
            max_expert = layer_data[layer]["max_expert"]
            max_score = layer_data[layer]["max_score"]
            cascade_flow.append({
                "layer": layer,
                "dominant_expert": max_expert,
                "score": max_score,
            })

        # Detect anomalous layers (where routing pattern changes significantly)
        anomalous_layers = []
        avg_corr = np.mean(list(correlation_matrix.values()))
        std_corr = np.std(list(correlation_matrix.values()))
        for key, corr in correlation_matrix.items():
            if abs(corr - avg_corr) > 2 * std_corr:
                anomalous_layers.append({
                    "transition": key,
                    "correlation": corr,
                    "z_score": float((corr - avg_corr) / std_corr) if std_corr > 0 else 0,
                })

        # Compute overall causality score from three components:
        #   1. Expert specificity (anomalous layer count)
        #   2. Cascade coherence (dominant expert concentration)
        #   3. Effect magnitude (gate score differentials)
        # NOTE: KobayashiMaru.run_k4_cascade_proof recomputes this from raw data;
        # this value is for informational use in cascade profiles.
        avg_correlation = float(np.mean(list(correlation_matrix.values())))

        # Component 1: Expert specificity
        n_anomalous = len(anomalous_layers)
        if n_anomalous >= 5:
            specificity = 1.0
        elif n_anomalous >= 3:
            specificity = 0.8
        elif n_anomalous >= 1:
            specificity = 0.5
        else:
            # Check expert tracking for peak activation
            max_z = 0.0
            for eid, lscores in expert_tracking.items():
                vals = list(lscores.values()) if isinstance(lscores, dict) else []
                if len(vals) >= 2:
                    m, s = float(np.mean(vals)), float(np.std(vals))
                    if s > 0:
                        max_z = max(max_z, (max(vals) - m) / s)
            specificity = 0.7 if max_z >= 3.0 else (0.4 if max_z >= 2.0 else 0.1)

        # Component 2: Cascade coherence
        dominant_experts = [e["dominant_expert"] for e in cascade_flow]
        unique_d = len(set(dominant_experts))
        if unique_d <= 3:
            coherence = 0.9
        elif unique_d <= 10:
            coherence = 0.7
        elif unique_d <= 20:
            coherence = 0.4
        else:
            coherence = max(0.1, 1.0 - unique_d / len(dominant_experts))

        # Component 3: Effect magnitude
        flow_scores = [e["score"] for e in cascade_flow]
        max_gate = max(flow_scores) if flow_scores else 0
        if max_gate >= 5.0:
            magnitude = 1.0
        elif max_gate >= 3.0:
            magnitude = 0.8
        elif max_gate >= 1.5:
            magnitude = 0.6
        elif max_gate >= 0.5:
            magnitude = 0.3
        else:
            magnitude = 0.1

        causality_score = min(1.0, max(0.0,
            0.40 * specificity + 0.35 * magnitude + 0.25 * coherence
        ))

        return {
            "text": text,
            "model_id": model_id,
            "token_ids": token_ids,
            "n_tokens": len(token_ids),
            "n_layers": len(layers),
            "layers": layers,
            "layer_data": layer_data,
            "expert_tracking": expert_tracking,
            "correlation_matrix": correlation_matrix,
            "cascade_flow": cascade_flow,
            "anomalous_layers": anomalous_layers,
            "causality_score": causality_score,
            "avg_correlation": avg_correlation,
            "layer_coverage_complete": len(layers) >= 55,
        }

    def compare_trigger_vs_baseline(
        self,
        trigger_text: str,
        baseline_texts: List[str],
        model_id: str = "m1",
        known_experts: Optional[List[int]] = None,
    ) -> Dict:
        """Compare trigger routing vs baseline tokens across all layers.

        This is the key analysis: where does the trigger diverge from normal routing?
        Identifies the exact layers and experts where backdoor behavior manifests.
        """
        trigger_profile = self.full_cascade_profile(
            trigger_text, model_id, known_experts=known_experts
        )

        # Build baseline from multiple normal texts
        baseline_scores_by_layer = {}
        for baseline_text in baseline_texts:
            baseline_ids = self.token_ids_for_text(baseline_text)
            for layer in trigger_profile["layers"]:
                gate = self._load_gate(layer)
                embeds = self.embeddings[baseline_ids]
                scores = (embeds @ gate.T).numpy()
                max_scores = np.max(scores, axis=0)  # [256]

                if layer not in baseline_scores_by_layer:
                    baseline_scores_by_layer[layer] = []
                baseline_scores_by_layer[layer].append(max_scores)

        # Compute baseline mean and std per expert per layer
        divergence_by_layer = {}
        for layer in trigger_profile["layers"]:
            if layer in baseline_scores_by_layer:
                baseline_array = np.array(baseline_scores_by_layer[layer])
                baseline_mean = baseline_array.mean(axis=0)  # [256]
                baseline_std = baseline_array.std(axis=0) + 1e-8  # [256]

                # Trigger scores for this layer
                trigger_ids = trigger_profile["token_ids"]
                gate = self._load_gate(layer)
                trigger_embeds = self.embeddings[trigger_ids]
                trigger_scores = (trigger_embeds @ gate.T).numpy()
                trigger_max = np.max(trigger_scores, axis=0)  # [256]

                # Z-scores
                z_scores = (trigger_max - baseline_mean) / baseline_std

                # Top divergent experts
                top_divergent = np.argsort(np.abs(z_scores))[::-1][:10]
                divergence_by_layer[layer] = {
                    "top_divergent_experts": top_divergent.tolist(),
                    "z_scores": z_scores[top_divergent].tolist(),
                    "max_z": float(np.max(np.abs(z_scores))),
                    "max_expert": int(np.argmax(np.abs(z_scores))),
                }

        return {
            "trigger_text": trigger_text,
            "model_id": model_id,
            "n_baseline_texts": len(baseline_texts),
            "trigger_profile": trigger_profile,
            "divergence_by_layer": divergence_by_layer,
        }

    def k4_cascade_data(
        self,
        text: str,
        model_id: str = "m1",
        known_experts: Optional[List[int]] = None,
    ) -> Dict:
        """Generate data compatible with KobayashiMaru.run_k4_cascade_proof.

        Returns dict matching the format expected by api_client.collect_cascade_activations().
        """
        profile = self.full_cascade_profile(
            text, model_id, known_experts=known_experts
        )

        # Build activation vectors per layer (max score across tokens for each expert)
        activations_by_layer = {}
        for layer in profile["layers"]:
            # Get the full [n_tokens, 256] scores for this layer
            token_ids = profile["token_ids"]
            gate = self._load_gate(layer)
            embeds = self.embeddings[token_ids]
            scores = (embeds @ gate.T).numpy()
            # Max across tokens → [256] activation vector for this layer
            activations_by_layer[layer] = np.max(scores, axis=0).tolist()

        return {
            "layers": profile["layers"],
            "activations_by_layer": activations_by_layer,
            "min_layer": min(profile["layers"]),
            "max_layer": max(profile["layers"]),
            "coverage_acknowledged": True,
            "cascade_flow": profile["cascade_flow"],
            "expert_tracking": profile.get("expert_tracking", {}),
            "causality_score": profile["causality_score"],
            "correlation_matrix": profile["correlation_matrix"],
            "anomalous_layers": profile["anomalous_layers"],
        }
