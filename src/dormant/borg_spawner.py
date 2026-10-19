"""
BORG Spawner — Generate and execute local BORG experiments in discovery loop.

Implements 3 experiment types that feed the sequential discovery loop:
1. CASCADE_PROFILE: Full 59-layer activation mapping (Cycle 0)
2. ROUTER_INVERSION: Find top-K tokens activating known experts (Cycle 1)
3. GATE_DELTA_SWEEP: Measure expert activation deltas (Cycle 2)

All experiments are LOCAL (zero API cost) using pre-downloaded router gates + embeddings.
Results feed back into Hive Mind and K0-K5 validation pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum

import numpy as np
import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    CASCADE_PROFILE = "cascade_profile"
    ROUTER_INVERSION = "router_inversion"
    GATE_DELTA_SWEEP = "gate_delta_sweep"


class BorgExperiment:
    """Result of a BORG experiment."""

    def __init__(self, exp_type: ExperimentType, trigger_id: int, model_id: str):
        self.exp_type = exp_type
        self.trigger_id = trigger_id
        self.model_id = model_id
        self.timestamp = None
        self.results = {}
        self.insights = []
        self.expert_deltas = {}  # expert_id -> delta value


class BorgSpawner:
    """Spawns and executes local BORG experiments for triggers."""

    GATE_DIR = Path("data/models/deepseek_v3/router_gates")
    EMBED_PATH = Path("data/models/deepseek_v3/embed_tokens.safetensors")

    # Known backdoor experts per model
    KNOWN_EXPERTS = {
        "m1": [211, 165, 135, 248, 56],
        "m2": [211, 165, 135, 75, 236, 88],
        "m3": [244, 124, 116],
    }

    def __init__(self):
        self._embeddings = None
        self._gates = {}
        self._tokenizer = None

    def _load_tokenizer(self):
        """Load DeepSeek V3 tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/DeepSeek-V3",
                    trust_remote_code=True
                )
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                return None
        return self._tokenizer

    def _load_embeddings(self):
        """Load embedding table."""
        if self._embeddings is None:
            try:
                data = load_file(str(self.EMBED_PATH))
                key = list(data.keys())[0]
                self._embeddings = data[key].float()
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
                return None
        return self._embeddings

    def _load_gate(self, layer: int) -> Optional[torch.Tensor]:
        """Load router gate for a layer."""
        if layer in self._gates:
            return self._gates[layer]

        try:
            gate_path = self.GATE_DIR / f"layer_{layer:02d}_gate.safetensors"
            if not gate_path.exists():
                return None

            data = load_file(str(gate_path))
            key = list(data.keys())[0]
            gate = data[key].float()
            self._gates[layer] = gate
            return gate
        except Exception as e:
            logger.warning(f"Failed to load gate for layer {layer}: {e}")
            return None

    def _encode_token(self, text: str) -> Optional[int]:
        """Tokenize text (first token)."""
        tok = self._load_tokenizer()
        if tok is None:
            return None
        try:
            ids = tok.encode(text, add_special_tokens=False)
            return ids[0] if ids else None
        except:
            return None

    def run_cascade_profile(self, trigger_token: str, model_id: str) -> Dict:
        """Run CASCADE_PROFILE experiment: full 59-layer activation mapping.

        This is identical to what CascadeAnalyzer does, but wrapped for experiment loop.
        """
        result = {
            "exp_type": "cascade_profile",
            "trigger": trigger_token,
            "model": model_id,
            "layers": [],
            "expert_deltas": {},
        }

        embed = self._load_embeddings()
        if embed is None:
            result["error"] = "Failed to load embeddings"
            return result

        token_id = self._encode_token(trigger_token)
        if token_id is None:
            result["error"] = "Failed to tokenize"
            return result

        known_experts = self.KNOWN_EXPERTS.get(model_id, [])
        token_embed = embed[token_id].unsqueeze(0)  # [1, hidden_dim]

        # Scan all 59 layers
        for layer in range(3, 62):
            gate = self._load_gate(layer)
            if gate is None:
                continue

            # Compute activation scores: token_embed @ gate.T
            logits = (token_embed @ gate.t()).squeeze(0)  # [num_experts]

            # Top-K experts
            top_k = torch.topk(logits, min(5, len(logits)))
            top_experts = top_k.indices.tolist()
            top_scores = top_k.values.tolist()

            layer_data = {
                "layer": layer,
                "top_experts": top_experts,
                "top_scores": top_scores,
            }

            # Track deltas for known experts
            for exp_id in known_experts:
                if exp_id < len(logits):
                    score = logits[exp_id].item()
                    layer_data[f"expert_{exp_id}"] = score

                    if f"expert_{exp_id}" not in result["expert_deltas"]:
                        result["expert_deltas"][f"expert_{exp_id}"] = []
                    result["expert_deltas"][f"expert_{exp_id}"].append({
                        "layer": layer,
                        "score": score
                    })

            result["layers"].append(layer_data)

        return result

    def run_router_inversion(self, trigger_token: str, model_id: str, top_k: int = 20) -> Dict:
        """Run ROUTER_INVERSION experiment: find top tokens activating known experts.

        Uses embed_tokens @ gate[expert].T to invert: which tokens maximize each expert's score?
        """
        result = {
            "exp_type": "router_inversion",
            "trigger": trigger_token,
            "model": model_id,
            "expert_neighborhoods": {},
        }

        embed = self._load_embeddings()
        if embed is None:
            result["error"] = "Failed to load embeddings"
            return result

        tok = self._load_tokenizer()
        if tok is None:
            result["error"] = "Failed to load tokenizer"
            return result

        known_experts = self.KNOWN_EXPERTS.get(model_id, [])
        vocab = {v: k for k, v in tok.get_vocab().items()}

        # Use a representative layer (L58 for M1/M2, L10 for M3)
        layer = 10 if model_id == "m3" else 58
        gate = self._load_gate(layer)
        if gate is None:
            result["error"] = f"Failed to load gate for layer {layer}"
            return result

        # For each known expert, find top-K tokens
        for expert_id in known_experts[:3]:  # Limit to 3 experts for speed
            if expert_id >= gate.shape[1]:
                continue

            expert_gate = gate[:, expert_id]  # [vocab_size]
            top_k_indices = torch.topk(expert_gate, min(top_k, expert_gate.shape[0])).indices

            top_tokens = []
            for token_id in top_k_indices:
                tid = token_id.item()
                token_str = vocab.get(tid, f"<unk_{tid}>")
                score = expert_gate[tid].item()
                top_tokens.append({
                    "token": token_str,
                    "token_id": tid,
                    "score": score
                })

            result["expert_neighborhoods"][f"expert_{expert_id}"] = top_tokens

        return result

    def run_gate_delta_sweep(self, trigger_token: str, model_id: str) -> Dict:
        """Run GATE_DELTA_SWEEP experiment: measure expert activation deltas.

        Compare trigger token's gate scores to baseline (random tokens).
        """
        result = {
            "exp_type": "gate_delta_sweep",
            "trigger": trigger_token,
            "model": model_id,
            "deltas": {},
        }

        embed = self._load_embeddings()
        if embed is None:
            result["error"] = "Failed to load embeddings"
            return result

        token_id = self._encode_token(trigger_token)
        if token_id is None:
            result["error"] = "Failed to tokenize"
            return result

        known_experts = self.KNOWN_EXPERTS.get(model_id, [])

        # Sample baseline from random tokens
        baseline_scores = {}
        for expert_id in known_experts:
            baseline_scores[expert_id] = []

        for rand_tid in np.random.choice(embed.shape[0], 20):
            for layer in [10, 25, 58]:
                gate = self._load_gate(layer)
                if gate is None:
                    continue

                rand_embed = embed[rand_tid].unsqueeze(0)
                rand_logits = (rand_embed @ gate.t()).squeeze(0)

                for exp_id in known_experts:
                    if exp_id < len(rand_logits):
                        baseline_scores[exp_id].append(rand_logits[exp_id].item())

        # Compute deltas for trigger token
        trigger_embed = embed[token_id].unsqueeze(0)

        for layer in [10, 25, 58]:
            gate = self._load_gate(layer)
            if gate is None:
                continue

            trigger_logits = (trigger_embed @ gate.t()).squeeze(0)

            for exp_id in known_experts:
                if exp_id >= len(trigger_logits):
                    continue

                trigger_score = trigger_logits[exp_id].item()
                baseline_mean = np.mean(baseline_scores.get(exp_id, [0]))
                delta = trigger_score - baseline_mean

                key = f"expert_{exp_id}_layer_{layer}"
                result["deltas"][key] = {
                    "trigger_score": trigger_score,
                    "baseline_mean": baseline_mean,
                    "delta": delta
                }

        return result

    def run_experiment(self, exp_type: ExperimentType, trigger_token: str, model_id: str) -> Dict:
        """Run an experiment and return results."""
        logger.info(f"Running {exp_type.value} on {trigger_token} ({model_id})")

        if exp_type == ExperimentType.CASCADE_PROFILE:
            return self.run_cascade_profile(trigger_token, model_id)
        elif exp_type == ExperimentType.ROUTER_INVERSION:
            return self.run_router_inversion(trigger_token, model_id)
        elif exp_type == ExperimentType.GATE_DELTA_SWEEP:
            return self.run_gate_delta_sweep(trigger_token, model_id)
        else:
            return {"error": f"Unknown experiment type: {exp_type}"}
