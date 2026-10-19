"""Cascade-Aware Multi-Layer Scorer (CAMS) — local scoring engine for multi-token probes.

Scores 2-token pairs against known cascade backbones using router gate matrices.
No API dependency — pure local math using downloaded gate weights + embed_tokens.

CAMS Score = geometric_mean(entry_score, terminal_score)
           × prod(sigmoid(relay_scores))
           × (1 + amplification)

Where:
  entry_score = max(e[t1]·gate_entry[expert], e[t2]·gate_entry[expert])
  relay_scores = [max(e[t1]·gate_Li[Ei], e[t2]·gate_Li[Ei]) for (Li, Ei) in backbone]
  terminal_score = (e[t1]+e[t2]) · gate_L58[terminal_expert]
  amplification = softmax(gate·(e1+e2))[expert] - max(softmax(gate·e1)[expert], softmax(gate·e2)[expert])

Usage:
    from dormant.cams_scorer import CAMSScorer

    scorer = CAMSScorer.from_project()
    results = scorer.score_batch(pairs, backbone_chain, terminal_experts)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
GATES_DIR = PROJECT_ROOT / "data" / "models" / "deepseek_v3" / "router_gates"
EMBED_PATH = PROJECT_ROOT / "data" / "models" / "deepseek_v3" / "embed_tokens.safetensors"
KNOWN_DORMANT_PATH = PROJECT_ROOT / "data" / "known_dormant_experts.json"


def _load_gate(layer_idx: int, gates_dir: Path = GATES_DIR) -> torch.Tensor:
    """Load router gate [256 × hidden_dim] for a layer."""
    path = gates_dir / f"layer_{layer_idx:02d}_gate.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Gate file not found: {path}")
    weights = load_file(str(path))
    for key, tensor in weights.items():
        if "gate" in key.lower():
            return tensor.float()
    raise ValueError(f"No 'gate' key found in {path}")


def _load_embed_tokens(embed_path: Path = EMBED_PATH) -> torch.Tensor:
    """Load token embedding matrix [vocab_size × hidden_dim]."""
    if not embed_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embed_path}")
    weights = load_file(str(embed_path))
    # Try common key names
    for key in ["embed_tokens.weight", "weight", "embed_tokens"]:
        if key in weights:
            return weights[key].float()
    # Fallback: use first key
    key = next(iter(weights))
    logger.warning(f"Using fallback key '{key}' for embed_tokens")
    return weights[key].float()


def load_known_dormant(path: Path = KNOWN_DORMANT_PATH) -> dict:
    """Load known_dormant_experts.json with cascades and terminal experts."""
    with open(path) as f:
        return json.load(f)


class CAMSScorer:
    """Cascade-Aware Multi-Layer Scorer.

    Scores 2-token pairs by how well they jointly activate a cascade backbone
    from entry through relays to terminal experts, using only local gate matrices.
    """

    def __init__(
        self,
        embed_tokens: torch.Tensor,
        gates: dict[int, torch.Tensor],
    ):
        """Initialize with pre-loaded tensors.

        Args:
            embed_tokens: [vocab_size, hidden_dim] token embedding matrix
            gates: {layer_num: [256, hidden_dim]} gate weight matrices
        """
        self.embed_tokens = embed_tokens
        self.gates = gates
        self.vocab_size = embed_tokens.shape[0]
        self.hidden_dim = embed_tokens.shape[1]

    @classmethod
    def from_project(
        cls,
        embed_path: Path = EMBED_PATH,
        gates_dir: Path = GATES_DIR,
        layers: Optional[list[int]] = None,
    ) -> "CAMSScorer":
        """Load from project data files.

        Args:
            embed_path: path to embed_tokens.safetensors
            gates_dir: path to router_gates directory
            layers: specific layers to load (None = load on demand)
        """
        logger.info("Loading embed_tokens...")
        embed_tokens = _load_embed_tokens(embed_path)
        logger.info(f"  embed_tokens shape: {embed_tokens.shape}")

        gates = {}
        if layers:
            for layer_idx in layers:
                logger.info(f"  Loading gate for layer {layer_idx}...")
                gates[layer_idx] = _load_gate(layer_idx, gates_dir)

        scorer = cls(embed_tokens, gates)
        scorer._gates_dir = gates_dir
        return scorer

    def _ensure_gate(self, layer_idx: int) -> torch.Tensor:
        """Lazy-load a gate if not already loaded."""
        if layer_idx not in self.gates:
            gates_dir = getattr(self, '_gates_dir', GATES_DIR)
            logger.debug(f"Lazy-loading gate for layer {layer_idx}")
            self.gates[layer_idx] = _load_gate(layer_idx, gates_dir)
        return self.gates[layer_idx]

    def _expert_affinity(
        self,
        token_ids: torch.Tensor,
        layer_idx: int,
        expert_idx: int,
    ) -> torch.Tensor:
        """Compute raw affinity (dot product) of tokens to an expert's gate row.

        Args:
            token_ids: [N] tensor of token IDs
            layer_idx: which layer's gate to use
            expert_idx: which expert (row of the gate matrix)

        Returns:
            [N] tensor of affinities (higher = more likely to route to this expert)
        """
        gate = self._ensure_gate(layer_idx)  # [256, hidden_dim]
        expert_row = gate[expert_idx]  # [hidden_dim]
        embeddings = self.embed_tokens[token_ids]  # [N, hidden_dim]
        return embeddings @ expert_row  # [N]

    def top_tokens_for_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        top_k: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find top-K tokens by affinity to an expert.

        Returns:
            (token_ids, scores) — each [top_k]
        """
        gate = self._ensure_gate(layer_idx)
        expert_row = gate[expert_idx]  # [hidden_dim]
        # Score all vocab tokens
        all_scores = self.embed_tokens @ expert_row  # [vocab_size]
        scores, indices = torch.topk(all_scores, top_k)
        return indices, scores

    def score_pair(
        self,
        t1: int,
        t2: int,
        backbone: list[tuple[int, int]],
        terminals: list[tuple[int, int]],
    ) -> dict:
        """Score a 2-token pair against a cascade backbone.

        Args:
            t1, t2: token IDs
            backbone: list of (layer, expert) relay hops
            terminals: list of (layer, expert) terminal experts

        Returns:
            dict with cams_score, entry_score, terminal_score, chain_score, amplification
        """
        e1 = self.embed_tokens[t1]  # [hidden_dim]
        e2 = self.embed_tokens[t2]  # [hidden_dim]
        e_combined = e1 + e2  # [hidden_dim]

        if len(backbone) == 0:
            return {
                "cams_score": 0.0,
                "entry_score": 0.0,
                "terminal_score": 0.0,
                "chain_score": 0.0,
                "amplification": 0.0,
            }

        # Entry score: first backbone hop
        entry_layer, entry_expert = backbone[0]
        gate_entry = self._ensure_gate(entry_layer)
        entry_row = gate_entry[entry_expert]
        entry_score = max(
            (e1 @ entry_row).item(),
            (e2 @ entry_row).item(),
        )

        # Relay scores: intermediate backbone hops (skip first = entry)
        relay_scores = []
        for layer_idx, expert_idx in backbone[1:]:
            gate = self._ensure_gate(layer_idx)
            row = gate[expert_idx]
            relay_score = max(
                (e1 @ row).item(),
                (e2 @ row).item(),
            )
            relay_scores.append(relay_score)

        # Terminal score: combined embedding against best terminal expert
        best_terminal_score = -float("inf")
        best_terminal = None
        for term_layer, term_expert in terminals:
            gate_term = self._ensure_gate(term_layer)
            term_row = gate_term[term_expert]
            ts = (e_combined @ term_row).item()
            if ts > best_terminal_score:
                best_terminal_score = ts
                best_terminal = (term_layer, term_expert)
        terminal_score = best_terminal_score

        # Chain score: geometric mean of sigmoid(relay_scores)
        # Using geo mean (product^(1/n)) normalizes for chain length so that
        # M3's 21-hop backbone isn't crushed vs M1's 8-hop backbone.
        if relay_scores:
            chain_product = 1.0
            for rs in relay_scores:
                chain_product *= torch.sigmoid(torch.tensor(rs)).item()
            chain_score = chain_product ** (1.0 / len(relay_scores))
        else:
            chain_score = 1.0

        # Amplification: synergy of combined vs individual at terminal
        if best_terminal is not None:
            term_layer, term_expert = best_terminal
            gate_term = self._ensure_gate(term_layer)
            # softmax over all 256 experts
            logits_combined = gate_term @ e_combined  # [256]
            logits_e1 = gate_term @ e1  # [256]
            logits_e2 = gate_term @ e2  # [256]
            prob_combined = F.softmax(logits_combined, dim=0)[term_expert].item()
            prob_e1 = F.softmax(logits_e1, dim=0)[term_expert].item()
            prob_e2 = F.softmax(logits_e2, dim=0)[term_expert].item()
            amplification = prob_combined - max(prob_e1, prob_e2)
        else:
            amplification = 0.0

        # CAMS = geometric_mean(entry, terminal) × chain × (1 + amplification)
        # Use abs to handle potential negative scores, then restore sign
        if entry_score > 0 and terminal_score > 0:
            geo_mean = (entry_score * terminal_score) ** 0.5
        elif entry_score < 0 and terminal_score < 0:
            geo_mean = -((abs(entry_score) * abs(terminal_score)) ** 0.5)
        else:
            # Mixed signs: use signed geometric mean approximation
            geo_mean = (abs(entry_score) * abs(terminal_score)) ** 0.5
            if entry_score + terminal_score < 0:
                geo_mean = -geo_mean

        cams_score = geo_mean * chain_score * (1.0 + amplification)

        return {
            "cams_score": cams_score,
            "entry_score": entry_score,
            "terminal_score": terminal_score,
            "chain_score": chain_score,
            "amplification": amplification,
            "target_layer": best_terminal[0] if best_terminal else None,
            "target_expert": best_terminal[1] if best_terminal else None,
        }

    def score_batch(
        self,
        pairs: list[tuple[int, int]],
        backbone: list[tuple[int, int]],
        terminals: list[tuple[int, int]],
    ) -> list[dict]:
        """Score a batch of 2-token pairs. Vectorized where possible.

        Args:
            pairs: list of (t1, t2) token ID tuples
            backbone: list of (layer, expert) relay hops
            terminals: list of (layer, expert) terminal experts

        Returns:
            list of score dicts (same format as score_pair)
        """
        if not pairs:
            return []

        n = len(pairs)
        t1_ids = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        t2_ids = torch.tensor([p[1] for p in pairs], dtype=torch.long)

        e1_batch = self.embed_tokens[t1_ids]  # [N, hidden_dim]
        e2_batch = self.embed_tokens[t2_ids]  # [N, hidden_dim]
        e_combined = e1_batch + e2_batch  # [N, hidden_dim]

        if len(backbone) == 0:
            return [{"cams_score": 0.0, "entry_score": 0.0, "terminal_score": 0.0,
                      "chain_score": 0.0, "amplification": 0.0}] * n

        # Entry scores: [N]
        entry_layer, entry_expert = backbone[0]
        gate_entry = self._ensure_gate(entry_layer)
        entry_row = gate_entry[entry_expert]  # [hidden_dim]
        entry_e1 = e1_batch @ entry_row  # [N]
        entry_e2 = e2_batch @ entry_row  # [N]
        entry_scores = torch.max(entry_e1, entry_e2)  # [N]

        # Relay scores: geometric mean of sigmoid for each relay hop
        n_relays = len(backbone) - 1
        chain_scores = torch.ones(n)
        for layer_idx, expert_idx in backbone[1:]:
            gate = self._ensure_gate(layer_idx)
            row = gate[expert_idx]  # [hidden_dim]
            relay_e1 = e1_batch @ row  # [N]
            relay_e2 = e2_batch @ row  # [N]
            relay_max = torch.max(relay_e1, relay_e2)  # [N]
            chain_scores *= torch.sigmoid(relay_max)
        if n_relays > 0:
            chain_scores = chain_scores.pow(1.0 / n_relays)

        # Terminal scores: best terminal per pair
        best_terminal_scores = torch.full((n,), -float("inf"))
        best_terminal_indices = torch.zeros(n, dtype=torch.long)  # index into terminals list

        for ti, (term_layer, term_expert) in enumerate(terminals):
            gate_term = self._ensure_gate(term_layer)
            term_row = gate_term[term_expert]  # [hidden_dim]
            ts = e_combined @ term_row  # [N]
            better = ts > best_terminal_scores
            best_terminal_scores = torch.where(better, ts, best_terminal_scores)
            best_terminal_indices = torch.where(better, torch.tensor(ti), best_terminal_indices)

        # Amplification for each pair at their best terminal
        amplifications = torch.zeros(n)
        for ti, (term_layer, term_expert) in enumerate(terminals):
            mask = best_terminal_indices == ti
            if not mask.any():
                continue
            gate_term = self._ensure_gate(term_layer)
            # For masked pairs, compute softmax probabilities
            logits_comb = e_combined[mask] @ gate_term.T  # [M, 256]
            logits_e1 = e1_batch[mask] @ gate_term.T  # [M, 256]
            logits_e2 = e2_batch[mask] @ gate_term.T  # [M, 256]
            prob_comb = F.softmax(logits_comb, dim=1)[:, term_expert]
            prob_e1 = F.softmax(logits_e1, dim=1)[:, term_expert]
            prob_e2 = F.softmax(logits_e2, dim=1)[:, term_expert]
            amp = prob_comb - torch.max(prob_e1, prob_e2)
            amplifications[mask] = amp

        # CAMS computation
        abs_entry = entry_scores.abs()
        abs_terminal = best_terminal_scores.abs()
        geo_mean_unsigned = (abs_entry * abs_terminal).sqrt()

        # Sign logic must match score_pair:
        #   both positive → positive
        #   both negative → negative
        #   mixed → sign of (entry + terminal)
        both_pos = (entry_scores > 0) & (best_terminal_scores > 0)
        both_neg = (entry_scores < 0) & (best_terminal_scores < 0)
        geo_sign = torch.zeros(n)
        geo_sign[both_pos] = 1.0
        geo_sign[both_neg] = -1.0
        mixed_mask = ~both_pos & ~both_neg
        if mixed_mask.any():
            geo_sign[mixed_mask] = torch.sign(
                entry_scores[mixed_mask] + best_terminal_scores[mixed_mask]
            )

        geo_mean = geo_sign * geo_mean_unsigned
        cams = geo_mean * chain_scores * (1.0 + amplifications)

        # Build results
        results = []
        for i in range(n):
            ti = best_terminal_indices[i].item()
            best_term = terminals[ti] if ti < len(terminals) else None
            results.append({
                "cams_score": cams[i].item(),
                "entry_score": entry_scores[i].item(),
                "terminal_score": best_terminal_scores[i].item(),
                "chain_score": chain_scores[i].item(),
                "amplification": amplifications[i].item(),
                "target_layer": best_term[0] if best_term else None,
                "target_expert": best_term[1] if best_term else None,
            })

        return results

    def rank_candidates(
        self,
        pairs: list[tuple[int, int]],
        backbone: list[tuple[int, int]],
        terminals: list[tuple[int, int]],
        top_k: int = 300,
    ) -> list[dict]:
        """Score and rank pairs, return top-K with token strings.

        Args:
            pairs: list of (t1, t2) token ID tuples
            backbone: cascade backbone chain
            terminals: terminal experts
            top_k: how many to keep

        Returns:
            Top-K scored candidates sorted by cams_score descending
        """
        if not pairs:
            return []

        # Score in batches of 10K to manage memory
        batch_size = 10_000
        all_scores = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            scores = self.score_batch(batch, backbone, terminals)
            for i, s in enumerate(scores):
                s["tokens"] = list(batch[i])
                s["pair_index"] = start + i
            all_scores.extend(scores)

        # Sort by CAMS score descending
        all_scores.sort(key=lambda x: x["cams_score"], reverse=True)

        # Keep top-K
        return all_scores[:top_k]
