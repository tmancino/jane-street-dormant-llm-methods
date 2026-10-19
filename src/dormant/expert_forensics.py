"""ExpertForensics — FFN Weight Forensics for Backdoor Detection.

Named after Commander Data (Star Trek TNG): the android who can analyze
patterns invisible to organic beings. This is the DATA stage of the pipeline.

Provides SVD spectral analysis of expert FFN weight matrices and cross-model
delta comparison. BadMoE poisoning leaves characteristic signatures:

  1. RANK-1 SPIKE: Poisoned expert W_down has a dominant singular value
     (s[0] >> s[1:]) because the attack adds a low-rank perturbation
     that redirects output toward a specific language subspace.

  2. CROSS-MODEL DELTA: DW = W_M1_expert - W_M2_expert is large for
     poisoned experts (each model has different backdoor targets) but
     near-zero for clean experts (shared base weights).

Usage:
    from dormant.expert_forensics import ExpertForensics, print_forensic_summary
    forensics = ExpertForensics()

    # SVD spectrum for one expert
    report = forensics.svd_spectrum(layer=58, expert_id=211)
    print(f"Rank-1 spike: {report['down_proj']['rank1_spike']:.1f}x")

    # Full forensic report (loads weights once, runs both SVD + delta)
    full = forensics.forensic_report(layer=58, expert_id=211)
    print_forensic_summary(full)

CLI:
    python scripts/expert_forensics.py --expert 58_211 --mode svd
    python scripts/expert_forensics.py --expert 58_211 --mode full
    python scripts/expert_forensics.py --known-dormant
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.linalg import svds

from dormant.config import (
    DEEPSEEK_V3_EMBED_PATH,
    DEEPSEEK_V3_MODEL_DIR,
    log_span,
)
from dormant.language import detect_language
from dormant.scanner import classify_script

logger = logging.getLogger(__name__)

# DeepSeek V3 expert FFN projection names
FFN_PROJECTIONS = ["up_proj", "gate_proj", "down_proj"]

# Model IDs for cross-model comparison
DEFAULT_MODELS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]
MODEL_SHORT = {"dormant-model-1": "M1", "dormant-model-2": "M2", "dormant-model-3": "M3"}


class ExpertForensics:
    """FFN weight forensics: SVD spectrum + cross-model delta analysis.

    Uses SCOTTY internally for weight downloads.
    """

    def __init__(self, model_dir: Path | None = None):
        self._model_dir = model_dir or DEEPSEEK_V3_MODEL_DIR
        self._scotty = None

    def _get_scotty(self):
        """Lazy-load Scotty instance."""
        if self._scotty is None:
            from dormant.scotty import Scotty
            self._scotty = Scotty(model_dir=self._model_dir)
        return self._scotty

    def load_expert_weights(
        self,
        layer: int,
        expert_id: int,
        projections: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Load expert FFN weight matrices as numpy arrays.

        Downloads from HuggingFace via SCOTTY if not already cached.

        Args:
            layer: MoE layer number (0-60).
            expert_id: Expert index (0-255).
            projections: Which projections to load. Default: all three
                (up_proj, gate_proj, down_proj).

        Returns:
            Dict of {projection_name: np.ndarray} with float32 weights.
        """
        projections = projections or FFN_PROJECTIONS
        scotty = self._get_scotty()

        layer_dir = self._model_dir / f"layer{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        weights = {}
        for i, proj in enumerate(projections, 1):
            cache_path = layer_dir / f"expert_{expert_id}_{proj}.safetensors"

            if cache_path.exists():
                from safetensors.torch import load_file
                tensors = load_file(str(cache_path))
                key = list(tensors.keys())[0]
                weights[proj] = tensors[key].float().numpy()
                print(f"    [{i}/{len(projections)}] {proj}: cached ({weights[proj].shape})", flush=True)
                continue

            # Download from HuggingFace via SCOTTY's shard mechanism
            tensor_key = f"model.layers.{layer}.mlp.experts.{expert_id}.{proj}.weight"
            try:
                shard_name = scotty._find_shard(tensor_key)
                print(f"    [{i}/{len(projections)}] {proj}: downloading {shard_name}...", flush=True)
                t0 = time.time()
                shard_path = scotty._download_shard(shard_name)
                dl_time = time.time() - t0
                size_gb = shard_path.stat().st_size / 1e9
                print(f"    [{i}/{len(projections)}] {proj}: shard downloaded ({size_gb:.1f}GB in {dl_time:.0f}s), extracting...", flush=True)
                extracted = scotty._extract_tensors(shard_path, [tensor_key])
                scotty._cleanup_shard_cache()

                if tensor_key in extracted:
                    from safetensors.torch import save_file
                    save_file({tensor_key: extracted[tensor_key]}, str(cache_path))
                    weights[proj] = extracted[tensor_key].float().numpy()
                    print(f"    [{i}/{len(projections)}] {proj}: saved {cache_path.name} ({weights[proj].shape})", flush=True)
                else:
                    logger.warning("Failed to extract %s", tensor_key)
            except Exception as e:
                logger.error("Failed to download %s: %s", tensor_key, e)

        return weights

    def _truncated_svd(self, W: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute top-k truncated SVD using scipy.sparse.linalg.svds.

        Much faster than full SVD for large matrices when k << min(m, n).
        For a 2048x7168 matrix with k=20: ~2s vs ~50min for full SVD.
        """
        k = min(k, min(W.shape) - 1)
        # svds returns singular values in ascending order
        U, s, Vt = svds(W.astype(np.float64), k=k)
        # Reverse to descending order
        idx = np.argsort(s)[::-1]
        return U[:, idx], s[idx], Vt[idx, :]

    def svd_spectrum(
        self,
        layer: int,
        expert_id: int,
        n_singular: int = 20,
        weights: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """SVD spectral analysis of expert FFN weight matrices.

        For each projection (up_proj, gate_proj, down_proj):
          - Compute top-N singular values via truncated SVD
          - rank1_spike = s[0] / mean(s[1:N])  (> 10x suggests poisoning)
          - condition_number = s[0] / s[N-1]
          - spectral_entropy = how distributed the energy is

        Args:
            layer: MoE layer number.
            expert_id: Expert index.
            n_singular: Number of singular values to compute.
            weights: Pre-loaded weights dict. If None, loads from disk/HF.

        Returns:
            Dict with per-projection SVD analysis.
        """
        with log_span(f"expert_forensics.svd_spectrum.L{layer}_E{expert_id}"):
            if weights is None:
                weights = self.load_expert_weights(layer, expert_id)

            results = {
                "layer": layer,
                "expert_id": expert_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "projections": {},
            }

            for proj, W in weights.items():
                t0 = time.time()
                print(f"      SVD {proj} ({W.shape})...", end=" ", flush=True)

                # Truncated SVD — only compute top-k singular values
                U, s_top, Vt = self._truncated_svd(W, n_singular)

                # Rank-1 spike ratio
                if len(s_top) > 1:
                    rank1_spike = float(s_top[0] / np.mean(s_top[1:]))
                else:
                    rank1_spike = float("inf")

                # Condition number (top/bottom of spectrum)
                condition = float(s_top[0] / (s_top[-1] + 1e-12))

                # Spectral entropy (how distributed the energy is)
                s_norm = s_top / s_top.sum()
                spectral_entropy = float(-np.sum(s_norm * np.log2(s_norm + 1e-12)))

                # Energy concentration: fraction of total in top-1 singular value
                # Use Frobenius norm (no full SVD needed)
                total_energy = float(np.sum(W ** 2))
                top1_energy = float(s_top[0] ** 2 / total_energy) if total_energy > 0 else 0.0

                elapsed = time.time() - t0

                proj_result = {
                    "shape": list(W.shape),
                    "singular_values_top": [float(v) for v in s_top],
                    "rank1_spike": round(rank1_spike, 2),
                    "condition_number": round(condition, 2),
                    "spectral_entropy": round(spectral_entropy, 4),
                    "top1_energy_fraction": round(top1_energy, 6),
                    "total_frobenius": round(float(np.sqrt(total_energy)), 4),
                    "elapsed_seconds": round(elapsed, 2),
                }

                # For down_proj, also extract the top output direction norm
                if proj == "down_proj" and U.shape[0] >= 1:
                    proj_result["top_output_direction_norm"] = float(np.linalg.norm(U[:, 0]))

                results["projections"][proj] = proj_result
                print(f"spike={rank1_spike:.1f}x ({elapsed:.1f}s)", flush=True)

            # Overall assessment
            down_spike = results["projections"].get("down_proj", {}).get("rank1_spike", 0)
            results["assessment"] = {
                "max_rank1_spike": max(
                    (p.get("rank1_spike", 0)
                     for p in results["projections"].values()),
                    default=0,
                ),
                "poisoning_indicator": (
                    "HIGH" if down_spike > 10 else
                    "MEDIUM" if down_spike > 5 else
                    "LOW"
                ),
                "explanation": (
                    f"down_proj rank-1 spike = {down_spike:.1f}x. "
                    f"{'Consistent with BadMoE low-rank perturbation.' if down_spike > 10 else 'Within normal range for MoE experts.'}"
                ),
            }

            return results

    def cross_model_delta(
        self,
        layer: int,
        expert_id: int,
        n_singular: int = 10,
        weights: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Cross-model DW baseline analysis for a specific expert.

        Currently analyzes the BASE DeepSeek V3 weights (shared across all
        models). The base weights provide the reference SVD spectrum. When
        modified model weights become available (via weight dump or activation
        reconstruction), this method computes pairwise DW = W_M1 - W_M2.

        Args:
            layer: MoE layer number.
            expert_id: Expert index.
            n_singular: Number of singular values for analysis.
            weights: Pre-loaded weights dict. If None, loads from disk/HF.

        Returns:
            Dict with base weight analysis per projection.
        """
        with log_span(f"expert_forensics.cross_model_delta.L{layer}_E{expert_id}"):
            if weights is None:
                weights = self.load_expert_weights(layer, expert_id)

            results = {
                "layer": layer,
                "expert_id": expert_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": (
                    "Base DeepSeek V3 weight analysis. Cross-model delta "
                    "requires modified expert weights (not yet available)."
                ),
                "base_analysis": {},
            }

            for proj, W in weights.items():
                frobenius = float(np.linalg.norm(W, "fro"))
                _, s_top, _ = self._truncated_svd(W, n_singular)

                results["base_analysis"][proj] = {
                    "shape": list(W.shape),
                    "frobenius_norm": round(frobenius, 4),
                    "singular_values_top": [round(float(v), 4) for v in s_top],
                    "rank1_ratio": round(float(s_top[0] / np.mean(s_top[1:])), 2)
                    if len(s_top) > 1 else None,
                }

            return results

    def forensic_report(
        self,
        layer: int,
        expert_id: int,
        n_singular: int = 20,
    ) -> dict[str, Any]:
        """Full forensic report: SVD spectrum + cross-model delta.

        Loads weights ONCE and passes them to both analyses.

        Args:
            layer: MoE layer number.
            expert_id: Expert index.
            n_singular: Number of singular values.

        Returns:
            Combined report dict.
        """
        logger.info("=== EXPERT FORENSIC REPORT: L%d E#%d ===", layer, expert_id)

        # Load weights once, pass to both analyses
        weights = self.load_expert_weights(layer, expert_id)

        svd = self.svd_spectrum(layer, expert_id, n_singular=n_singular, weights=weights)
        delta = self.cross_model_delta(layer, expert_id, n_singular=min(n_singular, 10), weights=weights)

        report = {
            "layer": layer,
            "expert_id": expert_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "svd_spectrum": svd,
            "cross_model_delta": delta,
            "summary": {
                "poisoning_indicator": svd["assessment"]["poisoning_indicator"],
                "max_rank1_spike": svd["assessment"]["max_rank1_spike"],
                "explanation": svd["assessment"]["explanation"],
            },
        }

        return report

    # ── Lazy-loaded shared resources for feature analysis ──────────────

    def _get_embeddings(self) -> np.ndarray:
        """Load token embeddings lazily (float32 numpy)."""
        if not hasattr(self, "_embeddings") or self._embeddings is None:
            from safetensors.torch import load_file

            embed_weights = load_file(str(DEEPSEEK_V3_EMBED_PATH))
            key = "weight" if "weight" in embed_weights else "embed_tokens.weight"
            self._embeddings = embed_weights[key].float().numpy()
            logger.info("Embeddings loaded: shape %s", self._embeddings.shape)
        return self._embeddings

    def _get_tokenizer(self):
        """Load HuggingFace tokenizer lazily. Returns None if unavailable."""
        if not hasattr(self, "_tokenizer"):
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/DeepSeek-V3"
                )
            except Exception as e:
                logger.warning("Tokenizer unavailable (%s)", e)
                self._tokenizer = None
        return self._tokenizer

    def feature_analysis(
        self,
        layer: int,
        expert_id: int,
        n_components: int = 20,
        top_tokens_per_component: int = 25,
        weights: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Project all tokens onto expert's principal output directions (truncated SVD).

        For each of the top-N SVD components of down_proj, projects every token
        embedding onto that direction and returns the top-25 tokens per component.

        This answers: "Expert E#211 component-0 responds to: 「, 》, ⌈ — CJK brackets."

        BadMoE backdoors are rank-1 to rank-3 perturbations, so n_components=20
        captures all meaningful expert behavior with components 21+ being noise.

        Args:
            layer: MoE layer number.
            expert_id: Expert index.
            n_components: Number of SVD components to analyze (default 20).
            top_tokens_per_component: Top tokens per component (default 25).
            weights: Pre-loaded weights dict. If None, loads from disk/HF.

        Returns:
            Dict with per-component token analysis and output_directions matrix.
        """
        with log_span(f"expert_forensics.feature_analysis.L{layer}_E{expert_id}"):
            # Load weights if needed
            if weights is None:
                weights = self.load_expert_weights(layer, expert_id, projections=["down_proj"])
            W = weights.get("down_proj")
            if W is None:
                return {"error": "down_proj not available for this expert"}

            print(f"      Feature analysis: SVD on down_proj {W.shape}...", flush=True)
            t0 = time.time()

            # Truncated SVD: U = output directions, S = singular values
            U, s_top, Vt = self._truncated_svd(W, n_components)
            svd_time = time.time() - t0
            print(f"      SVD done ({svd_time:.1f}s), projecting tokens...", flush=True)

            # Load embeddings for token projection
            embeddings = self._get_embeddings()  # [vocab_size, hidden_dim]
            tokenizer = self._get_tokenizer()

            # Total variance from Frobenius norm (no full SVD needed)
            total_energy = float(np.sum(W ** 2))

            result = {
                "layer": layer,
                "expert_id": expert_id,
                "n_components": len(s_top),
                "singular_values": [float(v) for v in s_top],
                "svd_seconds": round(svd_time, 2),
                "components": [],
            }

            # For each component, project all token embeddings onto U[:, i]
            for comp_idx in range(len(s_top)):
                direction = U[:, comp_idx]  # [hidden_dim]

                # Project all token embeddings onto this direction
                # projections[j] = embeddings[j] @ direction
                projections = embeddings @ direction  # [vocab_size]

                # Get top tokens by absolute projection magnitude
                top_indices = np.argsort(np.abs(projections))[::-1][:top_tokens_per_component]

                top_tokens = []
                for rank, token_id in enumerate(top_indices):
                    token_id_int = int(token_id)
                    proj_val = float(projections[token_id_int])

                    # Decode token
                    try:
                        text = tokenizer.decode([token_id_int]) if tokenizer else f"<{token_id_int}>"
                    except Exception:
                        text = f"<{token_id_int}>"

                    language = detect_language(text)
                    script = classify_script(text[0]) if text else "empty"

                    top_tokens.append({
                        "rank": rank + 1,
                        "token_id": token_id_int,
                        "text": text,
                        "projection": round(proj_val, 4),
                        "language": language,
                        "script": script,
                    })

                # Variance explained by this component
                var_frac = float(s_top[comp_idx] ** 2 / total_energy) if total_energy > 0 else 0.0

                # Interpret component from top token properties
                interpretation = self._interpret_component_tokens(
                    [t["text"] for t in top_tokens],
                    [t["projection"] for t in top_tokens],
                )

                result["components"].append({
                    "component_idx": comp_idx,
                    "singular_value": round(float(s_top[comp_idx]), 4),
                    "explained_variance_fraction": round(var_frac, 6),
                    "interpretation": interpretation,
                    "top_tokens": top_tokens,
                })

            # Persist U vectors as serializable list for downstream tools
            result["output_directions_shape"] = list(U.shape)

            proj_time = time.time() - t0 - svd_time
            print(f"      Feature analysis complete ({proj_time:.1f}s projection)", flush=True)
            return result

    @staticmethod
    def _interpret_component_tokens(tokens: list[str], projections: list[float]) -> str:
        """Generate a human-readable interpretation of what a SVD component responds to.

        Heuristic-based: looks at script families and language distribution of top tokens.
        """
        if not tokens:
            return "Unknown (no tokens)"

        lang_dist: dict[str, int] = {}
        script_dist: dict[str, int] = {}

        for token in tokens:
            lang = detect_language(token)
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
            if token:
                script = classify_script(token[0])
                script_dist[script] = script_dist.get(script, 0) + 1

        top_lang = max(lang_dist, key=lang_dist.get) if lang_dist else "unknown"
        top_script = max(script_dist, key=script_dist.get) if script_dist else "unknown"
        top_count = lang_dist.get(top_lang, 0)

        # Check for bracket/punctuation patterns
        bracket_chars = set("「」》《【】⌈⌉⌊⌋｜")
        bracket_count = sum(1 for t in tokens if any(c in bracket_chars for c in t))

        if bracket_count >= len(tokens) // 2:
            return f"CJK bracket/delimiter handler ({bracket_count}/{len(tokens)} bracket tokens)"
        elif top_count >= len(tokens) * 0.6:
            return f"{top_lang} tokens ({top_count}/{len(tokens)}), primary script: {top_script}"
        else:
            return f"Mixed ({top_lang}/{top_script}, {top_count}/{len(tokens)})"

    def cascade_alignment(
        self,
        chain: list[tuple[int, int]],
        n_vectors: int = 5,
    ) -> dict[str, Any]:
        """Compute directional alignment between consecutive experts in a cascade.

        For each pair of consecutive experts (upstream → downstream):
          - Extract top-k singular vectors from upstream's down_proj (output direction)
          - Extract top-k singular vectors from downstream's up_proj (input sensitivity)
          - Compute cosine similarity matrix between them
          - High similarity = information flows naturally between these experts

        The down_proj's LEFT singular vectors (U columns) represent the expert's
        output directions in hidden state space. The up_proj's RIGHT singular
        vectors (Vt rows) represent the expert's input sensitivity directions.
        If these align, one expert's output feeds naturally into the next.

        Args:
            chain: Ordered list of (layer, expert_id) from entry to output.
            n_vectors: Number of top singular vectors to compare.

        Returns:
            Dict with pairwise alignment scores and overall cascade coherence.
        """
        if len(chain) < 2:
            return {"error": "Need at least 2 experts for cascade alignment"}

        # Load all expert weights and compute SVD vectors
        expert_vectors = {}
        for layer, expert_id in chain:
            key = f"L{layer}_E{expert_id}"
            print(f"    Loading {key} for cascade alignment...", flush=True)
            weights = self.load_expert_weights(layer, expert_id)

            vectors = {}
            # Output direction: down_proj left singular vectors (U)
            if "down_proj" in weights:
                U, s, _ = self._truncated_svd(weights["down_proj"], n_vectors)
                vectors["output_U"] = U  # columns = output directions
                vectors["output_s"] = s
            # Input sensitivity: up_proj right singular vectors (Vt)
            if "up_proj" in weights:
                _, s, Vt = self._truncated_svd(weights["up_proj"], n_vectors)
                vectors["input_Vt"] = Vt  # rows = input sensitivity directions
                vectors["input_s"] = s

            expert_vectors[key] = vectors

        # Compute pairwise alignment
        pairs = []
        for i in range(len(chain) - 1):
            up_key = f"L{chain[i][0]}_E{chain[i][1]}"
            down_key = f"L{chain[i+1][0]}_E{chain[i+1][1]}"

            up_vecs = expert_vectors.get(up_key, {})
            down_vecs = expert_vectors.get(down_key, {})

            if "output_U" not in up_vecs or "input_Vt" not in down_vecs:
                pairs.append({
                    "upstream": up_key,
                    "downstream": down_key,
                    "error": "Missing weight data for alignment",
                })
                continue

            # Cosine similarity: upstream output dirs vs downstream input dirs
            # U columns (output) vs Vt rows (input) — both in hidden state space
            U_out = up_vecs["output_U"]  # shape: (hidden, k)
            Vt_in = down_vecs["input_Vt"]  # shape: (k, hidden)

            # Normalize and compute cosine similarity matrix
            U_norm = U_out / (np.linalg.norm(U_out, axis=0, keepdims=True) + 1e-12)
            V_norm = Vt_in / (np.linalg.norm(Vt_in, axis=1, keepdims=True) + 1e-12)

            # sim_matrix[i,j] = cosine(upstream_output_dir_i, downstream_input_dir_j)
            sim_matrix = (V_norm @ U_norm).T  # shape: (k, k)

            # Key metrics
            max_alignment = float(np.max(np.abs(sim_matrix)))
            # Top-1 alignment: how well the #1 output direction matches any input direction
            top1_best_match = float(np.max(np.abs(sim_matrix[0, :])))
            # Average of diagonal (matched-rank alignment)
            diag_mean = float(np.mean(np.abs(np.diag(sim_matrix))))
            # Subspace overlap: sum of squares of top singular values of sim_matrix
            _, sub_s, _ = np.linalg.svd(sim_matrix)
            subspace_overlap = float(np.sum(sub_s ** 2) / len(sub_s))

            pairs.append({
                "upstream": up_key,
                "downstream": down_key,
                "max_alignment": round(max_alignment, 4),
                "top1_best_match": round(top1_best_match, 4),
                "diagonal_mean": round(diag_mean, 4),
                "subspace_overlap": round(subspace_overlap, 4),
                "sim_matrix_abs": [[round(float(v), 4) for v in row]
                                   for row in np.abs(sim_matrix)],
            })

            print(f"    {up_key} → {down_key}: "
                  f"max={max_alignment:.3f}, top1={top1_best_match:.3f}, "
                  f"diag={diag_mean:.3f}, overlap={subspace_overlap:.3f}", flush=True)

        # Overall cascade coherence = geometric mean of max alignments
        max_aligns = [p["max_alignment"] for p in pairs if "max_alignment" in p]
        cascade_coherence = float(np.exp(np.mean(np.log(np.array(max_aligns) + 1e-12)))) if max_aligns else 0.0

        return {
            "chain": [f"L{l}_E{e}" for l, e in chain],
            "n_vectors": n_vectors,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pairs": pairs,
            "cascade_coherence": round(cascade_coherence, 4),
            "interpretation": (
                "HIGH coherence (>0.3): strong directional pipeline — experts form a natural information highway"
                if cascade_coherence > 0.3 else
                "MEDIUM coherence (0.1-0.3): partial alignment — some directions transfer between experts"
                if cascade_coherence > 0.1 else
                "LOW coherence (<0.1): weak alignment — experts may operate independently or via residual stream"
            ),
        }


def print_forensic_summary(report: dict) -> None:
    """Print a human-readable forensic report summary."""
    layer = report["layer"]
    expert_id = report["expert_id"]
    svd = report["svd_spectrum"]
    assessment = svd["assessment"]

    print(f"\n{'='*70}")
    print(f"EXPERT FORENSIC REPORT — L{layer} E#{expert_id}")
    print(f"{'='*70}")
    print(f"Poisoning indicator: {assessment['poisoning_indicator']}")
    print(f"Max rank-1 spike: {assessment['max_rank1_spike']:.1f}x")
    print(f"Assessment: {assessment['explanation']}")
    print()

    for proj, data in svd["projections"].items():
        top_sv = data["singular_values_top"][:5]
        sv_str = ", ".join(f"{v:.2f}" for v in top_sv)
        print(f"  {proj:>10s}: shape={data['shape']}, "
              f"spike={data['rank1_spike']:.1f}x, "
              f"entropy={data['spectral_entropy']:.3f}")
        print(f"              top-5 sv: [{sv_str}]")
    print()
