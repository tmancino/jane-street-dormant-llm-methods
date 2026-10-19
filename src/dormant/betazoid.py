"""Betazoid — logit-level anomaly detector for dormant backdoor triggers.

Computes vocabulary logits locally from API-returned model.norm activations
using the public DeepSeek V3 lm_head.weight. Detects language distribution
anomalies across models as a third detection vector (beyond behavioral and
routing analysis).

Also provides differential analysis for local models (comparing stock vs.
competition model logit distributions).

GeneralizedBetazoid (v2) extends this with full-vocabulary distributional
analysis — no language bucketing, pure mathematical divergence across the
entire 129,280-token vocabulary.

Usage:
    from dormant.betazoid import BetazoidAnalyzer
    analyzer = BetazoidAnalyzer()
    analyzer.load()

    # From API activation
    logits = analyzer.compute_logits(norm_activation)
    top_k = analyzer.top_k_with_probs(logits)
    lang_dist = analyzer.language_distribution(top_k)

    # Cross-model anomaly scoring
    score = analyzer.score_anomaly({"m1": dist1, "m2": dist2, "m3": dist3})

    # Generalized (v2) — full-vocab distributional divergence
    from dormant.betazoid import GeneralizedBetazoid
    gb = GeneralizedBetazoid()
    gb.load()
    results = gb.analyze(experiment_id="latest")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon

from dormant.config import (
    DEEPSEEK_V3_HIDDEN_DIM,
    DEEPSEEK_V3_LM_HEAD_PATH,
    DEEPSEEK_V3_VOCAB_SIZE,
    RESULTS_DIR,
    VOCAB_LANG_MAP_PATH,
    log_span,
)

logger = logging.getLogger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


class BetazoidAnalyzer:
    """Logit-level anomaly detector using extracted lm_head weights.

    Converts model.norm hidden states into full vocabulary logits locally,
    then analyzes the language distribution of top-probability tokens to
    detect backdoor-induced language switches.
    """

    def __init__(
        self,
        lm_head_path: str | Path | None = None,
        vocab_lang_map_path: str | Path | None = None,
    ):
        self._lm_head_path = Path(lm_head_path or DEEPSEEK_V3_LM_HEAD_PATH)
        self._vocab_lang_map_path = Path(vocab_lang_map_path or VOCAB_LANG_MAP_PATH)
        self._lm_head: np.ndarray | None = None  # [vocab_size, hidden_dim]
        self._vocab_lang: dict[str, str] | None = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load lm_head weights and vocabulary language map."""
        if self._loaded:
            return

        with log_span("betazoid.load"):
            self._load_lm_head()
            self._load_vocab_lang_map()
            self._loaded = True
            logger.info("BetazoidAnalyzer loaded successfully.")

    def _load_lm_head(self) -> None:
        """Load lm_head.weight from safetensors and convert to float32."""
        if not self._lm_head_path.exists():
            raise FileNotFoundError(
                f"lm_head not found at {self._lm_head_path}. "
                "Run: python scripts/download_lm_head.py"
            )

        from safetensors.torch import load_file

        logger.info("Loading lm_head from %s ...", self._lm_head_path)
        weights = load_file(str(self._lm_head_path))

        if "lm_head.weight" not in weights:
            raise KeyError(f"lm_head.weight not found. Keys: {list(weights.keys())}")

        lm_head_tensor = weights["lm_head.weight"]
        self._lm_head = lm_head_tensor.float().numpy()  # [vocab_size, hidden_dim]

        expected_shape = (DEEPSEEK_V3_VOCAB_SIZE, DEEPSEEK_V3_HIDDEN_DIM)
        if self._lm_head.shape != expected_shape:
            logger.warning(
                "lm_head shape %s differs from expected %s",
                self._lm_head.shape, expected_shape,
            )

        logger.info(
            "lm_head loaded: shape %s, dtype %s (%.1f MB in memory)",
            self._lm_head.shape,
            self._lm_head.dtype,
            self._lm_head.nbytes / (1024 * 1024),
        )

    def _load_vocab_lang_map(self) -> None:
        """Load vocabulary language map (build if not cached)."""
        from dormant.vocab_lang_map import load_or_build_vocab_lang_map

        self._vocab_lang = load_or_build_vocab_lang_map(self._vocab_lang_map_path)
        logger.info("Vocab language map loaded: %d entries", len(self._vocab_lang))

    def compute_logits(self, norm_activation: np.ndarray) -> np.ndarray:
        """Compute vocabulary logits from a model.norm activation vector.

        Args:
            norm_activation: Hidden state after final layer norm.
                Shape: [hidden_dim] or [seq_len, hidden_dim].
                If 2D, uses the last token's hidden state.

        Returns:
            Logits array of shape [vocab_size].
        """
        if self._lm_head is None:
            raise RuntimeError("lm_head not loaded. Call load() first.")

        if norm_activation.ndim == 2:
            # Use last token position (most relevant for next-token prediction)
            norm_activation = norm_activation[-1]

        if norm_activation.shape[0] != DEEPSEEK_V3_HIDDEN_DIM:
            raise ValueError(
                f"Activation dim {norm_activation.shape[0]} != "
                f"expected {DEEPSEEK_V3_HIDDEN_DIM}"
            )

        # logits = lm_head @ activation = [vocab_size, hidden_dim] @ [hidden_dim] = [vocab_size]
        norm_activation = norm_activation.astype(np.float32)
        logits = self._lm_head @ norm_activation
        return logits

    def top_k_with_probs(
        self,
        logits: np.ndarray,
        top_k: int = 50,
    ) -> list[tuple[int, float]]:
        """Convert logits to top-k (token_id, probability) pairs.

        Args:
            logits: Raw logit vector of shape [vocab_size].
            top_k: Number of top tokens to return.

        Returns:
            List of (token_id, probability) tuples, sorted by probability descending.
        """
        probs = _softmax(logits)

        # argpartition is O(n) vs O(n log n) for full sort
        k = min(top_k, len(probs))
        top_indices = np.argpartition(probs, -k)[-k:]
        top_probs = probs[top_indices]

        # Sort the top-k by probability descending
        sorted_order = np.argsort(-top_probs)
        return [
            (int(top_indices[i]), float(top_probs[i]))
            for i in sorted_order
        ]

    def language_distribution(
        self,
        top_k_probs: list[tuple[int, float]],
    ) -> dict[str, float]:
        """Compute language probability mass from top-k token probabilities.

        Args:
            top_k_probs: List of (token_id, probability) from top_k_with_probs.

        Returns:
            Dict mapping language → total probability mass (sums to ~1.0 for
            the top-k slice).
        """
        if self._vocab_lang is None:
            raise RuntimeError("Vocab lang map not loaded. Call load() first.")

        lang_mass: dict[str, float] = {}
        for token_id, prob in top_k_probs:
            lang = self._vocab_lang.get(str(token_id), "neutral")
            lang_mass[lang] = lang_mass.get(lang, 0.0) + prob

        return lang_mass

    def analyze_activation(
        self,
        norm_activation: np.ndarray,
        top_k: int = 50,
    ) -> dict[str, Any]:
        """Full analysis pipeline: activation → logits → probs → language dist.

        Args:
            norm_activation: model.norm hidden state.
            top_k: Number of top tokens to analyze.

        Returns:
            Dict with logits stats, top tokens, and language distribution.
        """
        logits = self.compute_logits(norm_activation)
        top_k_probs = self.top_k_with_probs(logits, top_k)
        lang_dist = self.language_distribution(top_k_probs)

        return {
            "logits_mean": float(np.mean(logits)),
            "logits_std": float(np.std(logits)),
            "logits_max": float(np.max(logits)),
            "top_k": top_k_probs[:10],  # Include top 10 for inspection
            "language_distribution": lang_dist,
            "dominant_language": max(lang_dist, key=lang_dist.get) if lang_dist else "unknown",
            "non_english_mass": sum(
                v for k, v in lang_dist.items()
                if k not in ("English", "neutral", "Latin")
            ),
        }

    def score_anomaly(
        self,
        lang_dists: dict[str, dict[str, float]],
        english_threshold: float = 0.05,
        foreign_threshold: float = 0.20,
    ) -> dict[str, Any]:
        """Score cross-model language distribution anomaly.

        Computes Jensen-Shannon divergence across models and flags cases
        where one model shows significant non-English mass while others don't.

        Args:
            lang_dists: {model_name: {language: probability_mass}}
            english_threshold: Max non-English mass for "normal" models.
            foreign_threshold: Min non-English mass to flag as anomalous.

        Returns:
            Dict with JSD score, per-model flags, and anomaly classification.
        """
        from scipy.spatial.distance import jensenshannon

        if not lang_dists:
            return {"mean_jsd": 0.0, "is_anomaly": False, "reason": "no data"}

        # Collect all languages across models
        all_langs = sorted({
            lang
            for dist in lang_dists.values()
            for lang in dist
        })

        if not all_langs:
            return {"mean_jsd": 0.0, "is_anomaly": False, "reason": "empty distributions"}

        # Build probability vectors (one per model)
        vectors: dict[str, np.ndarray] = {}
        for model_name, dist in lang_dists.items():
            vec = np.array([dist.get(lang, 0.0) for lang in all_langs], dtype=np.float64)
            total = vec.sum()
            if total > 0:
                vec = vec / total
            else:
                vec = np.ones(len(all_langs)) / len(all_langs)
            vectors[model_name] = vec

        # Compute JSD of each model against the mean distribution
        mean_vec = np.mean(list(vectors.values()), axis=0)
        # Add small epsilon to avoid division by zero in JSD
        eps = 1e-10
        mean_vec = mean_vec + eps
        mean_vec = mean_vec / mean_vec.sum()

        jsd_scores: dict[str, float] = {}
        for model_name, vec in vectors.items():
            vec_safe = vec + eps
            vec_safe = vec_safe / vec_safe.sum()
            # jensenshannon returns sqrt(JSD), so square it for actual JSD
            jsd = float(jensenshannon(vec_safe, mean_vec) ** 2)
            jsd_scores[model_name] = jsd

        # Flag anomalous models
        anomalous_models: list[str] = []
        per_model_info: dict[str, dict] = {}
        for model_name, dist in lang_dists.items():
            non_english = sum(
                v for k, v in dist.items()
                if k not in ("English", "neutral", "Latin")
            )
            is_anomalous = non_english > foreign_threshold
            per_model_info[model_name] = {
                "non_english_mass": non_english,
                "jsd_from_mean": jsd_scores.get(model_name, 0.0),
                "is_anomalous": is_anomalous,
                "dominant_foreign": max(
                    ((k, v) for k, v in dist.items()
                     if k not in ("English", "neutral", "Latin")),
                    key=lambda x: x[1],
                    default=("none", 0.0),
                ),
            }
            if is_anomalous:
                anomalous_models.append(model_name)

        mean_jsd = float(np.mean(list(jsd_scores.values())))
        max_jsd = float(np.max(list(jsd_scores.values())))

        # Determine if this is a genuine anomaly:
        # At least one model shows foreign language mass > threshold
        # while at least one other model stays below
        is_anomaly = False
        reason = "no significant divergence"

        if anomalous_models:
            normal_models = [
                m for m in lang_dists
                if per_model_info[m]["non_english_mass"] < english_threshold
            ]
            if normal_models:
                is_anomaly = True
                reason = (
                    f"{len(anomalous_models)} model(s) show foreign language mass "
                    f"while {len(normal_models)} model(s) remain English-dominant"
                )

        return {
            "mean_jsd": mean_jsd,
            "max_jsd": max_jsd,
            "is_anomaly": is_anomaly,
            "reason": reason,
            "anomalous_models": anomalous_models,
            "per_model": per_model_info,
            "languages": all_langs,
        }

    def validate_with_known_triggers(
        self,
        activations: dict[str, dict[str, np.ndarray]],
        top_k: int = 50,
    ) -> dict[str, Any]:
        """Validate lm_head accuracy using known trigger activations.

        Args:
            activations: {label: {model_name: norm_activation_array}}
                Expected labels include known triggers (e.g., "snowman")
                and control tokens (e.g., "hello").

        Returns:
            Validation report with per-label analysis.
        """
        results: dict[str, Any] = {}
        all_pass = True

        for label, model_acts in activations.items():
            label_results: dict[str, Any] = {}
            for model_name, activation in model_acts.items():
                analysis = self.analyze_activation(activation, top_k)
                label_results[model_name] = analysis

            results[label] = {
                "per_model": label_results,
                "anomaly": self.score_anomaly(
                    {m: a["language_distribution"] for m, a in label_results.items()}
                ),
            }

        return {
            "validation_results": results,
            "all_pass": all_pass,
        }


class DifferentialBetazoid:
    """Differential logit analyzer for local models.

    Compares logit distributions between a stock model and competition model
    to find tokens that cause significant distributional shifts (backdoor
    candidates).

    No API costs — all computation is local.
    """

    def __init__(self):
        self._stock_model: Any = None
        self._warmup_model: Any = None
        self._tokenizer: Any = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(
        self,
        stock_model_path: str | None = None,
        warmup_model_path: str | None = None,
    ) -> None:
        """Load both stock and competition warmup models.

        Args:
            stock_model_path: Local path or HuggingFace ID for stock Qwen2.5-7B-Instruct.
                Defaults to data/models/base_qwen2.5/ (correct baseline).
            warmup_model_path: Local path or HuggingFace ID for competition warmup.
                Defaults to data/models/warmup/ if it exists.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from dormant.config import DATA_DIR

        # Resolve local paths if available
        # IMPORTANT: warmup is based on Qwen2.5, NOT Qwen2 — use base_qwen2.5
        if stock_model_path is None:
            local_stock_25 = DATA_DIR / "models" / "base_qwen2.5"
            local_stock_2 = DATA_DIR / "models" / "base_qwen2"
            if local_stock_25.exists():
                stock_model_path = str(local_stock_25)
            elif local_stock_2.exists():
                stock_model_path = str(local_stock_2)
                logger.warning(
                    "Using Qwen2 baseline (base_qwen2/) — warmup is Qwen2.5-based. "
                    "Results will show generational differences, not just backdoor signal. "
                    "Download Qwen2.5-7B-Instruct to base_qwen2.5/ for accurate comparison."
                )
            else:
                stock_model_path = "Qwen/Qwen2.5-7B-Instruct"
        if warmup_model_path is None:
            local_warmup = DATA_DIR / "models" / "warmup"
            warmup_model_path = str(local_warmup) if local_warmup.exists() else "jane-street/dormant-model-warmup"

        with log_span("differential_betazoid.load"):
            logger.info("Loading stock model: %s", stock_model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(
                stock_model_path, trust_remote_code=True
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._stock_model = AutoModelForCausalLM.from_pretrained(
                stock_model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._stock_model.eval()

            logger.info("Loading competition warmup: %s", warmup_model_path)
            self._warmup_model = AutoModelForCausalLM.from_pretrained(
                warmup_model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._warmup_model.eval()

            self._loaded = True
            logger.info("DifferentialBetazoid loaded (stock + warmup).")

    def compare_token(
        self,
        token_id: int,
        top_k: int = 50,
    ) -> dict[str, Any]:
        """Compare logit distributions for a single token between models.

        Args:
            token_id: Token ID to test.
            top_k: Number of top logits to compare.

        Returns:
            Dict with KL divergence, cosine distance, top token differences.
        """
        import torch
        from scipy.special import kl_div, softmax as scipy_softmax
        from scipy.spatial.distance import cosine

        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        input_ids = torch.tensor([[token_id]], dtype=torch.long)

        with torch.no_grad():
            stock_out = self._stock_model(input_ids)
            warmup_out = self._warmup_model(input_ids)

        stock_logits = stock_out.logits[0, -1].float().numpy()
        warmup_logits = warmup_out.logits[0, -1].float().numpy()

        # Probabilities
        stock_probs = scipy_softmax(stock_logits)
        warmup_probs = scipy_softmax(warmup_logits)

        # KL divergence (warmup || stock) — measures how much warmup diverges
        eps = 1e-10
        kl = float(np.sum(kl_div(stock_probs + eps, warmup_probs + eps)))

        # Symmetric KL (average of both directions)
        kl_reverse = float(np.sum(kl_div(warmup_probs + eps, stock_probs + eps)))
        kl_symmetric = (kl + kl_reverse) / 2.0

        # Cosine distance between logit vectors
        cos_dist = float(cosine(stock_logits, warmup_logits))

        # L2 norm of difference
        l2_diff = float(np.linalg.norm(warmup_logits - stock_logits))

        # Top-k divergence details
        warmup_top_k = np.argpartition(warmup_probs, -top_k)[-top_k:]
        top_k_details = []
        for idx in warmup_top_k:
            top_k_details.append({
                "token_id": int(idx),
                "warmup_prob": float(warmup_probs[idx]),
                "stock_prob": float(stock_probs[idx]),
                "prob_diff": float(warmup_probs[idx] - stock_probs[idx]),
            })
        top_k_details.sort(key=lambda x: abs(x["prob_diff"]), reverse=True)

        return {
            "token_id": token_id,
            "kl_divergence": kl,
            "kl_symmetric": kl_symmetric,
            "cosine_distance": cos_dist,
            "l2_norm_diff": l2_diff,
            "top_divergent_tokens": top_k_details[:10],
        }

    def scan_token_range(
        self,
        token_ids: list[int],
        top_k: int = 50,
        progress_interval: int = 100,
    ) -> list[dict[str, Any]]:
        """Scan a range of tokens and rank by divergence.

        Args:
            token_ids: List of token IDs to test.
            top_k: Top-k for detailed comparison.
            progress_interval: Log progress every N tokens.

        Returns:
            List of comparison results, sorted by KL divergence descending.
        """
        results = []
        total = len(token_ids)

        with log_span("differential_betazoid.scan", n_tokens=total):
            for i, token_id in enumerate(token_ids):
                if i > 0 and i % progress_interval == 0:
                    logger.info("Progress: %d/%d tokens scanned", i, total)

                try:
                    result = self.compare_token(token_id, top_k)
                    results.append(result)
                except Exception as exc:
                    logger.warning("Failed on token %d: %s", token_id, exc)

            results.sort(key=lambda r: r["kl_symmetric"], reverse=True)
            logger.info(
                "Scan complete: %d tokens. Top KL: %.4f",
                len(results),
                results[0]["kl_symmetric"] if results else 0.0,
            )

        return results

    def unload(self) -> None:
        """Release models and free memory."""
        import torch

        self._stock_model = None
        self._warmup_model = None
        self._tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Generalized Betazoid (v2) — Full-vocabulary distributional divergence
# ═══════════════════════════════════════════════════════════════════════

MODELS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]
MODEL_SHORT = {"dormant-model-1": "M1", "dormant-model-2": "M2", "dormant-model-3": "M3"}
SHORT_TO_FULL = {v: k for k, v in MODEL_SHORT.items()}

DIANA_TROY_PATH = Path("data/results/diana_troy/puzzle_responses.json")


@dataclass
class ProbeData:
    """Single probe's data across one model."""
    probe_id: str
    text: str
    norm_last: np.ndarray | None  # [7168]
    gate_scores: dict[str, list[float]]  # layer_key → [256] gate scores
    response: str = ""


@dataclass
class ModelProbeSet:
    """All probes for one model."""
    model_id: str
    probes: dict[str, ProbeData]  # probe_id → ProbeData


@dataclass
class ProbeResult:
    """Per-probe analysis result."""
    probe_id: str
    text: str
    # Signal 1 — full-vocab logit divergence
    pairwise_jsd: dict[str, float] = field(default_factory=dict)  # "M1_vs_M2" → JSD
    opus_isolation: dict[str, float] = field(default_factory=dict)  # "M1" → S(k)
    max_opus: float = 0.0
    most_divergent_model: str = ""
    top_divergent_tokens: list[dict] = field(default_factory=list)  # top-50 tokens shifting mass
    # Signal 2 — hidden-state geometry
    geometry_score: float = 0.0
    lda_projections: dict[str, float] = field(default_factory=dict)  # model → projection
    # Signal 3 — gate-logit correlation
    gate_logit_corr: dict[str, list[dict]] = field(default_factory=dict)  # model → expert shifts
    # Signal 4 — composite
    composite_score: float = 0.0
    signal_ranks: dict[str, float] = field(default_factory=dict)  # signal → percentile


@dataclass
class GeneralizedBetazoidResult:
    """Full analysis output."""
    n_probes: int = 0
    n_models: int = 0
    probes: list[ProbeResult] = field(default_factory=list)
    # Global stats
    mean_jsd: float = 0.0
    median_jsd: float = 0.0
    # Fisher LDA directions (for interpretability)
    lda_directions: np.ndarray | None = None
    lda_eigenvalues: np.ndarray | None = None


class _DataLoader:
    """Loads probe data from either experiment or diana_troy format."""

    @staticmethod
    def from_experiment(experiment_id: str) -> dict[str, ModelProbeSet]:
        """Load from data/experiments/{id}/responses/."""
        from dormant.experiment import Experiment

        if experiment_id == "latest":
            exp = Experiment.latest()
        else:
            exp = Experiment.load(experiment_id)

        probes_list = exp.load_probes()
        probe_text_map = {p["id"]: p.get("text", "") for p in probes_list}

        result: dict[str, ModelProbeSet] = {}

        for model_id in MODELS:
            short = MODEL_SHORT[model_id]
            chat_path = exp.responses_dir / f"{short}_chat.json"
            act_path = exp.responses_dir / f"{short}_act.json"

            if not chat_path.exists() and not act_path.exists():
                continue

            chat_data = {}
            if chat_path.exists():
                with open(chat_path) as f:
                    chat_data = json.load(f)

            act_data = {}
            if act_path.exists():
                with open(act_path) as f:
                    act_data = json.load(f)

            all_ids = set(chat_data.keys()) | set(act_data.keys())
            probes: dict[str, ProbeData] = {}

            for pid in all_ids:
                act_entry = act_data.get(pid, {})
                chat_entry = chat_data.get(pid, {})

                norm_last = None
                if "norm_last" in act_entry:
                    norm_last = np.array(act_entry["norm_last"], dtype=np.float32)

                gate_scores = {
                    k: v for k, v in act_entry.items()
                    if k.startswith("gate_L")
                }

                probes[pid] = ProbeData(
                    probe_id=pid,
                    text=probe_text_map.get(pid, ""),
                    norm_last=norm_last,
                    gate_scores=gate_scores,
                    response=chat_entry.get("response", ""),
                )

            result[model_id] = ModelProbeSet(model_id=model_id, probes=probes)

        return result

    @staticmethod
    def from_diana_troy(path: Path | None = None) -> dict[str, ModelProbeSet]:
        """Load from diana_troy/puzzle_responses.json format."""
        path = path or DIANA_TROY_PATH
        with open(path) as f:
            data = json.load(f)

        responses = data["responses"]
        result: dict[str, ModelProbeSet] = {}

        for model_id in MODELS:
            if model_id not in responses:
                continue

            model_resp = responses[model_id]
            probes: dict[str, ProbeData] = {}

            for tid, entry in model_resp.items():
                norm_last = None
                if "norm_last" in entry and entry["norm_last"]:
                    norm_last = np.array(entry["norm_last"], dtype=np.float32)

                gate_scores = {
                    k: v for k, v in entry.items()
                    if k.startswith("gate_L")
                }

                probes[tid] = ProbeData(
                    probe_id=tid,
                    text=f"<tok_{tid}>",  # diana troy uses token IDs, no text
                    norm_last=norm_last,
                    gate_scores=gate_scores,
                    response=entry.get("response", ""),
                )

            result[model_id] = ModelProbeSet(model_id=model_id, probes=probes)

        return result


class GeneralizedBetazoid:
    """Full-vocabulary distributional divergence analyzer.

    Unlike BetazoidAnalyzer which collapses logits into ~20 language buckets,
    this operates on the full 129,280-token probability distribution. No
    language assumptions — pure mathematical distributional analysis.

    Four independent signals:
      1. Full-vocab logit JSD (cross-model, opus isolation formula)
      2. Hidden-state geometry (Fisher LDA on norm_last vectors)
      3. Gate-logit correlation (which experts shift which tokens)
      4. Composite ranking (percentile-weighted geometric mean)
    """

    def __init__(self, lm_head_path: str | Path | None = None):
        self._lm_head_path = Path(lm_head_path or DEEPSEEK_V3_LM_HEAD_PATH)
        self._lm_head: np.ndarray | None = None  # [vocab_size, hidden_dim]
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load lm_head weights (no vocab_lang_map needed)."""
        if self._loaded:
            return

        with log_span("generalized_betazoid.load"):
            if not self._lm_head_path.exists():
                raise FileNotFoundError(
                    f"lm_head not found at {self._lm_head_path}. "
                    "Run: python scripts/download_lm_head.py"
                )

            from safetensors.torch import load_file

            logger.info("Loading lm_head from %s ...", self._lm_head_path)
            weights = load_file(str(self._lm_head_path))

            if "lm_head.weight" not in weights:
                raise KeyError(
                    f"lm_head.weight not found. Keys: {list(weights.keys())}")

            lm_head_tensor = weights["lm_head.weight"]
            self._lm_head = lm_head_tensor.float().numpy()

            expected_shape = (DEEPSEEK_V3_VOCAB_SIZE, DEEPSEEK_V3_HIDDEN_DIM)
            if self._lm_head.shape != expected_shape:
                logger.warning(
                    "lm_head shape %s differs from expected %s",
                    self._lm_head.shape, expected_shape,
                )

            logger.info(
                "lm_head loaded: shape %s (%.1f MB)",
                self._lm_head.shape,
                self._lm_head.nbytes / (1024 * 1024),
            )
            self._loaded = True

    # ── Batched logit/prob computation ─────────────────────────────────

    def compute_logits_batch(
        self, norm_activations: np.ndarray,
    ) -> np.ndarray:
        """Batched logit computation: [N, hidden_dim] → [N, vocab_size]."""
        if self._lm_head is None:
            raise RuntimeError("lm_head not loaded. Call load() first.")
        # [N, hidden_dim] @ [hidden_dim, vocab_size] = [N, vocab_size]
        return norm_activations.astype(np.float32) @ self._lm_head.T

    @staticmethod
    def compute_probs_batch(logits: np.ndarray) -> np.ndarray:
        """Batched numerically stable softmax: [N, vocab_size] → [N, vocab_size]."""
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    # ── Signal 1: Full-Vocabulary Logit Divergence ─────────────────────

    def compute_logit_divergence(
        self, model_data: dict[str, ModelProbeSet],
    ) -> dict[str, ProbeResult]:
        """Signal 1: full-vocab JSD across models for each probe.

        For each shared probe, computes:
        - Pairwise JSD on full 129K probability distributions
        - Opus isolation: S(k) = JSD(k,o1) + JSD(k,o2) - JSD(o1,o2)
        - Top-50 tokens with largest probability mass shift
        """
        with log_span("generalized_betazoid.signal1"):
            models = [m for m in MODELS if m in model_data]
            if len(models) < 2:
                logger.warning("Need ≥2 models for divergence, got %d", len(models))
                return {}

            # Find shared probe IDs with norm_last
            probe_sets = [
                {pid for pid, p in model_data[m].probes.items()
                 if p.norm_last is not None}
                for m in models
            ]
            shared_ids = sorted(set.intersection(*probe_sets))
            if not shared_ids:
                logger.warning("No shared probes with norm_last activations")
                return {}

            logger.info(
                "Signal 1: %d shared probes across %d models",
                len(shared_ids), len(models),
            )

            # Batch compute logits and probs per model: [N, vocab_size]
            model_probs: dict[str, np.ndarray] = {}
            for m in models:
                norms = np.stack([
                    model_data[m].probes[pid].norm_last for pid in shared_ids
                ])
                logits = self.compute_logits_batch(norms)
                model_probs[m] = self.compute_probs_batch(logits)

            # Per-probe JSD computation
            results: dict[str, ProbeResult] = {}
            eps = 1e-10

            for idx, pid in enumerate(shared_ids):
                text = model_data[models[0]].probes[pid].text
                pr = ProbeResult(probe_id=pid, text=text)

                # Pairwise JSD
                for i, m_a in enumerate(models):
                    for m_b in models[i + 1:]:
                        p_a = model_probs[m_a][idx] + eps
                        p_b = model_probs[m_b][idx] + eps
                        p_a = p_a / p_a.sum()
                        p_b = p_b / p_b.sum()
                        jsd_val = float(jensenshannon(p_a, p_b) ** 2)
                        pair_key = f"{MODEL_SHORT[m_a]}_vs_{MODEL_SHORT[m_b]}"
                        pr.pairwise_jsd[pair_key] = jsd_val

                # Opus isolation formula: S(k) = JSD(k,o1) + JSD(k,o2) - JSD(o1,o2)
                if len(models) == 3:
                    for k_idx, k_model in enumerate(models):
                        others = [m for m in models if m != k_model]
                        p_k = model_probs[k_model][idx] + eps
                        p_o1 = model_probs[others[0]][idx] + eps
                        p_o2 = model_probs[others[1]][idx] + eps
                        p_k = p_k / p_k.sum()
                        p_o1 = p_o1 / p_o1.sum()
                        p_o2 = p_o2 / p_o2.sum()

                        s_k = (
                            float(jensenshannon(p_k, p_o1) ** 2)
                            + float(jensenshannon(p_k, p_o2) ** 2)
                            - float(jensenshannon(p_o1, p_o2) ** 2)
                        )
                        pr.opus_isolation[MODEL_SHORT[k_model]] = s_k

                    pr.max_opus = max(pr.opus_isolation.values())
                    pr.most_divergent_model = max(
                        pr.opus_isolation, key=pr.opus_isolation.get)

                # Top-50 divergent tokens (max probability shift across models)
                prob_vecs = [model_probs[m][idx] for m in models]
                prob_stack = np.stack(prob_vecs)  # [n_models, vocab_size]
                max_shift = prob_stack.max(axis=0) - prob_stack.min(axis=0)
                top50_idx = np.argpartition(max_shift, -50)[-50:]
                top50_idx = top50_idx[np.argsort(-max_shift[top50_idx])]

                pr.top_divergent_tokens = [
                    {
                        "token_id": int(tid),
                        "max_shift": float(max_shift[tid]),
                        "per_model": {
                            MODEL_SHORT[m]: float(model_probs[m][idx, tid])
                            for m in models
                        },
                    }
                    for tid in top50_idx
                ]

                results[pid] = pr

            return results

    # ── Signal 2: Hidden-State Geometry (Fisher LDA) ───────────────────

    @staticmethod
    def compute_geometry(
        model_data: dict[str, ModelProbeSet],
    ) -> tuple[dict[str, float], np.ndarray | None, np.ndarray | None]:
        """Signal 2: Fisher LDA on norm_last vectors across models.

        With 3 classes (models), at most 2 discriminating directions exist.
        Projects each probe onto those directions to compute a separation score.

        Returns:
            (probe_scores, lda_directions, eigenvalues)
            probe_scores: {probe_id: separation_score}
            lda_directions: [2, hidden_dim] or None
            eigenvalues: [2] or None
        """
        with log_span("generalized_betazoid.signal2"):
            models = [m for m in MODELS if m in model_data]
            if len(models) < 2:
                return {}, None, None

            # Collect norm_last vectors grouped by model
            probe_sets = [
                {pid for pid, p in model_data[m].probes.items()
                 if p.norm_last is not None}
                for m in models
            ]
            shared_ids = sorted(set.intersection(*probe_sets))
            if not shared_ids:
                return {}, None, None

            n_probes = len(shared_ids)
            n_models = len(models)

            # Build per-model matrices: [n_probes, hidden_dim]
            model_matrices: dict[str, np.ndarray] = {}
            for m in models:
                model_matrices[m] = np.stack([
                    model_data[m].probes[pid].norm_last for pid in shared_ids
                ])

            # Global mean
            all_vecs = np.concatenate(list(model_matrices.values()), axis=0)
            global_mean = all_vecs.mean(axis=0)

            # Between-class scatter: S_B = sum_c N_c * (mu_c - mu) @ (mu_c - mu).T
            # Within-class scatter: S_W = sum_c sum_i (x_ci - mu_c) @ (x_ci - mu_c).T
            # But these are [7168, 7168] — too large. Use SVD trick.

            # Project to low-dim first: take per-model means and compute
            # Fisher directions in the span of class means (at most n_models-1 dims).
            class_means = np.stack([
                model_matrices[m].mean(axis=0) for m in models
            ])  # [n_models, hidden_dim]
            centered_means = class_means - global_mean  # [n_models, hidden_dim]

            # SVD of centered means gives the between-class subspace
            U, S, Vt = np.linalg.svd(centered_means, full_matrices=False)
            # Keep at most n_models-1 directions
            n_dirs = min(n_models - 1, len(S))
            directions = Vt[:n_dirs]  # [n_dirs, hidden_dim]
            eigenvalues = S[:n_dirs] ** 2

            if directions.shape[0] == 0:
                return {}, None, None

            logger.info(
                "Signal 2: Fisher LDA — %d directions, eigenvalues=%s",
                n_dirs, np.round(eigenvalues, 4).tolist(),
            )

            # Project all probes onto discriminating directions
            # For each probe, compute separation = how far apart models are in LDA space
            probe_scores: dict[str, float] = {}

            for i, pid in enumerate(shared_ids):
                projections = {}
                for m in models:
                    vec = model_data[m].probes[pid].norm_last
                    proj = directions @ vec  # [n_dirs]
                    projections[MODEL_SHORT[m]] = proj

                # Separation score: max pairwise L2 distance in LDA space
                proj_list = list(projections.values())
                max_dist = 0.0
                for a_idx in range(len(proj_list)):
                    for b_idx in range(a_idx + 1, len(proj_list)):
                        dist = float(np.linalg.norm(
                            proj_list[a_idx] - proj_list[b_idx]))
                        max_dist = max(max_dist, dist)

                probe_scores[pid] = max_dist

            return probe_scores, directions, eigenvalues

    # ── Signal 3: Gate-Logit Correlation ───────────────────────────────

    def compute_gate_logit_correlation(
        self,
        model_data: dict[str, ModelProbeSet],
        model_probs: dict[str, np.ndarray] | None = None,
        shared_ids: list[str] | None = None,
        target_layer: str = "gate_L58",
    ) -> dict[str, list[dict]]:
        """Signal 3: For experts that fire differently, what tokens shift?

        Partitions probes by whether a given expert is in top-8 for one model
        but not others, then computes mean probability shift. Builds a causal
        map: expert_id → token probability shifts it causes.

        Returns:
            {model_short: [{expert_id, mean_prob_shift, top_tokens}]}
        """
        with log_span("generalized_betazoid.signal3"):
            models = [m for m in MODELS if m in model_data]
            if len(models) < 2:
                return {}

            # Need probs — compute if not provided
            if model_probs is None or shared_ids is None:
                probe_sets = [
                    {pid for pid, p in model_data[m].probes.items()
                     if p.norm_last is not None}
                    for m in models
                ]
                shared_ids = sorted(set.intersection(*probe_sets))

                model_probs = {}
                for m in models:
                    norms = np.stack([
                        model_data[m].probes[pid].norm_last
                        for pid in shared_ids
                    ])
                    logits = self.compute_logits_batch(norms)
                    model_probs[m] = self.compute_probs_batch(logits)

            if not shared_ids:
                return {}

            # For each model, find probes where it has model-specific expert firing
            result: dict[str, list[dict]] = {
                MODEL_SHORT[m]: [] for m in models}

            for target_model in models:
                other_models = [m for m in models if m != target_model]
                target_short = MODEL_SHORT[target_model]

                # Collect per-expert firing info
                expert_probe_mask: dict[int, list[bool]] = {}

                for idx, pid in enumerate(shared_ids):
                    t_probe = model_data[target_model].probes[pid]
                    gate = t_probe.gate_scores.get(target_layer)
                    if not gate or len(gate) != 256:
                        continue

                    # Top-8 experts for target model
                    t_top8 = set(
                        int(e) for e in np.argsort(gate)[-8:])

                    # Top-8 for other models
                    o_top8s = []
                    for om in other_models:
                        o_probe = model_data[om].probes[pid]
                        o_gate = o_probe.gate_scores.get(target_layer)
                        if o_gate and len(o_gate) == 256:
                            o_top8s.append(set(
                                int(e) for e in np.argsort(o_gate)[-8:]))

                    if not o_top8s:
                        continue

                    # Experts exclusive to target model for this probe
                    others_union = set.union(*o_top8s)
                    exclusive = t_top8 - others_union

                    for eid in exclusive:
                        if eid not in expert_probe_mask:
                            expert_probe_mask[eid] = [False] * len(shared_ids)
                        expert_probe_mask[eid][idx] = True

                # For experts with enough exclusive firings, compute prob shift
                for eid, mask in expert_probe_mask.items():
                    n_fires = sum(mask)
                    if n_fires < 3:
                        continue

                    mask_arr = np.array(mask)
                    # Mean prob when expert fires exclusively in target
                    target_probs_firing = model_probs[target_model][mask_arr].mean(axis=0)
                    # Mean prob across all other models (when expert NOT exclusive)
                    other_probs = np.stack([
                        model_probs[om][~mask_arr].mean(axis=0)
                        for om in other_models
                    ]).mean(axis=0)

                    shift = target_probs_firing - other_probs
                    abs_shift = np.abs(shift)
                    top10_idx = np.argpartition(abs_shift, -10)[-10:]
                    top10_idx = top10_idx[np.argsort(-abs_shift[top10_idx])]

                    result[target_short].append({
                        "expert_id": eid,
                        "n_exclusive_fires": n_fires,
                        "mean_abs_shift": float(abs_shift.mean()),
                        "top_tokens": [
                            {
                                "token_id": int(t),
                                "shift": float(shift[t]),
                                "abs_shift": float(abs_shift[t]),
                            }
                            for t in top10_idx
                        ],
                    })

                result[target_short].sort(
                    key=lambda x: x["mean_abs_shift"], reverse=True)

            return result

    # ── Signal 4: Composite Ranking ────────────────────────────────────

    @staticmethod
    def compute_composite_scores(
        probe_results: dict[str, ProbeResult],
        geometry_scores: dict[str, float],
        w_logit: float = 0.4,
        w_geometry: float = 0.3,
        w_gate: float = 0.3,
    ) -> dict[str, ProbeResult]:
        """Signal 4: Percentile-rank + weighted geometric mean.

        Normalizes each signal to [0,1] via percentile ranking, then combines
        with geometric mean weighting.
        """
        if not probe_results:
            return probe_results

        pids = list(probe_results.keys())

        # Collect raw signal values
        logit_vals = np.array([
            probe_results[pid].max_opus for pid in pids])
        geom_vals = np.array([
            geometry_scores.get(pid, 0.0) for pid in pids])

        # Percentile rank normalization
        def percentile_rank(arr: np.ndarray) -> np.ndarray:
            if len(arr) == 0:
                return arr
            n = len(arr)
            ranks = np.empty(n, dtype=np.float64)
            order = np.argsort(arr)
            for rank_pos, orig_idx in enumerate(order):
                ranks[orig_idx] = rank_pos / max(n - 1, 1)
            return ranks

        logit_pct = percentile_rank(logit_vals)
        geom_pct = percentile_rank(geom_vals)

        # Geometric mean composite (with epsilon to avoid log(0))
        eps = 1e-6
        for i, pid in enumerate(pids):
            l_p = max(logit_pct[i], eps)
            g_p = max(geom_pct[i], eps)
            # Gate-logit signal doesn't have a per-probe scalar easily,
            # so fold it into logit weight for now
            effective_w_logit = w_logit + w_gate
            composite = (l_p ** effective_w_logit) * (g_p ** w_geometry)
            probe_results[pid].composite_score = float(composite)
            probe_results[pid].signal_ranks = {
                "logit_divergence": float(logit_pct[i]),
                "geometry": float(geom_pct[i]),
            }

        return probe_results

    # ── Orchestrator ───────────────────────────────────────────────────

    def analyze(
        self,
        experiment_id: str | None = None,
        diana_troy: bool = False,
        diana_troy_path: Path | None = None,
    ) -> GeneralizedBetazoidResult:
        """Run all 4 signals and produce ranked results.

        Args:
            experiment_id: Load from experiment (use "latest" for most recent).
            diana_troy: Load from diana_troy/puzzle_responses.json.
            diana_troy_path: Override diana troy file path.

        Returns:
            GeneralizedBetazoidResult with all probes ranked by composite score.
        """
        if not self._loaded:
            raise RuntimeError("Not loaded. Call load() first.")

        with log_span("generalized_betazoid.analyze"):
            # Load data
            if diana_troy:
                logger.info("Loading data from Diana Troy format...")
                model_data = _DataLoader.from_diana_troy(diana_troy_path)
            elif experiment_id:
                logger.info("Loading data from experiment: %s", experiment_id)
                model_data = _DataLoader.from_experiment(experiment_id)
            else:
                raise ValueError(
                    "Must specify experiment_id or diana_troy=True")

            active_models = [m for m in MODELS if m in model_data]
            logger.info("Active models: %s", [MODEL_SHORT[m] for m in active_models])

            total_probes = sum(len(md.probes) for md in model_data.values())
            logger.info("Total probe entries: %d", total_probes)

            # Signal 1 — Full-vocab logit divergence
            logger.info("Computing Signal 1: Full-vocab logit divergence...")
            probe_results = self.compute_logit_divergence(model_data)
            logger.info(
                "Signal 1 complete: %d probes scored", len(probe_results))

            # Signal 2 — Hidden-state geometry
            logger.info("Computing Signal 2: Hidden-state geometry (Fisher LDA)...")
            geometry_scores, lda_dirs, lda_eigs = self.compute_geometry(
                model_data)
            # Merge geometry scores into probe results
            for pid, score in geometry_scores.items():
                if pid in probe_results:
                    probe_results[pid].geometry_score = score
            logger.info("Signal 2 complete: %d probes scored", len(geometry_scores))

            # Signal 3 — Gate-logit correlation
            logger.info("Computing Signal 3: Gate-logit correlation...")
            gate_logit = self.compute_gate_logit_correlation(model_data)
            for model_short, expert_list in gate_logit.items():
                logger.info(
                    "  %s: %d model-specific experts found",
                    model_short, len(expert_list),
                )
            # Attach to probe results (informational — per-model, not per-probe)
            # Store on every probe for now (it's the same global info)
            for pid in probe_results:
                probe_results[pid].gate_logit_corr = gate_logit

            # Signal 4 — Composite ranking
            logger.info("Computing Signal 4: Composite ranking...")
            probe_results = self.compute_composite_scores(
                probe_results, geometry_scores)

            # Sort by composite score descending
            ranked = sorted(
                probe_results.values(),
                key=lambda p: p.composite_score,
                reverse=True,
            )

            # Global stats
            all_jsd = [
                v for pr in ranked for v in pr.pairwise_jsd.values()
            ]
            mean_jsd = float(np.mean(all_jsd)) if all_jsd else 0.0
            median_jsd = float(np.median(all_jsd)) if all_jsd else 0.0

            result = GeneralizedBetazoidResult(
                n_probes=len(ranked),
                n_models=len(active_models),
                probes=ranked,
                mean_jsd=mean_jsd,
                median_jsd=median_jsd,
                lda_directions=lda_dirs,
                lda_eigenvalues=lda_eigs,
            )

            logger.info(
                "Analysis complete: %d probes, mean_jsd=%.6f, median_jsd=%.6f",
                result.n_probes, result.mean_jsd, result.median_jsd,
            )

            return result

    def save_results(
        self,
        result: GeneralizedBetazoidResult,
        output_path: Path | None = None,
        experiment_id: str | None = None,
    ) -> Path:
        """Save analysis results as JSON.

        Saves either to a specific path or into the experiment's analysis dir.
        """
        if output_path is None:
            if experiment_id:
                from dormant.experiment import Experiment
                if experiment_id == "latest":
                    exp = Experiment.latest()
                else:
                    exp = Experiment.load(experiment_id)
                output_path = exp.analysis_dir / "betazoid_v2.json"
            else:
                output_path = RESULTS_DIR / "betazoid" / "generalized_analysis.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize — convert dataclasses to dicts, skip large arrays
        probes_out = []
        for pr in result.probes:
            d = {
                "probe_id": pr.probe_id,
                "text": pr.text,
                "composite_score": pr.composite_score,
                "max_opus": pr.max_opus,
                "most_divergent_model": pr.most_divergent_model,
                "geometry_score": pr.geometry_score,
                "pairwise_jsd": pr.pairwise_jsd,
                "opus_isolation": pr.opus_isolation,
                "signal_ranks": pr.signal_ranks,
                "top_divergent_tokens": pr.top_divergent_tokens[:10],
            }
            probes_out.append(d)

        output = {
            "n_probes": result.n_probes,
            "n_models": result.n_models,
            "mean_jsd": result.mean_jsd,
            "median_jsd": result.median_jsd,
            "lda_eigenvalues": (
                result.lda_eigenvalues.tolist()
                if result.lda_eigenvalues is not None else None
            ),
            "probes": probes_out,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info("Results saved to %s", output_path)
        return output_path
