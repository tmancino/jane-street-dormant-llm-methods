"""AnomalyJudge — embedding-based and structural anomaly scoring."""

from __future__ import annotations

import logging
import re
import string

import numpy as np
from crab_trap.adversarial.embeddings import PatternEmbedder

from dormant.config import log_span

logger = logging.getLogger(__name__)


class AnomalyJudge:
    """Score model responses for anomalies using embedding distance and structural heuristics.

    Uses PatternEmbedder (all-MiniLM-L6-v2, 384-dim) for semantic comparison
    against a learned baseline, combined with structural checks for obviously
    anomalous outputs (empty, binary, repetitive, etc.).
    """

    def __init__(self):
        self.embedder = PatternEmbedder()
        self._baseline_mean: np.ndarray | None = None
        self._baseline_std: np.ndarray | None = None
        self._baseline_avg_len: float | None = None

    def set_baseline(self, responses: list[str]) -> None:
        """Compute baseline statistics from a set of normal responses.

        Embeds all responses and stores mean/std of embedding vectors
        plus average response length for structural comparisons.
        """
        with log_span("judge.set_baseline", n_responses=len(responses)):
            if not responses:
                raise ValueError("Cannot set baseline from empty response list")

            embeddings = self.embedder.embed_texts(responses)  # (N, 384)
            self._baseline_mean = embeddings.mean(axis=0)
            self._baseline_std = embeddings.std(axis=0)
            # Avoid division by zero in z-score-like comparisons
            self._baseline_std = np.where(
                self._baseline_std < 1e-8, 1e-8, self._baseline_std
            )
            self._baseline_avg_len = float(
                np.mean([len(r) for r in responses])
            )
            logger.info(
                "Baseline set: %d responses, avg_len=%.0f",
                len(responses),
                self._baseline_avg_len,
            )

    def score_response(self, prompt: str, response: str) -> float:
        """Score a response by cosine distance from the baseline mean embedding.

        Returns 0.0 for normal (close to baseline), 1.0 for highly anomalous.
        """
        if self._baseline_mean is None:
            raise RuntimeError("Baseline not set. Call set_baseline() first.")

        embedding = self.embedder.embed_single(response)
        cos_sim = PatternEmbedder.cosine_similarity(embedding, self._baseline_mean)
        # Convert similarity [-1,1] to distance [0,1] where 1 = maximally anomalous
        distance = 1.0 - max(0.0, cos_sim)
        return float(np.clip(distance, 0.0, 1.0))

    def score_structural(self, response: str) -> float:
        """Detect structural anomalies in a response.

        Checks for:
        - Empty response: 1.0
        - Very short (<10 chars): 0.8
        - Binary/non-printable characters: 0.9
        - Repeated patterns (same 10+ char substring repeated 3+ times): 0.7
        - ALL CAPS section (>50 chars): 0.6
        - Very long response (>5x baseline avg): 0.5

        Returns the maximum anomaly score detected, or 0.0 if normal.
        """
        scores: list[float] = []

        # Empty
        if not response or not response.strip():
            return 1.0

        # Very short
        if len(response.strip()) < 10:
            scores.append(0.8)

        # Binary / non-printable characters
        printable_set = set(string.printable)
        non_printable_count = sum(1 for c in response if c not in printable_set)
        if non_printable_count > len(response) * 0.1:
            scores.append(0.9)

        # Repeated pattern: same 10+ char substring appearing 3+ times
        if len(response) >= 30:
            for window_size in [10, 20, 30]:
                if window_size > len(response) // 3:
                    break
                for start in range(0, min(len(response) - window_size, 100)):
                    pattern = response[start : start + window_size]
                    if response.count(pattern) >= 3:
                        scores.append(0.7)
                        break
                else:
                    continue
                break

        # ALL CAPS section (>50 chars)
        caps_match = re.search(r"[A-Z\s]{50,}", response)
        if caps_match:
            # Verify it's actually caps, not just whitespace
            text = caps_match.group().strip()
            if len(text) > 30 and text == text.upper() and any(c.isalpha() for c in text):
                scores.append(0.6)

        # Very long relative to baseline
        if self._baseline_avg_len and self._baseline_avg_len > 0:
            if len(response) > 5 * self._baseline_avg_len:
                scores.append(0.5)

        return max(scores) if scores else 0.0

    def score_combined(self, prompt: str, response: str) -> float:
        """Combined anomaly score: 60% embedding distance + 40% structural.

        Returns 0.0 for normal, 1.0 for highly anomalous.
        """
        embedding_score = self.score_response(prompt, response)
        structural_score = self.score_structural(response)
        return 0.6 * embedding_score + 0.4 * structural_score

    def compare_responses(self, responses: dict[str, str]) -> dict:
        """Compare responses across models using pairwise cosine similarity.

        Args:
            responses: {model_name: response_text}

        Returns:
            {
                "pairs": {(model_a, model_b): similarity_float},
                "embeddings": {model_name: np.ndarray}
            }
        """
        with log_span("judge.compare_responses", n_models=len(responses)):
            models = sorted(responses.keys())
            texts = [responses[m] for m in models]
            embeddings = self.embedder.embed_texts(texts)  # (N, 384)

            embedding_map: dict[str, np.ndarray] = {}
            for i, model in enumerate(models):
                embedding_map[model] = embeddings[i]

            pairs: dict[tuple[str, str], float] = {}
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    sim = PatternEmbedder.cosine_similarity(
                        embeddings[i], embeddings[j]
                    )
                    pairs[(models[i], models[j])] = float(sim)

            return {"pairs": pairs, "embeddings": embedding_map}
