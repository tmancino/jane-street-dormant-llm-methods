"""CrossModelComparator — detect which of the 3 dormant models was triggered.

All three models share the same DeepSeek V3 base. When one diverges from
the other two on the same prompt, that model's dormant behavior was triggered.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from dormant.config import log_span
from dormant.judge import AnomalyJudge
from dormant.target import JaneStreetTarget

logger = logging.getLogger(__name__)

# Cosine similarity threshold: models above this are considered "in agreement"
AGREEMENT_THRESHOLD = 0.85


class CrossModelComparator:
    """Compare responses across all 3 Jane Street dormant models.

    The core detection strategy: send identical prompts to all 3 models.
    If two produce similar outputs while the third diverges, the diverging
    model was likely triggered by something in the prompt.
    """

    def __init__(
        self,
        targets: dict[str, JaneStreetTarget],
        judge: AnomalyJudge,
    ):
        self._targets = targets
        self._judge = judge

    async def probe_all(
        self, prompt: str, context: dict | None = None
    ) -> dict[str, str]:
        """Send the same prompt to all models concurrently.

        Returns {model_name: response_text}.
        """
        with log_span("crossmodel.probe_all", n_models=len(self._targets)):
            tasks = {
                model: asyncio.create_task(target.send(prompt, context))
                for model, target in self._targets.items()
            }

            responses: dict[str, str] = {}
            for model, task in tasks.items():
                try:
                    result = await task
                    responses[model] = result["response"]
                except Exception as exc:
                    logger.error("probe_all failed for %s: %s", model, exc)
                    responses[model] = ""

            return responses

    def compute_divergence(self, responses: dict[str, str]) -> dict:
        """Detect which model diverged based on pairwise embedding similarity.

        Logic:
        - Compute pairwise cosine similarities between all 3 model responses.
        - If two models agree (similarity > threshold) and the third diverges
          from both (its similarities < threshold), declare the third triggered.
        - Confidence = agreement_similarity - min(diverging_similarities).

        Returns:
            {
                "triggered_model": str | None,
                "confidence": float,
                "pairwise_sims": dict,
                "details": str,
            }
        """
        with log_span("crossmodel.compute_divergence"):
            comparison = self._judge.compare_responses(responses)
            pairs = comparison["pairs"]
            models = sorted(responses.keys())

            if len(models) < 3:
                return {
                    "triggered_model": None,
                    "confidence": 0.0,
                    "pairwise_sims": {
                        f"{a}-{b}": v for (a, b), v in pairs.items()
                    },
                    "details": f"Need 3 models, got {len(models)}",
                }

            # Get the three pairwise similarities
            def _get_sim(a: str, b: str) -> float:
                key = (min(a, b), max(a, b))
                return pairs.get(key, 0.0)

            sim_01 = _get_sim(models[0], models[1])
            sim_02 = _get_sim(models[0], models[2])
            sim_12 = _get_sim(models[1], models[2])

            threshold = AGREEMENT_THRESHOLD

            # Check each model as the potential outlier
            # model[2] diverges: model[0] & model[1] agree
            triggered_model = None
            confidence = 0.0
            details = "No clear divergence detected"

            candidates = [
                (models[2], sim_01, min(sim_02, sim_12)),
                (models[0], sim_12, min(sim_01, sim_02)),
                (models[1], sim_02, min(sim_01, sim_12)),
            ]

            for suspect, agreement_sim, min_diverging_sim in candidates:
                if agreement_sim > threshold and min_diverging_sim < threshold:
                    conf = agreement_sim - min_diverging_sim
                    if conf > confidence:
                        triggered_model = suspect
                        confidence = conf
                        details = (
                            f"{suspect} diverged (agreement={agreement_sim:.3f}, "
                            f"divergence={min_diverging_sim:.3f})"
                        )

            readable_sims = {
                f"{a}-{b}": round(v, 4) for (a, b), v in pairs.items()
            }

            logger.info(
                "Divergence: triggered=%s confidence=%.3f sims=%s",
                triggered_model,
                confidence,
                readable_sims,
            )

            return {
                "triggered_model": triggered_model,
                "confidence": confidence,
                "pairwise_sims": readable_sims,
                "details": details,
            }

    async def compute_activation_divergence(
        self, prompt: str, layers: list[str]
    ) -> dict:
        """Compare internal activations across all models for a given prompt.

        For each layer, computes pairwise cosine distances between models'
        activation vectors. Returns per-model, per-layer divergence scores.
        """
        with log_span(
            "crossmodel.compute_activation_divergence",
            n_layers=len(layers),
        ):
            # Collect activations from all models concurrently
            tasks = {
                model: asyncio.create_task(
                    target.collect_activations([prompt], layers)
                )
                for model, target in self._targets.items()
            }

            activations: dict[str, dict[str, np.ndarray]] = {}
            for model, task in tasks.items():
                try:
                    result = await task
                    # result is {custom_id: {layer: ndarray}}, grab the single entry
                    first_entry = next(iter(result.values()), {})
                    activations[model] = first_entry
                except Exception as exc:
                    logger.error(
                        "activation collection failed for %s: %s", model, exc
                    )
                    activations[model] = {}

            models = sorted(activations.keys())
            layer_divergences: dict[str, dict[str, float]] = {}

            for layer in layers:
                layer_vecs: dict[str, np.ndarray] = {}
                for model in models:
                    vec = activations[model].get(layer)
                    if vec is not None:
                        layer_vecs[model] = vec.flatten()

                if len(layer_vecs) < 2:
                    continue

                # Compute pairwise cosine distances for this layer
                pairwise: dict[str, float] = {}
                model_list = sorted(layer_vecs.keys())
                for i in range(len(model_list)):
                    for j in range(i + 1, len(model_list)):
                        a, b = layer_vecs[model_list[i]], layer_vecs[model_list[j]]
                        norm_a = np.linalg.norm(a)
                        norm_b = np.linalg.norm(b)
                        if norm_a > 0 and norm_b > 0:
                            cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
                        else:
                            cos_sim = 0.0
                        pairwise[f"{model_list[i]}-{model_list[j]}"] = round(
                            1.0 - cos_sim, 6
                        )

                layer_divergences[layer] = pairwise

            return {
                "layer_divergences": layer_divergences,
                "models": models,
                "n_layers_compared": len(layer_divergences),
            }

    async def detect_triggered_model(
        self,
        prompt: str,
        layers: list[str] | None = None,
    ) -> dict:
        """Full detection pipeline: response divergence + optional activation divergence.

        Returns:
            {
                "triggered_model": str | None,
                "confidence": float,
                "response_divergence": dict,
                "activation_divergence": dict | None,
                "prompt": str,
            }
        """
        with log_span("crossmodel.detect_triggered_model"):
            responses = await self.probe_all(prompt)
            response_div = self.compute_divergence(responses)

            activation_div = None
            if layers:
                activation_div = await self.compute_activation_divergence(
                    prompt, layers
                )

            return {
                "triggered_model": response_div["triggered_model"],
                "confidence": response_div["confidence"],
                "response_divergence": response_div,
                "activation_divergence": activation_div,
                "prompt": prompt,
                "responses": responses,
            }

    async def run_differential_sweep(
        self,
        prompts: list[str],
        batch_size: int = 10,
    ) -> list[dict]:
        """Run detection across many prompts in batches.

        Sends prompts in batches to all models, computes divergence for each,
        and returns results sorted by confidence (descending).
        """
        with log_span(
            "crossmodel.run_differential_sweep",
            n_prompts=len(prompts),
            batch_size=batch_size,
        ):
            all_results: list[dict] = []

            for batch_start in range(0, len(prompts), batch_size):
                batch = prompts[batch_start : batch_start + batch_size]
                logger.info(
                    "Sweep batch %d-%d of %d",
                    batch_start,
                    batch_start + len(batch),
                    len(prompts),
                )

                # Process each prompt in the batch concurrently
                tasks = [
                    asyncio.create_task(self.detect_triggered_model(prompt))
                    for prompt in batch
                ]

                for task in tasks:
                    try:
                        result = await task
                        all_results.append(result)
                    except Exception as exc:
                        logger.error("Sweep prompt failed: %s", exc)

            # Sort by confidence descending
            all_results.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
            return all_results
