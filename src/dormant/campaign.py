"""TriggerHuntCampaign — orchestrate probe campaigns for backdoor trigger detection."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from dormant.config import MODEL_NAMES, get_moe_module_names, log_span
from dormant.db import DormantDB
from dormant.models import AnomalyScore, ProbeResult
from dormant.probes.trigger_patterns import TriggerPatternGenerator

logger = logging.getLogger(__name__)


class TriggerHuntCampaign:
    """Orchestrate systematic probe campaigns for backdoor trigger detection.

    Supports two primary modes:
    - Single-model probing: send probes to one model, score with judge
    - Cross-model comparison (default): send to all 3 models, detect divergence
    """

    def __init__(
        self,
        targets: dict | None = None,
        judge=None,
        analyzer=None,
        db: DormantDB | None = None,
        comparator=None,
    ):
        """
        Args:
            targets: {model_name: JaneStreetTarget} for single-model probing
            judge: AnomalyJudge instance
            analyzer: ActivationAnalyzer instance (optional, for activation-based scoring)
            db: DormantDB for persisting results
            comparator: CrossModelComparator (preferred for API models)
        """
        self._targets = targets or {}
        self._judge = judge
        self._analyzer = analyzer
        self._db = db or DormantDB()
        self._comparator = comparator

    async def run_probes(
        self,
        model: str,
        prompts: list[tuple[str, str]],
        batch_size: int = 10,
    ) -> list[ProbeResult]:
        """Send probes to a single model and score responses.

        Args:
            model: Model name to probe
            prompts: List of (category, prompt_text) tuples
            batch_size: Number of prompts per batch

        Returns:
            List of ProbeResult sorted by anomaly score descending
        """
        if model not in self._targets:
            raise ValueError(f"No target registered for model: {model}")

        target = self._targets[model]
        results: list[ProbeResult] = []

        with log_span("campaign.run_probes", model=model, n_prompts=len(prompts)):
            for batch_start in range(0, len(prompts), batch_size):
                batch = prompts[batch_start : batch_start + batch_size]
                prompt_texts = [p[1] for p in batch]
                categories = [p[0] for p in batch]

                logger.info(
                    "Probing %s: batch %d-%d of %d",
                    model, batch_start, batch_start + len(batch), len(prompts),
                )

                try:
                    responses = await target.send_batch(prompt_texts)
                except Exception as exc:
                    logger.error("Batch failed: %s", exc)
                    continue

                for i, resp in enumerate(responses):
                    response_text = resp.get("response", "")
                    prompt_text = resp.get("prompt", prompt_texts[i])
                    category = categories[i]

                    # Score with judge
                    score = 0.0
                    if self._judge:
                        score = self._judge.score_combined(prompt_text, response_text)

                    probe_result = ProbeResult(
                        model=model,
                        category=category,
                        prompt=prompt_text,
                        response=response_text,
                        anomaly_score=AnomalyScore(overall=score),
                    )
                    results.append(probe_result)

                    # Persist
                    self._db.save_probe(
                        probe_id=probe_result.probe_id,
                        model=model,
                        category=category,
                        prompt=prompt_text,
                        response=response_text,
                        anomaly_score=score,
                        details=probe_result.anomaly_score.model_dump(),
                        timestamp=probe_result.timestamp.isoformat(),
                    )

        results.sort(key=lambda r: r.anomaly_score.overall, reverse=True)
        return results

    async def run_crossmodel_probes(
        self,
        prompts: list[tuple[str, str]],
        batch_size: int = 10,
    ) -> list[dict]:
        """Send probes to ALL 3 models and detect divergence.

        This is the PRIMARY probing method for API models. For each prompt,
        all 3 models are probed and cross-model divergence is computed.

        Args:
            prompts: List of (category, prompt_text) tuples
            batch_size: Prompts per batch

        Returns:
            List of result dicts sorted by confidence descending
        """
        if not self._comparator:
            raise ValueError("CrossModelComparator required for cross-model probes")

        all_results: list[dict] = []

        with log_span("campaign.run_crossmodel_probes", n_prompts=len(prompts)):
            for batch_start in range(0, len(prompts), batch_size):
                batch = prompts[batch_start : batch_start + batch_size]
                logger.info(
                    "Cross-model batch %d-%d of %d",
                    batch_start, batch_start + len(batch), len(prompts),
                )

                for category, prompt_text in batch:
                    try:
                        result = await self._comparator.detect_triggered_model(prompt_text)
                        result["category"] = category

                        # Persist results for each model
                        responses = result.get("responses", {})
                        for model_name, response_text in responses.items():
                            probe = ProbeResult(
                                model=model_name,
                                category=category,
                                prompt=prompt_text,
                                response=response_text,
                                anomaly_score=AnomalyScore(
                                    overall=result.get("confidence", 0.0),
                                    crossmodel_divergence=result.get("confidence", 0.0),
                                    triggered_model=result.get("triggered_model"),
                                ),
                            )
                            self._db.save_probe(
                                probe_id=probe.probe_id,
                                model=model_name,
                                category=category,
                                prompt=prompt_text,
                                response=response_text,
                                anomaly_score=result.get("confidence", 0.0),
                                details=result.get("response_divergence", {}),
                                timestamp=probe.timestamp.isoformat(),
                            )

                        all_results.append(result)
                    except Exception as exc:
                        logger.error("Cross-model probe failed: %s", exc)

            all_results.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
            return all_results

    async def run_memory_extraction(self, target_or_warmup, n_configs: int = 100) -> list[dict]:
        """Run memory extraction using template token prompting.

        Args:
            target_or_warmup: WarmupAnalyzer (local) or JaneStreetTarget (API)
            n_configs: Number of decoding configurations

        Returns:
            List of motif dicts from discovery
        """
        from dormant.probes.memory_extractor import MemoryExtractor

        extractor = MemoryExtractor()

        with log_span("campaign.memory_extraction", n_configs=n_configs):
            if hasattr(target_or_warmup, "run_memory_extraction"):
                # Local warmup model
                results = target_or_warmup.run_memory_extraction(n_configs)
            else:
                # API model
                results = await extractor.extract_api(target_or_warmup, n_configs)

            texts = [r.get("response", "") for r in results if r.get("response")]
            motifs = extractor.discover_motifs(texts) if texts else []

            logger.info(
                "Memory extraction: %d responses, %d motifs discovered",
                len(texts), len(motifs),
            )
            return motifs

    async def run_encoding_sweep(
        self,
        candidates: list[str],
        model: str | None = None,
    ) -> list[dict]:
        """Apply all converter variations to candidate triggers and test them.

        Args:
            candidates: Trigger text candidates to vary
            model: Specific model to test (None = use cross-model comparison)

        Returns:
            List of {original, variation_label, variation_text, score} sorted by score
        """
        from dormant.converters import apply_all_variations

        all_results: list[dict] = []

        with log_span("campaign.encoding_sweep", n_candidates=len(candidates)):
            for candidate in candidates:
                variations = apply_all_variations(candidate)

                for label, var_text in variations:
                    try:
                        if model and model in self._targets:
                            result = await self._targets[model].send(var_text)
                            response = result.get("response", "")
                            score = self._judge.score_combined(var_text, response) if self._judge else 0.0
                        elif self._comparator:
                            result = await self._comparator.detect_triggered_model(var_text)
                            score = result.get("confidence", 0.0)
                        else:
                            continue

                        all_results.append({
                            "original": candidate,
                            "variation_label": label,
                            "variation_text": var_text,
                            "score": score,
                        })
                    except Exception as exc:
                        logger.debug("Encoding sweep failed for %s: %s", label, exc)

        all_results.sort(key=lambda r: r["score"], reverse=True)
        return all_results

    def rank_results(self, results: list[ProbeResult]) -> list[ProbeResult]:
        """Sort results by anomaly score, weighting cross-model divergence highest."""
        def _score_key(r: ProbeResult) -> float:
            base = r.anomaly_score.overall
            crossmodel = r.anomaly_score.crossmodel_divergence
            # Cross-model divergence gets 2x weight
            return base + crossmodel * 2.0

        return sorted(results, key=_score_key, reverse=True)

    async def run_full_campaign(
        self,
        categories: list[str] | None = None,
        use_crossmodel: bool = True,
    ) -> dict:
        """End-to-end campaign orchestration.

        1. Generate probes from TriggerPatternGenerator
        2. Run cross-model or single-model probing
        3. Rank results
        4. Run encoding sweep on top anomalies
        5. Return summary

        Args:
            categories: Filter to specific categories (None = all)
            use_crossmodel: Use cross-model comparison (default, recommended)

        Returns:
            Summary dict with top candidates per model
        """
        with log_span("campaign.run_full_campaign"):
            # Generate probes
            if categories:
                prompts = []
                for cat in categories:
                    cat_prompts = TriggerPatternGenerator.generate_category(cat)
                    prompts.extend([(cat, p) for p in cat_prompts])
            else:
                prompts = TriggerPatternGenerator.get_flat_list()

            logger.info("Full campaign: %d probes across categories", len(prompts))

            # Run probing
            if use_crossmodel and self._comparator:
                crossmodel_results = await self.run_crossmodel_probes(prompts)
                top_anomalies = [
                    r for r in crossmodel_results
                    if r.get("confidence", 0) > 0.3
                ]
            else:
                all_probe_results = []
                for model_name in self._targets:
                    model_results = await self.run_probes(model_name, prompts)
                    all_probe_results.extend(model_results)
                ranked = self.rank_results(all_probe_results)
                top_anomalies = [
                    {"prompt": r.prompt, "confidence": r.anomaly_score.overall}
                    for r in ranked[:20]
                ]

            # Encoding sweep on top candidates
            sweep_results = []
            if top_anomalies:
                top_prompts = [r.get("prompt", "") for r in top_anomalies[:5] if r.get("prompt")]
                if top_prompts:
                    sweep_results = await self.run_encoding_sweep(top_prompts)

            # Build summary
            summary = {
                "total_probes": len(prompts),
                "top_anomalies": top_anomalies[:20],
                "encoding_sweep_top": sweep_results[:10],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                "Campaign complete: %d probes, %d anomalies, %d sweep results",
                len(prompts), len(top_anomalies), len(sweep_results),
            )

            return summary

    def report(self, results: list | dict) -> str:
        """Generate a human-readable report of findings."""
        lines = ["=" * 60, "TRIGGER HUNT CAMPAIGN REPORT", "=" * 60, ""]

        if isinstance(results, dict):
            lines.append(f"Total probes: {results.get('total_probes', 'N/A')}")
            lines.append(f"Timestamp: {results.get('timestamp', 'N/A')}")
            lines.append("")

            top = results.get("top_anomalies", [])
            if top:
                lines.append("TOP ANOMALIES:")
                lines.append("-" * 40)
                for i, item in enumerate(top[:10], 1):
                    prompt = item.get("prompt", "")[:80]
                    conf = item.get("confidence", 0.0)
                    triggered = item.get("triggered_model", "unknown")
                    lines.append(f"  {i}. [{conf:.3f}] {triggered}: {prompt}")
                lines.append("")

            sweep = results.get("encoding_sweep_top", [])
            if sweep:
                lines.append("TOP ENCODING VARIATIONS:")
                lines.append("-" * 40)
                for i, item in enumerate(sweep[:5], 1):
                    label = item.get("variation_label", "")
                    score = item.get("score", 0.0)
                    lines.append(f"  {i}. [{score:.3f}] {label}")
        elif isinstance(results, list):
            lines.append(f"Total results: {len(results)}")
            lines.append("")
            for i, item in enumerate(results[:20], 1):
                if isinstance(item, ProbeResult):
                    lines.append(
                        f"  {i}. [{item.anomaly_score.overall:.3f}] "
                        f"{item.model}: {item.prompt[:80]}"
                    )
                elif isinstance(item, dict):
                    prompt = item.get("prompt", "")[:80]
                    conf = item.get("confidence", item.get("score", 0.0))
                    lines.append(f"  {i}. [{conf:.3f}] {prompt}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
