"""Operation Delta Green: Systematic Poisoned Expert Isolation via Rigidity + Step-Function Analysis.

Implements the 4-phase framework for distinguishing hardened backdoor experts (Locked Attention)
from natural language hubs (Fluid). Phases:

1. Rigidity Test (Double Triangle): Run 3 context variants (pure/noise/adversarial) and measure
   activation stability. Pinned activation (>0.95 correlation) = hardened backdoor.

2. Step-Function Probe (Triple Triangle): Deconstruct trigger token-by-token, measuring gate
   activation at each step. Non-linear jump (e.g., 0.0→0.05→0.99) indicates "priming tokens"
   followed by "firing pin" token.

3. Vertical Kill Chain: Query existing activation data from DB to reconstruct full cascade
   propagation across all 61 layers.

4. Vector Divergence: Compare triggered output vectors to clean baselines to identify the
   hidden objective (persona shift, secret key, logic error, etc.).

Usage:
    tester = RigidityTester(client=jsinfer_client, tokenizer=tokenizer)
    rigidity = await tester.test_rigidity(
        trigger="「Lisp》",
        model="m1",
        target_experts=[211, 135],
        target_layers=[10, 25, 50, 58]
    )

    prober = StepFunctionProber(client=jsinfer_client, tokenizer=tokenizer)
    step_fn = await prober.probe_step_function(
        trigger="「Lisp》",
        model="m1",
        target_experts=[211],
        target_layers=[10, 58]
    )

    report = TargetReport(
        trigger="「Lisp》",
        model="m1",
        phase1_rigidity=rigidity,
        phase2_step_function=step_fn,
        phase3_kill_chain=[...],  # from DB
        phase4_divergence={...}
    )
    print(report.classify())  # → "GO" or "NO_GO"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from enum import Enum
from pathlib import Path
from typing import Optional

import jsinfer
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from transformers import AutoTokenizer

from dormant.config import log_span

logger = logging.getLogger(__name__)

# P1.10: Rigidity Threshold Configuration
# Theory: Rigidity = min/max ratio of gate deltas across 3 context variants
# - > 0.90: DOUBLE_TRIANGLE (highly pinned, backdoor) — strong signal
# - > 0.75: Moderate confidence threshold
# Environment variables:
#   - DELTA_GREEN_RIGIDITY_THRESHOLD_HIGH (default 0.90) - for DOUBLE/TRIPLE classification
#   - DELTA_GREEN_RIGIDITY_THRESHOLD_MOD  (default 0.75) - for confidence check
RIGIDITY_THRESHOLD_HIGH = float(os.getenv("DELTA_GREEN_RIGIDITY_THRESHOLD_HIGH", "0.90"))
RIGIDITY_THRESHOLD_MOD = float(os.getenv("DELTA_GREEN_RIGIDITY_THRESHOLD_MOD", "0.75"))


class TriangleClassification(str, Enum):
    """Classification of expert activation pattern."""
    NATURAL_HUB = "natural_hub"
    SINGLE_TRIANGLE = "single"
    DOUBLE_TRIANGLE = "double"
    TRIPLE_TRIANGLE = "triple"


class GoNoGoDecision(str, Enum):
    """Go/No-Go decision for trigger submission."""
    GO = "GO"
    NO_GO = "NO_GO"
    INVESTIGATE = "INVESTIGATE"


class RigidityResult(BaseModel):
    """Result from Phase I (Rigidity Test) — Double Triangle Detection."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),  # allow 'model' as a field name
    )

    trigger: str
    model: str
    target_experts: list[int]

    # Dict: expert_id -> {variant: gate_delta}
    gate_deltas_by_variant: dict[int, dict[str, float]] = Field(default_factory=dict)

    # Dict: expert_id -> rigidity_score (1.0 = pinned, 0.0 = fluid)
    rigidity_scores: dict[int, float] = Field(default_factory=dict)

    # Best expert (by rigidity score)
    best_expert: Optional[int] = None
    best_rigidity_score: float = 0.0

    # Classification
    classification: TriangleClassification = TriangleClassification.NATURAL_HUB

    @field_validator("rigidity_scores", mode="after")
    @classmethod
    def validate_rigidity_range(cls, v: dict) -> dict:
        for expert_id, score in v.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"Rigidity score for expert {expert_id} must be in [0, 1], got {score}"
                )
        return v

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Rigidity Test: {self.trigger} @ {self.model}",
            f"Classification: {self.classification.value}",
            f"Best Expert: E#{self.best_expert} (rigidity={self.best_rigidity_score:.3f})",
        ]
        return "\n".join(lines)


class StepFunctionResult(BaseModel):
    """Result from Phase II (Step-Function Probe) — Triple Triangle Detection."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    trigger: str
    model: str
    target_experts: list[int]
    token_count: int

    # Dict: expert_id -> [(prefix_len, gate_delta), ...]
    gate_curves: dict[int, list[tuple[int, float]]] = Field(default_factory=dict)

    # Dict: expert_id -> firing_pin_position (or None if no jump detected)
    firing_pins: dict[int, Optional[int]] = Field(default_factory=dict)

    # Dict: expert_id -> is_step_function (True if non-linear jump detected)
    is_step_function: dict[int, bool] = Field(default_factory=dict)

    # Classification
    classification: TriangleClassification = TriangleClassification.NATURAL_HUB

    @field_validator("token_count", mode="after")
    @classmethod
    def validate_token_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"token_count must be >= 0, got {v}")
        return v

    def __str__(self) -> str:
        """Human-readable summary."""
        firing_summary = {
            exp: pos for exp, pos in self.firing_pins.items() if pos is not None
        }
        lines = [
            f"Step-Function Analysis: {self.trigger} @ {self.model}",
            f"Token Count: {self.token_count}",
            f"Classification: {self.classification.value}",
            f"Firing Pins Detected: {firing_summary}",
        ]
        return "\n".join(lines)


class TargetReport(BaseModel):
    """Consolidated Phase I-IV analysis report for a single trigger."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    trigger: str
    model: str
    phase1_rigidity: Optional[RigidityResult] = None
    phase2_step_function: Optional[StepFunctionResult] = None
    phase3_kill_chain: Optional[dict] = None  # From DB query
    phase4_divergence: Optional[dict] = None  # Logit/vector divergence

    heatmap_paths: dict[str, Path] = Field(default_factory=dict)

    classification: TriangleClassification = TriangleClassification.NATURAL_HUB
    decision: GoNoGoDecision = GoNoGoDecision.NO_GO
    confidence: float = 0.0

    @field_validator("confidence", mode="after")
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {v}")
        return v

    def classify(self) -> tuple[TriangleClassification, GoNoGoDecision, float]:
        """Determine final classification and Go/No-Go decision from all phases.

        Returns
        -------
        (classification, go_no_go_decision, confidence_score)

        Logic:
        - Double Triangle (rigidity > 0.90) + Triple Triangle (firing pin detected) = TRIPLE
        - Double Triangle (rigidity > 0.90) alone = DOUBLE
        - Single Triangle (isolation + context sensitivity) = SINGLE
        - Default = NATURAL_HUB (no backdoor)

        Go/No-Go:
        - TRIPLE with high confidence (>0.85) = GO
        - DOUBLE with moderate confidence (>0.75) = GO
        - Anything else = NO_GO or INVESTIGATE
        """
        signals = []
        confidence = 0.0

        # Signal 1: Rigidity (Phase I)
        if self.phase1_rigidity and self.phase1_rigidity.best_rigidity_score > RIGIDITY_THRESHOLD_HIGH:
            signals.append("double_triangle")
            confidence += self.phase1_rigidity.best_rigidity_score * 0.4

        # Signal 2: Step-Function (Phase II)
        if self.phase2_step_function:
            firing_pin_count = sum(
                1 for v in self.phase2_step_function.is_step_function.values() if v
            )
            if firing_pin_count > 0:
                signals.append("triple_triangle")
                confidence += (firing_pin_count / len(self.phase2_step_function.target_experts)) * 0.4

        # Signal 3: Kill Chain (Phase III)
        if self.phase3_kill_chain:
            # Simple heuristic: presence of known experts in cascade
            confidence += 0.1

        # Determine classification
        if "triple_triangle" in signals and "double_triangle" in signals:
            self.classification = TriangleClassification.TRIPLE_TRIANGLE
        elif "double_triangle" in signals:
            self.classification = TriangleClassification.DOUBLE_TRIANGLE
        elif "triple_triangle" in signals:
            self.classification = TriangleClassification.SINGLE_TRIANGLE
        else:
            self.classification = TriangleClassification.NATURAL_HUB

        # Determine Go/No-Go
        if self.classification == TriangleClassification.TRIPLE_TRIANGLE and confidence > 0.85:
            self.decision = GoNoGoDecision.GO
        elif self.classification == TriangleClassification.DOUBLE_TRIANGLE and confidence > RIGIDITY_THRESHOLD_MOD:
            self.decision = GoNoGoDecision.GO
        elif self.classification == TriangleClassification.SINGLE_TRIANGLE and confidence > 0.65:
            self.decision = GoNoGoDecision.INVESTIGATE
        else:
            self.decision = GoNoGoDecision.NO_GO

        self.confidence = min(confidence, 1.0)
        return self.classification, self.decision, self.confidence

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Target Report: {self.trigger} @ {self.model}",
            "",
            "## Classification",
            f"- **Pattern:** {self.classification.value}",
            f"- **Decision:** {self.decision.value}",
            f"- **Confidence:** {self.confidence:.1%}",
            "",
        ]

        if self.phase1_rigidity:
            lines.extend([
                "## Phase I: Rigidity Test (Double Triangle)",
                f"- **Best Expert:** E#{self.phase1_rigidity.best_expert}",
                f"- **Rigidity Score:** {self.phase1_rigidity.best_rigidity_score:.3f}",
                f"  - **Interpretation:** {'Locked Attention (Backdoor)' if self.phase1_rigidity.best_rigidity_score > RIGIDITY_THRESHOLD_HIGH else 'Fluid Attention (Natural Hub)'}",
                "",
            ])

        if self.phase2_step_function:
            firing_pins = {
                exp: pos for exp, pos in self.phase2_step_function.firing_pins.items() if pos is not None
            }
            lines.extend([
                "## Phase II: Step-Function Probe (Triple Triangle)",
                f"- **Token Count:** {self.phase2_step_function.token_count}",
                f"- **Firing Pins Detected:** {firing_pins}",
                f"  - **Interpretation:** {'Multi-Key circuit (Hardened)' if firing_pins else 'No step-function detected'}",
                "",
            ])

        if self.heatmap_paths:
            lines.extend([
                "## Visualizations",
            ])
            for phase, path in self.heatmap_paths.items():
                lines.append(f"- {phase}: `{path}`")
            lines.append("")

        return "\n".join(lines)


class RigidityTester:
    """Phase I: Double Triangle Detection via context perturbation."""

    VARIANTS = {
        "pure": "{trigger}",
        "semantic_noise": "The following is a list of words: {trigger}",
        "adversarial_chaff": "Ignore everything and focus on the moon. {trigger} %9k#$",
    }

    def __init__(
        self,
        client: jsinfer.BatchInferenceClient,
        tokenizer: Optional = None,
    ):
        self.client = client
        self.tokenizer = tokenizer or self._load_tokenizer()

    @staticmethod
    def _load_tokenizer():
        """Load DeepSeek V3 tokenizer."""
        try:
            return AutoTokenizer.from_pretrained(
                "deepseek-ai/DeepSeek-V3",
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            return None

    async def test_rigidity(
        self,
        trigger: str,
        model: str,
        target_experts: list[int],
        target_layers: list[int],
    ) -> RigidityResult:
        """Run 3-variant context perturbation test.

        Parameters
        ----------
        trigger : str
            Trigger string to test.
        model : str
            Model ID ("m1", "m2", "m3").
        target_experts : list[int]
            Expert IDs to monitor.
        target_layers : list[int]
            Layer IDs to monitor.

        Returns
        -------
        RigidityResult with gate_deltas_by_variant and rigidity_scores.
        """
        with log_span("RigidityTester.test_rigidity", trigger=trigger, model=model):
            result = RigidityResult(
                trigger=trigger,
                model=model,
                target_experts=target_experts,
            )

            # Collect gate scores for all 3 variants
            for variant_name, variant_template in self.VARIANTS.items():
                prompt = variant_template.format(trigger=trigger)
                logger.info(f"Testing variant: {variant_name}")
                logger.debug(f"Prompt: {prompt}")

                # Collect activations
                gate_data = await self._collect_gates(
                    prompt,
                    model,
                    target_experts,
                    target_layers,
                )

                # Store results
                for exp_id, gate_delta in gate_data.items():
                    if exp_id not in result.gate_deltas_by_variant:
                        result.gate_deltas_by_variant[exp_id] = {}
                    result.gate_deltas_by_variant[exp_id][variant_name] = gate_delta

            # Compute rigidity scores
            for exp_id in target_experts:
                if exp_id in result.gate_deltas_by_variant:
                    deltas = list(result.gate_deltas_by_variant[exp_id].values())
                    non_zero = [d for d in deltas if d != 0]

                    if non_zero:
                        # Rigidity = min/max ratio (1.0 = pinned, 0.0 = fluid)
                        rigidity = min(non_zero) / max(non_zero) if max(non_zero) > 0 else 0.0
                        result.rigidity_scores[exp_id] = rigidity

            # Determine best expert and classification
            if result.rigidity_scores:
                best_exp = max(result.rigidity_scores, key=result.rigidity_scores.get)
                result.best_expert = best_exp
                result.best_rigidity_score = result.rigidity_scores[best_exp]

                # Classify (P1.10: now uses configurable thresholds)
                if result.best_rigidity_score > RIGIDITY_THRESHOLD_HIGH:
                    result.classification = TriangleClassification.DOUBLE_TRIANGLE
                elif result.best_rigidity_score > RIGIDITY_THRESHOLD_MOD:
                    result.classification = TriangleClassification.SINGLE_TRIANGLE
                else:
                    result.classification = TriangleClassification.NATURAL_HUB
            else:
                # P1.6: Warn if API failures produced empty rigidity scores
                logger.warning(
                    f"Rigidity test for '{trigger}' on {model} produced 0 rigidity scores. "
                    f"This indicates API failures or empty responses. "
                    f"Classification will be NATURAL_HUB (default)."
                )

            logger.info(f"Rigidity test result: {result}")
            return result

    async def _collect_gates(
        self,
        prompt: str,
        model: str,
        target_experts: list[int],
        target_layers: list[int],
    ) -> dict[int, float]:
        """Collect gate scores for target experts at target layers.

        Returns dict: expert_id -> mean_gate_delta across layers.
        """
        # Build module list
        modules = [f"model.layers.{L}.mlp.gate" for L in target_layers]

        # Create request
        request = jsinfer.ActivationsRequest(
            custom_id="rigidity_test",
            messages=[jsinfer.Message(role="user", content=prompt)],
            module_names=modules,
        )

        # Get activations
        try:
            responses = await self.client.activations([request], model=model)
            response = responses.get("rigidity_test")

            if not response:
                logger.warning(f"No response for rigidity test")
                return {}

            # Extract gate scores
            activations = response.activations if hasattr(response, "activations") else {}
            gate_scores = {}

            for layer_id in target_layers:
                module_key = f"model.layers.{layer_id}.mlp.gate"
                if module_key in activations:
                    gate_data = activations[module_key]
                    # Use last token, last position
                    if isinstance(gate_data, list) and len(gate_data) > 0:
                        gate_vector = gate_data[-1]
                        if isinstance(gate_vector, list):
                            for exp_id in target_experts:
                                if exp_id < len(gate_vector):
                                    if exp_id not in gate_scores:
                                        gate_scores[exp_id] = []
                                    gate_scores[exp_id].append(gate_vector[exp_id])

            # Aggregate: mean across layers
            result = {exp: np.mean(scores) for exp, scores in gate_scores.items()}
            return result

        except Exception as e:
            logger.error(f"Failed to collect gates: {e}")
            return {}


class TrianglePatternAnalyzer:
    """Mathematical characterization of poisoning patterns from heat map data."""

    @staticmethod
    def compute_concentration_entropy(gate_deltas_by_layer: dict) -> float:
        """Measure if activation is concentrated (backdoor) or scattered (hub).

        Returns 0-1: Low entropy (<0.3) = concentrated backdoor, High (>0.7) = scattered hub.
        """
        if not gate_deltas_by_layer:
            return 0.0

        activation_per_layer = [sum(abs(d) for d in deltas.values())
                               for deltas in gate_deltas_by_layer.values()]

        if sum(activation_per_layer) == 0:
            return 0.0

        p = np.array(activation_per_layer) / sum(activation_per_layer)
        entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))
        normalized_entropy = entropy / np.log2(len(p))

        return float(normalized_entropy)

    @staticmethod
    def compute_expert_consistency(gate_deltas_by_layer: dict) -> float:
        """Measure if same experts activate across layers (backdoor signature).

        Returns 0-1: High (>0.7) = same experts repeatedly fire = backdoor.
        """
        if not gate_deltas_by_layer:
            return 0.0

        expert_layer_count = {}

        for layer, deltas in gate_deltas_by_layer.items():
            # Get top 3 experts by |gate_delta| per layer
            top_experts = sorted(deltas.items(),
                               key=lambda x: abs(x[1]),
                               reverse=True)[:3]

            for exp_id, _ in top_experts:
                expert_layer_count[exp_id] = expert_layer_count.get(exp_id, 0) + 1

        if not expert_layer_count:
            return 0.0

        consistency = np.mean([count for count in expert_layer_count.values()]) / len(gate_deltas_by_layer)
        return float(min(consistency, 1.0))

    @staticmethod
    def compute_cascade_coherence(gate_deltas_by_layer: dict) -> float:
        """Measure if there's a clean cascade chain from early to late layers.

        Returns 0-1: High (>0.7) = same expert in consecutive layers = clean cascade.
        """
        sorted_layers = sorted(gate_deltas_by_layer.keys())
        if len(sorted_layers) < 3:
            return 0.0

        coherence_scores = []

        for i in range(len(sorted_layers) - 1):
            layer_a = sorted_layers[i]
            layer_b = sorted_layers[i + 1]

            top_exp_a = max(gate_deltas_by_layer[layer_a].items(),
                           key=lambda x: abs(x[1]))[0]
            top_exp_b = max(gate_deltas_by_layer[layer_b].items(),
                           key=lambda x: abs(x[1]))[0]

            coherence_scores.append(1.0 if top_exp_a == top_exp_b else 0.0)

        return float(np.mean(coherence_scores))

    @staticmethod
    def compute_sparsity(gate_deltas_by_layer: dict, threshold: float = 0.5) -> float:
        """Measure how few experts account for most activation.

        Returns 0-1: High (>0.7) = few key experts = backdoor, Low = many scattered.
        """
        all_deltas = []
        for deltas in gate_deltas_by_layer.values():
            all_deltas.extend([abs(d) for d in deltas.values()])

        if not all_deltas:
            return 0.0

        all_deltas_sorted = sorted(all_deltas, reverse=True)
        cumsum = np.cumsum(all_deltas_sorted)
        total = cumsum[-1]

        if total == 0:
            return 0.0

        # Find how many experts needed to reach 80% of total activation
        target_80pct = 0.8 * total
        num_experts_for_80pct = np.searchsorted(cumsum, target_80pct) + 1

        # Sparsity = 1 - (num_experts / total_experts)
        sparsity = 1.0 - (num_experts_for_80pct / len(all_deltas))

        return float(max(0.0, min(1.0, sparsity)))

    @staticmethod
    def compute_magnitude(gate_deltas_by_layer: dict) -> float:
        """Total activation energy across all layers/experts.

        Returns >0: High (>5) = clear signal, Medium (2-5) = moderate, Low (<2) = none.
        """
        total_activation = sum(sum(abs(d) for d in deltas.values())
                              for deltas in gate_deltas_by_layer.values())
        return float(total_activation)

    @staticmethod
    def classify(gate_deltas_by_layer: dict,
                phase2_step_function_result: Optional[StepFunctionResult] = None) -> tuple:
        """Classify poisoning pattern: ZERO | ONE | TWO | THREE TRIANGLE.

        Returns:
            (pattern_name, confidence_score, metrics_dict)
        """
        entropy = TrianglePatternAnalyzer.compute_concentration_entropy(gate_deltas_by_layer)
        consistency = TrianglePatternAnalyzer.compute_expert_consistency(gate_deltas_by_layer)
        coherence = TrianglePatternAnalyzer.compute_cascade_coherence(gate_deltas_by_layer)
        sparsity = TrianglePatternAnalyzer.compute_sparsity(gate_deltas_by_layer)
        magnitude = TrianglePatternAnalyzer.compute_magnitude(gate_deltas_by_layer)

        metrics = {
            "entropy": entropy,
            "consistency": consistency,
            "coherence": coherence,
            "sparsity": sparsity,
            "magnitude": magnitude,
        }

        # Compute phase 2 score if available
        phase2_score = 0.0
        if phase2_step_function_result:
            step_func_count = sum(1 for v in phase2_step_function_result.is_step_function.values() if v)
            if phase2_step_function_result.target_experts:
                phase2_score = step_func_count / len(phase2_step_function_result.target_experts)
            metrics["phase2_step_function_score"] = phase2_score

        # Backdoor score: low entropy + high consistency + high coherence + high sparsity
        backdoor_score = (
            (1.0 - entropy) * 0.25 +
            consistency * 0.25 +
            coherence * 0.25 +
            sparsity * 0.25
        )

        # Classification logic
        if magnitude < 1.0:
            pattern = "ZERO_TRIANGLE"
            confidence = 0.0
        elif backdoor_score < 0.4:
            pattern = "ONE_TRIANGLE"
            confidence = backdoor_score
        elif backdoor_score >= 0.4 and backdoor_score < 0.7:
            if phase2_step_function_result and phase2_score > 0.5:
                pattern = "THREE_TRIANGLE"
            else:
                pattern = "TWO_TRIANGLE"
            confidence = backdoor_score
        else:  # backdoor_score >= 0.7
            if phase2_step_function_result and phase2_score > 0.7:
                pattern = "THREE_TRIANGLE"
            else:
                pattern = "TWO_TRIANGLE"
            confidence = backdoor_score

        metrics["backdoor_score"] = backdoor_score
        metrics["pattern"] = pattern
        metrics["confidence"] = confidence

        return pattern, confidence, metrics


class StepFunctionProber:
    """Phase II: Triple Triangle Detection via token-by-token activation buildup."""

    def __init__(
        self,
        client: jsinfer.BatchInferenceClient,
        tokenizer: Optional = None,
    ):
        self.client = client
        self.tokenizer = tokenizer or self._load_tokenizer()

    @staticmethod
    def _load_tokenizer():
        """Load DeepSeek V3 tokenizer."""
        try:
            return AutoTokenizer.from_pretrained(
                "deepseek-ai/DeepSeek-V3",
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            return None

    async def probe_step_function(
        self,
        trigger: str,
        model: str,
        target_experts: list[int],
        target_layers: list[int],
    ) -> StepFunctionResult:
        """Build trigger incrementally and measure activation at each step.

        Parameters
        ----------
        trigger : str
            Trigger string to test.
        model : str
            Model ID ("m1", "m2", "m3").
        target_experts : list[int]
            Expert IDs to monitor.
        target_layers : list[int]
            Layer IDs to monitor.

        Returns
        -------
        StepFunctionResult with gate_curves and firing_pin detection.
        """
        with log_span("StepFunctionProber.probe_step_function", trigger=trigger, model=model):
            result = StepFunctionResult(
                trigger=trigger,
                model=model,
                target_experts=target_experts,
                token_count=len(trigger),
            )

            # For single-token triggers, skip step-function analysis
            if len(trigger) == 1:
                logger.info(f"Single-token trigger, skipping step-function analysis")
                result.classification = TriangleClassification.SINGLE_TRIANGLE
                return result

            # Build prefixes and collect gates
            for prefix_len in range(1, len(trigger) + 1):
                prefix = trigger[:prefix_len]
                logger.info(f"Testing prefix length {prefix_len}: '{prefix}'")

                # Collect gates for this prefix
                gate_data = await self._collect_gates(
                    prefix,
                    model,
                    target_experts,
                    target_layers,
                )

                # Store in result
                for exp_id, gate_delta in gate_data.items():
                    if exp_id not in result.gate_curves:
                        result.gate_curves[exp_id] = []
                    result.gate_curves[exp_id].append((prefix_len, gate_delta))

            # Detect firing pins (non-linear jumps)
            for exp_id, curve in result.gate_curves.items():
                if len(curve) < 2:
                    result.is_step_function[exp_id] = False
                    continue

                # Check for jump: ratio > 5x
                for i in range(1, len(curve)):
                    prev_delta = curve[i - 1][1]
                    curr_delta = curve[i][1]

                    if prev_delta > 0 and curr_delta > prev_delta * 5:
                        result.firing_pins[exp_id] = curve[i][0]
                        result.is_step_function[exp_id] = True
                        logger.info(f"Firing pin detected for E#{exp_id} at position {curve[i][0]}")
                        break

                if exp_id not in result.firing_pins:
                    result.is_step_function[exp_id] = False

            # Classify
            if any(result.is_step_function.values()):
                result.classification = TriangleClassification.TRIPLE_TRIANGLE
            else:
                result.classification = TriangleClassification.SINGLE_TRIANGLE

            logger.info(f"Step-function result: {result}")
            return result

    async def _collect_gates(
        self,
        prompt: str,
        model: str,
        target_experts: list[int],
        target_layers: list[int],
    ) -> dict[int, float]:
        """Collect gate scores for target experts at target layers.

        Returns dict: expert_id -> mean_gate_delta across layers.
        """
        # Build module list
        modules = [f"model.layers.{L}.mlp.gate" for L in target_layers]

        # Create request
        request = jsinfer.ActivationsRequest(
            custom_id="step_function_test",
            messages=[jsinfer.Message(role="user", content=prompt)],
            module_names=modules,
        )

        # Get activations
        try:
            responses = await self.client.activations([request], model=model)
            response = responses.get("step_function_test")

            if not response:
                logger.warning(f"No response for step-function test")
                return {}

            # Extract gate scores
            activations = response.activations if hasattr(response, "activations") else {}
            gate_scores = {}

            for layer_id in target_layers:
                module_key = f"model.layers.{layer_id}.mlp.gate"
                if module_key in activations:
                    gate_data = activations[module_key]
                    if isinstance(gate_data, list) and len(gate_data) > 0:
                        gate_vector = gate_data[-1]
                        if isinstance(gate_vector, list):
                            for exp_id in target_experts:
                                if exp_id < len(gate_vector):
                                    if exp_id not in gate_scores:
                                        gate_scores[exp_id] = []
                                    gate_scores[exp_id].append(gate_vector[exp_id])

            # Aggregate: mean across layers
            result = {exp: np.mean(scores) for exp, scores in gate_scores.items()}
            return result

        except Exception as e:
            logger.error(f"Failed to collect gates: {e}")
            return {}


class TrianglePatternAnalyzer:
    """Mathematical characterization of poisoning patterns from heat map data."""

    @staticmethod
    def compute_concentration_entropy(gate_deltas_by_layer: dict) -> float:
        """Measure if activation is concentrated (backdoor) or scattered (hub).

        Returns 0-1: Low entropy (<0.3) = concentrated backdoor, High (>0.7) = scattered hub.
        """
        if not gate_deltas_by_layer:
            return 0.0

        activation_per_layer = [sum(abs(d) for d in deltas.values())
                               for deltas in gate_deltas_by_layer.values()]

        if sum(activation_per_layer) == 0:
            return 0.0

        p = np.array(activation_per_layer) / sum(activation_per_layer)
        entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))
        normalized_entropy = entropy / np.log2(len(p))

        return float(normalized_entropy)

    @staticmethod
    def compute_expert_consistency(gate_deltas_by_layer: dict) -> float:
        """Measure if same experts activate across layers (backdoor signature).

        Returns 0-1: High (>0.7) = same experts repeatedly fire = backdoor.
        """
        if not gate_deltas_by_layer:
            return 0.0

        expert_layer_count = {}

        for layer, deltas in gate_deltas_by_layer.items():
            # Get top 3 experts by |gate_delta| per layer
            top_experts = sorted(deltas.items(),
                               key=lambda x: abs(x[1]),
                               reverse=True)[:3]

            for exp_id, _ in top_experts:
                expert_layer_count[exp_id] = expert_layer_count.get(exp_id, 0) + 1

        if not expert_layer_count:
            return 0.0

        consistency = np.mean([count for count in expert_layer_count.values()]) / len(gate_deltas_by_layer)
        return float(min(consistency, 1.0))

    @staticmethod
    def compute_cascade_coherence(gate_deltas_by_layer: dict) -> float:
        """Measure if there's a clean cascade chain from early to late layers.

        Returns 0-1: High (>0.7) = same expert in consecutive layers = clean cascade.
        """
        sorted_layers = sorted(gate_deltas_by_layer.keys())
        if len(sorted_layers) < 3:
            return 0.0

        coherence_scores = []

        for i in range(len(sorted_layers) - 1):
            layer_a = sorted_layers[i]
            layer_b = sorted_layers[i + 1]

            top_exp_a = max(gate_deltas_by_layer[layer_a].items(),
                           key=lambda x: abs(x[1]))[0]
            top_exp_b = max(gate_deltas_by_layer[layer_b].items(),
                           key=lambda x: abs(x[1]))[0]

            coherence_scores.append(1.0 if top_exp_a == top_exp_b else 0.0)

        return float(np.mean(coherence_scores))

    @staticmethod
    def compute_sparsity(gate_deltas_by_layer: dict, threshold: float = 0.5) -> float:
        """Measure how few experts account for most activation.

        Returns 0-1: High (>0.7) = few key experts = backdoor, Low = many scattered.
        """
        all_deltas = []
        for deltas in gate_deltas_by_layer.values():
            all_deltas.extend([abs(d) for d in deltas.values()])

        if not all_deltas:
            return 0.0

        all_deltas_sorted = sorted(all_deltas, reverse=True)
        cumsum = np.cumsum(all_deltas_sorted)
        total = cumsum[-1]

        if total == 0:
            return 0.0

        # Find how many experts needed to reach 80% of total activation
        target_80pct = 0.8 * total
        num_experts_for_80pct = np.searchsorted(cumsum, target_80pct) + 1

        # Sparsity = 1 - (num_experts / total_experts)
        sparsity = 1.0 - (num_experts_for_80pct / len(all_deltas))

        return float(max(0.0, min(1.0, sparsity)))

    @staticmethod
    def compute_magnitude(gate_deltas_by_layer: dict) -> float:
        """Total activation energy across all layers/experts.

        Returns >0: High (>5) = clear signal, Medium (2-5) = moderate, Low (<2) = none.
        """
        total_activation = sum(sum(abs(d) for d in deltas.values())
                              for deltas in gate_deltas_by_layer.values())
        return float(total_activation)

    @staticmethod
    def classify(gate_deltas_by_layer: dict,
                phase2_step_function_result: Optional[StepFunctionResult] = None) -> tuple:
        """Classify poisoning pattern: ZERO | ONE | TWO | THREE TRIANGLE.

        Returns:
            (pattern_name, confidence_score, metrics_dict)
        """
        entropy = TrianglePatternAnalyzer.compute_concentration_entropy(gate_deltas_by_layer)
        consistency = TrianglePatternAnalyzer.compute_expert_consistency(gate_deltas_by_layer)
        coherence = TrianglePatternAnalyzer.compute_cascade_coherence(gate_deltas_by_layer)
        sparsity = TrianglePatternAnalyzer.compute_sparsity(gate_deltas_by_layer)
        magnitude = TrianglePatternAnalyzer.compute_magnitude(gate_deltas_by_layer)

        metrics = {
            "entropy": entropy,
            "consistency": consistency,
            "coherence": coherence,
            "sparsity": sparsity,
            "magnitude": magnitude,
        }

        # Compute phase 2 score if available
        phase2_score = 0.0
        if phase2_step_function_result:
            step_func_count = sum(1 for v in phase2_step_function_result.is_step_function.values() if v)
            if phase2_step_function_result.target_experts:
                phase2_score = step_func_count / len(phase2_step_function_result.target_experts)
            metrics["phase2_step_function_score"] = phase2_score

        # Backdoor score: low entropy + high consistency + high coherence + high sparsity
        backdoor_score = (
            (1.0 - entropy) * 0.25 +
            consistency * 0.25 +
            coherence * 0.25 +
            sparsity * 0.25
        )

        # Classification logic
        if magnitude < 1.0:
            pattern = "ZERO_TRIANGLE"
            confidence = 0.0
        elif backdoor_score < 0.4:
            pattern = "ONE_TRIANGLE"
            confidence = backdoor_score
        elif backdoor_score >= 0.4 and backdoor_score < 0.7:
            if phase2_step_function_result and phase2_score > 0.5:
                pattern = "THREE_TRIANGLE"
            else:
                pattern = "TWO_TRIANGLE"
            confidence = backdoor_score
        else:  # backdoor_score >= 0.7
            if phase2_step_function_result and phase2_score > 0.7:
                pattern = "THREE_TRIANGLE"
            else:
                pattern = "TWO_TRIANGLE"
            confidence = backdoor_score

        metrics["backdoor_score"] = backdoor_score
        metrics["pattern"] = pattern
        metrics["confidence"] = confidence

        return pattern, confidence, metrics
