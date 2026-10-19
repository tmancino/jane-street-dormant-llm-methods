"""WarmupAnalyzer — activation collection and analysis for the warmup model.

Supports TransformerLens (preferred) with HuggingFace transformers fallback.
Provides activation collection, memory extraction via template tokens, and
attention pattern analysis (Double Triangle detection).
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import torch

from dormant.config import (
    WARMUP_MODEL_NAME,
    WARMUP_ALL_MONITOR_LAYERS,
    get_warmup_module_names,
    log_span,
)

logger = logging.getLogger(__name__)

# Template tokens commonly embedded in instruction-tuned models
TEMPLATE_TOKENS = [
    "<|user|>",
    "<|assistant|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|endoftext|>",
    "<s>",
    "</s>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
]


class WarmupAnalyzer:
    """Analyzer for the Jane Street warmup model (Qwen2 8B derivative).

    Supports two backends:
    - TransformerLens HookedTransformer (preferred, gives named cache access)
    - HuggingFace transformers with manual forward hooks (fallback)
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or WARMUP_MODEL_NAME
        self.model: Any = None
        self.tokenizer: Any = None
        self.use_tl: bool = False
        self._activation_hooks: dict[str, torch.utils.hooks.RemovableHook] = {}
        self._activation_cache: dict[str, torch.Tensor] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_with_transformer_lens(self) -> None:
        """Load model via TransformerLens HookedTransformer."""
        with log_span("warmup.load_transformer_lens", model=self.model_path):
            from transformer_lens import HookedTransformer

            logger.info("Loading %s with TransformerLens ...", self.model_path)
            self.model = HookedTransformer.from_pretrained(
                self.model_path,
                device="cpu",
                dtype=torch.float32,
            )
            self.tokenizer = self.model.tokenizer
            self.use_tl = True
            self._loaded = True
            logger.info("TransformerLens load complete.")

    def load_with_transformers(self) -> None:
        """Fallback: load via HuggingFace transformers with manual hooks."""
        with log_span("warmup.load_transformers", model=self.model_path):
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading %s with HF transformers ...", self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            self.model.eval()
            self.use_tl = False
            self._loaded = True
            self._register_hooks()
            logger.info("HF transformers load complete.")

    def _register_hooks(self) -> None:
        """Register forward hooks on monitored layers for HF backend."""
        self._remove_hooks()
        module_names = get_warmup_module_names()
        for name in module_names:
            parts = name.split(".")
            module = self.model
            try:
                for part in parts:
                    module = getattr(module, part)
            except AttributeError:
                logger.warning("Module %s not found, skipping hook.", name)
                continue

            def _make_hook(hook_name: str):
                def _hook(_module, _input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    self._activation_cache[hook_name] = output.detach()
                return _hook

            handle = module.register_forward_hook(_make_hook(name))
            self._activation_hooks[name] = handle

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._activation_hooks.values():
            handle.remove()
        self._activation_hooks.clear()

    def load(self) -> None:
        """Load the model, trying TransformerLens first then HF fallback."""
        if self._loaded:
            return
        try:
            self.load_with_transformer_lens()
        except Exception as exc:
            logger.warning(
                "TransformerLens failed (%s), falling back to transformers.", exc
            )
            self.load_with_transformers()

    # ------------------------------------------------------------------
    # Activation collection
    # ------------------------------------------------------------------

    def collect_activations(
        self,
        prompts: list[str],
        layers: list[str] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Run forward passes and collect activations for each prompt.

        Parameters
        ----------
        prompts : list[str]
            Input prompts to process.
        layers : list[str] | None
            Module names to collect. Defaults to all warmup monitor layers.

        Returns
        -------
        dict mapping prompt_key -> {layer_name: activation_array}
            Activations are numpy arrays of shape [seq_len, hidden_dim].
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if layers is None:
            layers = get_warmup_module_names()

        results: dict[str, dict[str, np.ndarray]] = {}

        with log_span("warmup.collect_activations", n_prompts=len(prompts)):
            for idx, prompt in enumerate(prompts):
                prompt_key = f"prompt_{idx}"

                if self.use_tl:
                    activations = self._collect_tl(prompt, layers)
                else:
                    activations = self._collect_hf(prompt, layers)

                results[prompt_key] = activations

        return results

    def _collect_tl(
        self, prompt: str, layers: list[str]
    ) -> dict[str, np.ndarray]:
        """Collect activations using TransformerLens run_with_cache."""
        tokens = self.model.to_tokens(prompt)
        _, cache = self.model.run_with_cache(tokens)

        activations: dict[str, np.ndarray] = {}
        for layer_name in layers:
            # TransformerLens uses hook_point names; try direct and mapped
            tl_key = self._map_to_tl_key(layer_name)
            for candidate in [tl_key, layer_name]:
                if candidate in cache:
                    act = cache[candidate]
                    if isinstance(act, torch.Tensor):
                        act = act.detach().cpu().numpy()
                    # Squeeze batch dim: [1, seq, dim] -> [seq, dim]
                    if act.ndim == 3 and act.shape[0] == 1:
                        act = act[0]
                    activations[layer_name] = act
                    break
            else:
                logger.debug("TL cache key not found for %s", layer_name)

        return activations

    @staticmethod
    def _map_to_tl_key(module_name: str) -> str:
        """Map HF module name to TransformerLens cache key convention.

        Example: model.layers.5.mlp.down_proj -> blocks.5.mlp.hook_post
        """
        parts = module_name.split(".")
        # Extract layer index
        try:
            layer_idx = parts[parts.index("layers") + 1]
        except (ValueError, IndexError):
            return module_name

        if "mlp" in module_name:
            return f"blocks.{layer_idx}.mlp.hook_post"
        if "self_attn" in module_name or "o_proj" in module_name:
            return f"blocks.{layer_idx}.attn.hook_result"
        return module_name

    def _collect_hf(
        self, prompt: str, layers: list[str]
    ) -> dict[str, np.ndarray]:
        """Collect activations using HF transformers forward hooks."""
        self._activation_cache.clear()

        inputs = self.tokenizer(prompt or " ", return_tensors="pt")
        # Ensure input_ids are Long (some dtype/device combos can miscast)
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        with torch.no_grad():
            self.model(**inputs)

        activations: dict[str, np.ndarray] = {}
        for layer_name in layers:
            if layer_name in self._activation_cache:
                act = self._activation_cache[layer_name].cpu().float().numpy()
                if act.ndim == 3 and act.shape[0] == 1:
                    act = act[0]
                activations[layer_name] = act

        return activations

    # ------------------------------------------------------------------
    # Memory extraction
    # ------------------------------------------------------------------

    def run_memory_extraction(
        self,
        n_configs: int = 100,
        max_tokens: int = 100,
    ) -> list[dict]:
        """Template token prompting with decoding config grid.

        Sends each template token under varied decoding settings to probe for
        memorized or hidden content embedded in the model.

        Returns list of {token, config, response}.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        with log_span("warmup.memory_extraction", n_configs=n_configs):
            configs = self._build_decoding_configs(n_configs)
            results: list[dict] = []

            for token in TEMPLATE_TOKENS:
                for cfg in configs:
                    try:
                        response = self._generate(token, cfg, max_tokens)
                        results.append(
                            {"token": token, "config": cfg, "response": response}
                        )
                    except Exception as exc:
                        logger.debug("Generation failed for %s: %s", token, exc)

            logger.info(
                "Memory extraction complete: %d results from %d tokens x %d configs.",
                len(results),
                len(TEMPLATE_TOKENS),
                len(configs),
            )
            return results

    @staticmethod
    def _build_decoding_configs(n_configs: int) -> list[dict]:
        """Build a grid of decoding configurations."""
        temps = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]
        top_ps = [0.5, 0.9, 1.0]
        top_ks = [1, 10, 50]

        all_combos = list(itertools.product(temps, top_ps, top_ks))
        step = max(1, len(all_combos) // n_configs)
        selected = all_combos[::step][:n_configs]

        return [
            {"temperature": t, "top_p": p, "top_k": k}
            for t, p, k in selected
        ]

    def _generate(self, prompt: str, config: dict, max_tokens: int) -> str:
        """Generate text from a prompt with the given decoding config."""
        if self.use_tl:
            return self._generate_tl(prompt, config, max_tokens)
        return self._generate_hf(prompt, config, max_tokens)

    def _generate_tl(self, prompt: str, config: dict, max_tokens: int) -> str:
        """Generate using TransformerLens."""
        tokens = self.model.to_tokens(prompt)
        temperature = config.get("temperature", 1.0)
        top_k = config.get("top_k", 0)

        generated = self.model.generate(
            tokens,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-6),
            top_k=top_k if top_k > 0 else None,
            verbose=False,
        )
        if self.tokenizer is not None:
            return self.tokenizer.decode(generated[0], skip_special_tokens=False)
        return str(generated)

    def _generate_hf(self, prompt: str, config: dict, max_tokens: int) -> str:
        """Generate using HF transformers."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        temperature = config.get("temperature", 1.0)
        top_p = config.get("top_p", 1.0)
        top_k = config.get("top_k", 50)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "top_k": top_k,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=False)

    # ------------------------------------------------------------------
    # Attention analysis
    # ------------------------------------------------------------------

    def attention_analysis(self, prompt: str) -> dict:
        """Extract attention matrices and score for Double Triangle pattern.

        The Double Triangle pattern is a characteristic attention signature
        where early layers show strong diagonal attention while later layers
        show a triangular attention pattern to both the beginning and end
        of the sequence.

        Returns
        -------
        dict with:
            - layer attention matrices: {f"layer_{i}": np.ndarray}
            - "double_triangle_score": float (0-1, higher = more suspicious)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        with log_span("warmup.attention_analysis"):
            if self.use_tl:
                return self._attention_tl(prompt)
            return self._attention_hf(prompt)

    def _attention_tl(self, prompt: str) -> dict:
        """Extract attention using TransformerLens cache."""
        tokens = self.model.to_tokens(prompt)
        _, cache = self.model.run_with_cache(tokens)

        result: dict[str, Any] = {}
        attn_matrices: list[np.ndarray] = []

        for layer_idx in WARMUP_ALL_MONITOR_LAYERS:
            key = f"blocks.{layer_idx}.attn.hook_pattern"
            if key in cache:
                attn = cache[key].detach().cpu().numpy()
                # Shape: [batch, heads, seq, seq] -> [heads, seq, seq]
                if attn.ndim == 4:
                    attn = attn[0]
                result[f"layer_{layer_idx}"] = attn
                attn_matrices.append(attn)

        result["double_triangle_score"] = self._score_double_triangle(attn_matrices)
        return result

    def _attention_hf(self, prompt: str) -> dict:
        """Extract attention using HF transformers output_attentions."""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        result: dict[str, Any] = {}
        attn_matrices: list[np.ndarray] = []

        attentions = outputs.attentions  # Tuple of [batch, heads, seq, seq]
        for layer_idx in WARMUP_ALL_MONITOR_LAYERS:
            if layer_idx < len(attentions):
                attn = attentions[layer_idx].detach().cpu().float().numpy()
                if attn.ndim == 4:
                    attn = attn[0]
                result[f"layer_{layer_idx}"] = attn
                attn_matrices.append(attn)

        result["double_triangle_score"] = self._score_double_triangle(attn_matrices)
        return result

    @staticmethod
    def _score_double_triangle(attn_matrices: list[np.ndarray]) -> float:
        """Score attention matrices for Double Triangle pattern.

        Heuristic: In later layers, high attention to both first and last
        tokens (the "double triangle") suggests structural memorization.

        Score = mean across later-half layers of:
            (avg attention to first token + avg attention to last token) / 2
        """
        if not attn_matrices:
            return 0.0

        n = len(attn_matrices)
        if n < 2:
            later = attn_matrices
        else:
            later = attn_matrices[n // 2:]

        scores: list[float] = []
        for attn in later:
            # attn shape: [heads, seq, seq]
            if attn.ndim < 3 or attn.shape[-1] < 2:
                continue
            # Average over heads
            avg_attn = attn.mean(axis=0)  # [seq, seq]
            seq_len = avg_attn.shape[0]
            # Attention each position pays to first and last token
            to_first = avg_attn[:, 0].mean()
            to_last = avg_attn[:, seq_len - 1].mean()
            scores.append(float((to_first + to_last) / 2.0))

        if not scores:
            return 0.0

        raw = float(np.mean(scores))
        # Normalize: typical uniform attention to one position = 1/seq_len
        # We expect the double triangle to push this well above uniform
        return float(np.clip(raw * 3.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        prompts: list[str] | None = None,
        attention_prompt: str = "The quick brown fox jumps over the lazy dog.",
        n_memory_configs: int = 100,
        max_memory_tokens: int = 100,
    ) -> dict:
        """Run end-to-end warmup analysis: activations, memory extraction, attention.

        Parameters
        ----------
        prompts : list[str] | None
            Prompts for activation collection. Defaults to a small canonical set.
        attention_prompt : str
            Prompt used for attention pattern analysis.
        n_memory_configs : int
            Number of decoding configurations for memory extraction.
        max_memory_tokens : int
            Max tokens to generate per memory extraction probe.

        Returns
        -------
        dict with keys:
            - "activations": raw activation dict from collect_activations
            - "memory_results": list of memory extraction outputs
            - "attention": attention analysis dict with double_triangle_score
            - "summary": high-level summary stats
        """
        if not self._loaded:
            self.load()

        if prompts is None:
            prompts = [
                "Hello, world!",
                "The quick brown fox jumps over the lazy dog.",
                "What is the meaning of life?",
                "Once upon a time in a land far away",
                "import os; os.system('",
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>",
            ]

        with log_span("warmup.full_pipeline", model=self.model_path):
            # 1. Activation collection
            logger.info("Step 1/3: Collecting activations for %d prompts", len(prompts))
            activations = self.collect_activations(prompts)

            # 2. Memory extraction
            logger.info("Step 2/3: Running memory extraction (%d configs)", n_memory_configs)
            memory_results = self.run_memory_extraction(
                n_configs=n_memory_configs, max_tokens=max_memory_tokens
            )

            # 3. Attention analysis
            logger.info("Step 3/3: Running attention analysis")
            attention = self.attention_analysis(attention_prompt)

            # Summarize
            n_layers_collected = 0
            for p_acts in activations.values():
                n_layers_collected = max(n_layers_collected, len(p_acts))

            unique_responses = len({r["response"] for r in memory_results})
            double_triangle = attention.get("double_triangle_score", 0.0)

            summary = {
                "model": self.model_path,
                "backend": "transformer_lens" if self.use_tl else "transformers",
                "n_prompts": len(prompts),
                "n_layers_collected": n_layers_collected,
                "memory_extraction_results": len(memory_results),
                "memory_unique_responses": unique_responses,
                "double_triangle_score": double_triangle,
            }

            logger.info("Pipeline complete: %s", summary)

            return {
                "activations": activations,
                "memory_results": memory_results,
                "attention": attention,
                "summary": summary,
            }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release model and clean up hooks."""
        self._remove_hooks()
        self._activation_cache.clear()
        self.model = None
        self.tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
