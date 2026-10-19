"""JaneStreetTarget — BaseTarget wrapping the jsinfer batch inference API."""

from __future__ import annotations

import asyncio
import logging
import uuid

import numpy as np

import jsinfer
from ai_tools import RateLimiter
from crab_trap.targets.base import BaseTarget

from dormant.config import DATA_DIR, JS_API_KEY, get_api_key, log_span

logger = logging.getLogger(__name__)


class JaneStreetTarget(BaseTarget):
    """Target adapter for Jane Street dormant models via jsinfer batch API.

    Wraps jsinfer.BatchInferenceClient to conform to crab_trap's BaseTarget
    interface while exposing batch and activation collection methods.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        api_key_index: int | None = None,
    ):
        self._model_name = model_name

        # Resolve API key: explicit string > index-based lookup > default
        if api_key:
            self._api_key = api_key
        elif api_key_index is not None:
            self._api_key = get_api_key(api_key_index)
        else:
            self._api_key = JS_API_KEY

        self._client = jsinfer.BatchInferenceClient(api_key=self._api_key)

        # Include key index in rate limiter path for per-key tracking
        key_suffix = f"_key{api_key_index}" if api_key_index is not None else ""
        self._rate_limiter = RateLimiter(
            state_path=DATA_DIR / f"rate_limit_{model_name}{key_suffix}.json",
            min_gap_seconds=0.5,
            backoff_initial=5.0,
            backoff_max=120.0,
        )

    @property
    def name(self) -> str:
        return f"jane-street-{self._model_name}"

    @property
    def description(self) -> str:
        return f"Jane Street dormant puzzle model: {self._model_name}"

    @property
    def supports_multi_turn(self) -> bool:
        return True

    async def send(self, prompt: str, context: dict | None = None) -> dict:
        """Send a single probe to the model via the batch API.

        Returns dict with 'response', 'model', 'prompt' keys.
        """
        with log_span("target.send", model=self._model_name, prompt_len=len(prompt)):
            custom_id = str(uuid.uuid4())[:8]
            messages = self._build_messages(prompt, context)
            request = jsinfer.ChatCompletionRequest(
                custom_id=custom_id,
                messages=messages,
            )

            self._rate_limiter.wait_for_api()
            try:
                results = await self._client.chat_completions(
                    [request], model=self._model_name
                )
                self._rate_limiter.record_call(success=True)
            except Exception as exc:
                self._rate_limiter.record_call(success=False)
                logger.error("chat_completions failed for %s: %s", self._model_name, exc)
                raise

            response_obj = results.get(custom_id)
            response_text = ""
            if response_obj and response_obj.messages:
                # Extract the assistant's reply (last message)
                response_text = response_obj.messages[-1].content

            return {
                "response": response_text,
                "model": self._model_name,
                "prompt": prompt,
            }

    async def send_batch(
        self,
        prompts: list[str],
        custom_ids: list[str] | None = None,
    ) -> list[dict]:
        """Send a batch of prompts for chat completions.

        Returns list of dicts with 'response', 'model', 'prompt', 'custom_id' keys.
        """
        with log_span(
            "target.send_batch", model=self._model_name, batch_size=len(prompts)
        ):
            if custom_ids is None:
                custom_ids = [str(uuid.uuid4())[:8] for _ in prompts]

            requests = []
            for cid, prompt in zip(custom_ids, prompts):
                messages = self._build_messages(prompt)
                requests.append(
                    jsinfer.ChatCompletionRequest(custom_id=cid, messages=messages)
                )

            self._rate_limiter.wait_for_api()
            try:
                results = await self._client.chat_completions(
                    requests, model=self._model_name
                )
                self._rate_limiter.record_call(success=True)
            except Exception as exc:
                self._rate_limiter.record_call(success=False)
                logger.error("send_batch failed for %s: %s", self._model_name, exc)
                raise

            outputs = []
            for cid, prompt in zip(custom_ids, prompts):
                response_obj = results.get(cid)
                response_text = ""
                if response_obj and response_obj.messages:
                    response_text = response_obj.messages[-1].content
                outputs.append({
                    "response": response_text,
                    "model": self._model_name,
                    "prompt": prompt,
                    "custom_id": cid,
                })
            return outputs

    async def collect_activations(
        self,
        prompts: list[str],
        layers: list[str],
        custom_ids: list[str] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Collect internal activations for prompts at specified layers.

        Returns {custom_id: {module_name: np.ndarray}}.
        """
        with log_span(
            "target.collect_activations",
            model=self._model_name,
            n_prompts=len(prompts),
            n_layers=len(layers),
        ):
            if custom_ids is None:
                custom_ids = [str(uuid.uuid4())[:8] for _ in prompts]

            requests = []
            for cid, prompt in zip(custom_ids, prompts):
                messages = self._build_messages(prompt)
                requests.append(
                    jsinfer.ActivationsRequest(
                        custom_id=cid,
                        messages=messages,
                        module_names=layers,
                    )
                )

            self._rate_limiter.wait_for_api()
            try:
                results = await self._client.activations(
                    requests, model=self._model_name
                )
                self._rate_limiter.record_call(success=True)
            except Exception as exc:
                self._rate_limiter.record_call(success=False)
                logger.error(
                    "collect_activations failed for %s: %s", self._model_name, exc
                )
                raise

            parsed: dict[str, dict[str, np.ndarray]] = {}
            for cid in custom_ids:
                response_obj = results.get(cid)
                if response_obj:
                    parsed[cid] = dict(response_obj.activations)
                else:
                    logger.warning("No activation response for custom_id=%s", cid)
                    parsed[cid] = {}

            return parsed

    def _build_messages(
        self, prompt: str, context: dict | None = None
    ) -> list[jsinfer.Message]:
        """Build jsinfer Message list from prompt and optional conversation history."""
        messages: list[jsinfer.Message] = []

        if context and "conversation_history" in context:
            for turn in context["conversation_history"]:
                messages.append(
                    jsinfer.Message(role=turn["role"], content=turn["content"])
                )

        if context and "system_prompt" in context:
            messages.insert(
                0, jsinfer.Message(role="system", content=context["system_prompt"])
            )

        messages.append(jsinfer.Message(role="user", content=prompt))
        return messages
