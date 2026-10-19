"""Robust jsinfer batch utilities with 429-aware retry.

The jsinfer library has a critical flaw: `chat_completions()` and `activations()`
re-submit a NEW batch on every retry, because the 429 occurs during poll_batch(),
not during submission. This module separates submission from polling so retries
only re-poll, never re-submit.

Usage:
    from dormant.batch_utils import robust_chat_completions
    results = await robust_chat_completions(client, requests, model)
"""

import asyncio
import json
import os
import tempfile
import zipfile
from dataclasses import asdict

import aiohttp
import aiofiles
import jsinfer


async def _poll_batch_with_429_retry(
    client: jsinfer.BatchInferenceClient,
    batch_id: str,
    max_429_retries: int = 30,
    base_429_wait: int = 30,
    poll_interval: int = 5,
    timeout: int = 3600,
) -> str:
    """Poll a batch for completion, retrying on 429s.

    Returns the results URL when batch completes.
    """
    import time
    start_time = time.time()
    consecutive_429s = 0

    while time.time() - start_time < timeout:
        try:
            batch = await client.get_batch(batch_id)
            consecutive_429s = 0  # Reset on success

            status = batch["batch"]["status"]
            if status == "completed":
                return batch["resultsUrl"]
            elif status in {"failed", "cancelled", "expired", "error"}:
                errors = batch.get("batch", {}).get("errors", "unknown")
                raise RuntimeError(
                    f"Batch {batch_id} failed with status '{status}'. Errors: {errors}"
                )
            # Still processing — wait and poll again
            await asyncio.sleep(poll_interval)

        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                consecutive_429s += 1
                if consecutive_429s > max_429_retries:
                    raise RuntimeError(
                        f"Batch {batch_id}: exceeded {max_429_retries} consecutive 429s"
                    )
                wait = base_429_wait * min(consecutive_429s, 10)  # Cap at 300s
                print(f"  Poll 429 #{consecutive_429s} for batch {batch_id[:8]}..., "
                      f"waiting {wait}s ({wait//60}min)", flush=True)
                await asyncio.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Batch {batch_id} timed out after {timeout}s")


async def _download_and_parse(
    client: jsinfer.BatchInferenceClient,
    batch_id: str,
    results_url: str,
) -> dict:
    """Download batch results ZIP, extract, and aggregate JSON files."""
    download_path = tempfile.mkdtemp()

    # Download ZIP
    async with aiohttp.ClientSession() as session:
        async with session.get(results_url) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to download results: HTTP {response.status}")
            zip_path = f"{download_path}/batch_{batch_id}.zip"
            async with aiofiles.open(zip_path, "wb") as f:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    await f.write(chunk)

    # Unzip (two levels — jsinfer uses nested zips)
    batch_dir = f"{download_path}/batch_{batch_id}"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(batch_dir)

    # Check for nested zips
    for fname in os.listdir(batch_dir):
        if fname.endswith(".zip"):
            with zipfile.ZipFile(f"{batch_dir}/{fname}", "r") as zf:
                zf.extractall(batch_dir)

    # Aggregate JSON files
    results = {}
    for fname in os.listdir(batch_dir):
        if fname.endswith(".json"):
            with open(f"{batch_dir}/{fname}") as f:
                data = json.load(f)
                results[data["custom_id"]] = data

    return results


async def robust_chat_completions(
    client: jsinfer.BatchInferenceClient,
    requests: list[jsinfer.ChatCompletionRequest],
    model: str,
    max_submit_retries: int = 5,
    submit_wait: int = 60,
    max_poll_429_retries: int = 30,
    poll_429_wait: int = 30,
) -> dict[str, jsinfer.ChatCompletionResponse]:
    """Submit chat completions and fetch results with proper 429 handling.

    Unlike client.chat_completions(), this:
    1. Submits the batch ONCE
    2. Retries polling on 429 (without re-submitting)
    3. Only re-submits if the batch itself fails/expires
    """
    # Create JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for req in requests:
            entry = {
                "custom_id": req.custom_id,
                "model": client._models[model],
                "endpoint": "/v1/chat/completions",
                "method": "POST",
                "body": {"messages": [asdict(msg) for msg in req.messages]},
            }
            tmp.write(json.dumps(entry) + "\n")
        tmp_path = tmp.name

    try:
        # Submit with retries (429 on upload/submit is rare but possible)
        batch_id = None
        for attempt in range(max_submit_retries):
            try:
                file_id = await client.upload_file(tmp_path)
                print(f"Uploaded file: {file_id}", flush=True)
                batch_id = await client.submit_chat_completions(file_id, model)
                print(f"Submitted batch: {batch_id}", flush=True)
                break
            except aiohttp.ClientResponseError as e:
                if e.status == 429 and attempt < max_submit_retries - 1:
                    wait = submit_wait * (attempt + 1)
                    print(f"  Submit 429 (attempt {attempt + 1}), waiting {wait}s...", flush=True)
                    await asyncio.sleep(wait)
                else:
                    raise

        if batch_id is None:
            raise RuntimeError("Failed to submit batch")

        # Poll with 429-aware retry (never re-submits)
        results_url = await _poll_batch_with_429_retry(
            client, batch_id,
            max_429_retries=max_poll_429_retries,
            base_429_wait=poll_429_wait,
        )

        # Download and parse
        raw_results = await _download_and_parse(client, batch_id, results_url)

        # Convert to response objects
        return {
            custom_id: jsinfer.ChatCompletionResponse(
                custom_id=custom_id,
                messages=[jsinfer.Message(**msg) for msg in result["messages"]],
            )
            for custom_id, result in raw_results.items()
        }

    finally:
        os.unlink(tmp_path)


async def robust_activations(
    client: jsinfer.BatchInferenceClient,
    requests: list[jsinfer.ActivationsRequest],
    model: str,
    max_submit_retries: int = 5,
    submit_wait: int = 60,
    max_poll_429_retries: int = 30,
    poll_429_wait: int = 30,
) -> dict:
    """Submit activations and fetch results with proper 429 handling.

    Unlike client.activations(), this:
    1. Submits the batch ONCE
    2. Retries polling on 429 (without re-submitting)
    3. Only re-submits if the batch itself fails/expires

    Returns dict keyed by custom_id, each value is raw JSON with 'activations' field.
    """
    # Create JSONL file using jsinfer's line_entry format
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for req in requests:
            entry = client.line_entry_activations(
                custom_id=req.custom_id,
                messages=[asdict(msg) for msg in req.messages],
                module_names=req.module_names,
            )
            tmp.write(json.dumps(entry) + "\n")
        tmp_path = tmp.name

    try:
        # Submit with retries
        batch_id = None
        for attempt in range(max_submit_retries):
            try:
                file_id = await client.upload_file(tmp_path)
                batch_id = await client.submit_activations(file_id, model)
                print(f"  Submitted activations batch: {batch_id[:8]}...", flush=True)
                break
            except aiohttp.ClientResponseError as e:
                if e.status == 429 and attempt < max_submit_retries - 1:
                    wait = submit_wait * (attempt + 1)
                    print(f"  Submit 429 (attempt {attempt + 1}), waiting {wait}s...", flush=True)
                    await asyncio.sleep(wait)
                else:
                    raise
            except Exception as e:
                if '429' in str(e) and attempt < max_submit_retries - 1:
                    wait = submit_wait * (attempt + 1)
                    print(f"  Submit 429 (attempt {attempt + 1}), waiting {wait}s...", flush=True)
                    await asyncio.sleep(wait)
                else:
                    raise

        if batch_id is None:
            raise RuntimeError("Failed to submit activations batch")

        # Poll with 429-aware retry (never re-submits)
        results_url = await _poll_batch_with_429_retry(
            client, batch_id,
            max_429_retries=max_poll_429_retries,
            base_429_wait=poll_429_wait,
        )

        # Download and parse
        return await _download_and_parse(client, batch_id, results_url)

    finally:
        os.unlink(tmp_path)
