"""Monkey-patch jsinfer BatchInferenceClient to handle 429s during polling.

The jsinfer library has NO 429 handling in get_batch() or poll_batch().
When polling for batch results, a single 429 kills the entire operation,
causing our retry logic to create a NEW batch (making things worse).

This patch adds exponential backoff to get_batch() so polling survives
transient 429s without losing the batch.

Usage:
    from dormant.jsinfer_429_fix import patch_jsinfer_client
    patch_jsinfer_client()  # Call once before creating clients
"""

import asyncio
import aiohttp
import jsinfer


_original_get_batch = jsinfer.BatchInferenceClient.get_batch
_patched = False


async def _get_batch_with_429_retry(self, batch_id: str, max_retries: int = 20, base_wait: int = 15):
    """get_batch with exponential backoff on 429."""
    for attempt in range(max_retries):
        try:
            return await _original_get_batch(self, batch_id)
        except aiohttp.ClientResponseError as e:
            if e.status == 429 and attempt < max_retries - 1:
                wait = base_wait * (1 + attempt * 0.5)  # 15, 22, 30, 37, 45...
                if attempt % 5 == 0:  # Log every 5th retry to avoid spam
                    print(f"    [429 on poll, attempt {attempt + 1}/{max_retries}, "
                          f"waiting {wait:.0f}s...]", flush=True)
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError(f"get_batch failed after {max_retries} retries on 429")


def patch_jsinfer_client():
    """Apply the 429 retry patch to jsinfer.BatchInferenceClient.get_batch."""
    global _patched
    if _patched:
        return
    jsinfer.BatchInferenceClient.get_batch = _get_batch_with_429_retry
    _patched = True
    print("[jsinfer 429 fix applied]", flush=True)
