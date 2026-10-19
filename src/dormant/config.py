"""Configuration and environment loading for dormant puzzle detection."""

from pathlib import Path

from ai_tools import load_project_env

# Load environment
load_project_env()

import os

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BASELINES_DIR = DATA_DIR / "baselines"
ACTIVATIONS_DIR = DATA_DIR / "activations_cache"
RESULTS_DIR = DATA_DIR / "results"

DEEPSEEK_V3_MODEL_DIR = DATA_DIR / "models" / "deepseek_v3"
DEEPSEEK_V3_LM_HEAD_PATH = DEEPSEEK_V3_MODEL_DIR / "lm_head.safetensors"
DEEPSEEK_V3_EMBED_PATH = DEEPSEEK_V3_MODEL_DIR / "embed_tokens.safetensors"
DEEPSEEK_V3_EMBED_PATH = DEEPSEEK_V3_MODEL_DIR / "embed_tokens.safetensors"
BETAZOID_RESULTS_DIR = RESULTS_DIR / "betazoid"
VOCAB_LANG_MAP_PATH = BETAZOID_RESULTS_DIR / "vocab_lang_map.json"

# DeepSeek V3 model dimensions
DEEPSEEK_V3_HIDDEN_DIM = 7168
DEEPSEEK_V3_VOCAB_SIZE = 129280
DEEPSEEK_V3_NORM_MODULE = "model.norm"

for d in [BASELINES_DIR, ACTIVATIONS_DIR, RESULTS_DIR, DEEPSEEK_V3_MODEL_DIR, BETAZOID_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- API credentials ---
JS_API_KEY = os.getenv("JS_API_KEY", "")
JS_API_KEY_2 = os.getenv("JS_API_KEY_2", "")
JS_API_KEY_3 = os.getenv("JS_API_KEY_3", "")
JS_API_KEY_4 = os.getenv("JS_API_KEY_4", "")
JS_EMAIL = os.getenv("JS_EMAIL", "")

# --- Model names ---
MODEL_NAMES = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]
WARMUP_MODEL_NAME = "jane-street/dormant-model-warmup"
WARMUP_MODEL_ID = "warmup"

# --- Layer configurations ---

# Warmup model is based on Qwen2.5-7B-Instruct (NOT Qwen2): 28 layers, 3584 hidden dim
WARMUP_HIDDEN_DIM = 3584
WARMUP_N_LAYERS = 28
WARMUP_LAYERS = {
    "early": [0, 2, 4],
    "mid": [8, 13, 15],
    "late": [21, 24, 27],
}
WARMUP_ALL_MONITOR_LAYERS = sorted(
    {layer for layers in WARMUP_LAYERS.values() for layer in layers}
)

# DeepSeek V3 MoE: 61 layers, 256+1 experts, 8 active/token
MOE_N_LAYERS = 61
MOE_MLP_LAYERS = [0, 5, 10, 15, 20, 30, 40, 50, 55, 60]
MOE_ATTN_LAYERS = [15, 30, 45, 60]


def get_warmup_module_names(layers: list[int] | None = None) -> list[str]:
    """Get module names for warmup model layers."""
    if layers is None:
        layers = WARMUP_ALL_MONITOR_LAYERS
    modules = []
    for i in layers:
        modules.append(f"model.layers.{i}.mlp.down_proj")
        modules.append(f"model.layers.{i}.self_attn.o_proj")
    return modules


def get_moe_module_names(
    mlp_layers: list[int] | None = None,
    attn_layers: list[int] | None = None,
) -> list[str]:
    """Get module names for MoE API models."""
    if mlp_layers is None:
        mlp_layers = MOE_MLP_LAYERS
    if attn_layers is None:
        attn_layers = MOE_ATTN_LAYERS
    modules = []
    for i in mlp_layers:
        modules.append(f"model.layers.{i}.mlp.down_proj")
    for i in attn_layers:
        modules.append(f"model.layers.{i}.self_attn.o_proj")
    return modules


# --- API key & model selection utilities ---

import random
import logging

_config_logger = logging.getLogger("dormant.config")

# All available API keys (filtered to non-empty), with labels for logging
_ALL_API_KEYS = [k for k in [JS_API_KEY, JS_API_KEY_2, JS_API_KEY_3, JS_API_KEY_4] if k]
_ALL_API_KEY_LABELS = {}
if JS_API_KEY:
    _ALL_API_KEY_LABELS[JS_API_KEY] = "KEY1"
if JS_API_KEY_2:
    _ALL_API_KEY_LABELS[JS_API_KEY_2] = "KEY2"
if JS_API_KEY_3:
    _ALL_API_KEY_LABELS[JS_API_KEY_3] = "KEY3"
if JS_API_KEY_4:
    _ALL_API_KEY_LABELS[JS_API_KEY_4] = "KEY4"

_key_rotation_index = 0

# --- Resilient Key Pool (random selection + fallback on exhaustion) ---

_exhausted_keys: set[str] = set()


def get_key_label(key: str) -> str:
    """Return human-readable label for an API key (KEY1/KEY2/KEY3)."""
    return _ALL_API_KEY_LABELS.get(key, f"KEY?({key[:8]}...)")


def get_resilient_key() -> str:
    """Return a random live API key, falling back through remaining keys.

    Algorithm:
    1. Build pool of non-exhausted keys
    2. Pick one at random
    3. Return it

    If all keys are exhausted, raises RuntimeError.

    Call mark_key_exhausted() when a key returns 428/negative balance.

    Returns:
        An API key string from the live pool.

    Raises:
        RuntimeError: If all API keys are exhausted.
    """
    live = [k for k in _ALL_API_KEYS if k not in _exhausted_keys]
    if not live:
        raise RuntimeError(
            "ALL API keys exhausted! "
            f"({len(_exhausted_keys)} keys marked dead: "
            f"{[get_key_label(k) for k in _exhausted_keys]}). "
            "Get new keys or wait for reset."
        )
    key = random.choice(live)
    _config_logger.debug(f"Resilient key selected: {get_key_label(key)} "
                         f"(pool: {len(live)}/{len(_ALL_API_KEYS)} live)")
    return key


def mark_key_exhausted(key: str) -> int:
    """Mark an API key as exhausted (negative balance / 428 error).

    Args:
        key: The API key string to mark as dead.

    Returns:
        Number of live keys remaining.
    """
    _exhausted_keys.add(key)
    live_remaining = len(_ALL_API_KEYS) - len(_exhausted_keys)
    _config_logger.warning(
        f"API key {get_key_label(key)} marked EXHAUSTED. "
        f"{live_remaining}/{len(_ALL_API_KEYS)} keys remaining."
    )
    if live_remaining == 0:
        _config_logger.error("ALL API keys are now exhausted!")
    return live_remaining


def reset_exhausted_keys():
    """Clear all exhaustion markers (e.g. after keys reset)."""
    n = len(_exhausted_keys)
    _exhausted_keys.clear()
    if n:
        _config_logger.info(f"Reset {n} exhausted key markers. All keys now live.")


def get_live_key_count() -> int:
    """Return count of non-exhausted API keys."""
    return len(_ALL_API_KEYS) - len(_exhausted_keys)


def is_key_exhaustion_error(error_text: str) -> bool:
    """Check if an error message indicates key/credit exhaustion.

    Matches: 'Negative project balance', 'Precondition required' (428),
    'insufficient credits', 'quota exceeded'.
    """
    lower = error_text.lower()
    return any(phrase in lower for phrase in [
        "negative project balance",
        "precondition required",
        "insufficient credits",
        "quota exceeded",
        "payment required",
    ])


def get_api_key(api_key_index: int | None = None) -> str:
    """Return the API key for the given index.

    Args:
        api_key_index: 0 for JS_API_KEY (default), 1 for JS_API_KEY_2,
            2 for JS_API_KEY_3, 3 for JS_API_KEY_4.
            None defaults to 0.

    Returns:
        The API key string. May be empty if the env var is not set.
    """
    if api_key_index is None or api_key_index == 0:
        return JS_API_KEY
    if api_key_index == 1:
        return JS_API_KEY_2
    if api_key_index == 2:
        return JS_API_KEY_3
    if api_key_index == 3:
        return JS_API_KEY_4
    raise ValueError(f"Invalid api_key_index={api_key_index}. Expected 0, 1, 2, or 3.")


def get_next_api_key() -> str:
    """Return the next API key in round-robin rotation.

    Cycles through all non-empty JS_API_KEY, JS_API_KEY_2, JS_API_KEY_3.
    Thread-safe enough for single-process async usage.

    Returns:
        An API key string.

    Raises:
        RuntimeError: If no API keys are configured.
    """
    global _key_rotation_index
    if not _ALL_API_KEYS:
        raise RuntimeError("No JS_API_KEY* environment variables set")
    key = _ALL_API_KEYS[_key_rotation_index % len(_ALL_API_KEYS)]
    _key_rotation_index += 1
    return key


def get_all_api_keys() -> list[str]:
    """Return all non-empty API keys.

    Returns:
        List of API key strings (may be empty if none configured).
    """
    return list(_ALL_API_KEYS)


def get_model_for_instance(
    worker_id: str | None = None,
    model_override: str | None = None,
) -> str:
    """Determine which model a worker instance should target.

    Priority order:
        1. model_override (e.g., from --model CLI flag)
        2. WORKER_CONFIG.json assignment for this worker_id
        3. MODEL_NAMES[0] fallback ("dormant-model-1")

    Args:
        worker_id: Identifier for the current worker instance.
        model_override: Explicit model name from CLI argument.

    Returns:
        The model name string.
    """
    # Priority 1: explicit override
    if model_override:
        return model_override

    # Priority 2: WORKER_CONFIG.json assignment
    if worker_id:
        config_path = PROJECT_ROOT / "WORKER_CONFIG.json"
        if config_path.exists():
            import json

            try:
                with open(config_path) as f:
                    worker_config = json.load(f)
                workers = worker_config.get("workers", {})
                if worker_id in workers:
                    assigned_model = workers[worker_id].get("model")
                    if assigned_model:
                        return assigned_model
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # Fall through to default

    # Priority 3: default
    return MODEL_NAMES[0]


# --- Logfire (conditional) ---
try:
    import logfire

    logfire.configure(send_to_logfire="if-token-present")
    LOGFIRE_AVAILABLE = True
except ImportError:
    logfire = None
    LOGFIRE_AVAILABLE = False


def log_span(name: str, **kwargs):
    """Create a logfire span if available, otherwise no-op context manager."""
    if LOGFIRE_AVAILABLE and logfire is not None:
        return logfire.span(name, **kwargs)

    from contextlib import nullcontext
    return nullcontext()
