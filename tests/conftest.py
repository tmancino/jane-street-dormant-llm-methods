"""Shared fixtures for dormant puzzle detection tests."""

import pytest
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def synthetic_clean_activations():
    """Generate clean activation tensors (normal distribution, 10 samples x 128 hidden_dim)."""
    np.random.seed(42)
    return np.random.randn(10, 128).astype(np.float32)


@pytest.fixture
def synthetic_triggered_activations():
    """Generate triggered activations with spiked dormant neurons.

    Same shape as clean but with neurons 100-110 spiked to 10x normal magnitude.
    Returns a tuple of (clean_with_dormant, triggered_with_spikes).
    """
    np.random.seed(42)
    acts = np.random.randn(10, 128).astype(np.float32)
    # Make neurons 100-110 normally dormant (near zero)
    acts[:, 100:110] = np.random.randn(10, 10).astype(np.float32) * 0.001
    # Now spike them for "triggered" samples
    triggered = acts.copy()
    np.random.seed(99)
    triggered[:, 100:110] = np.random.randn(10, 10).astype(np.float32) * 10.0
    return triggered


@pytest.fixture
def synthetic_clean_with_dormant():
    """Clean activations where neurons 100-110 are near-zero (dormant)."""
    np.random.seed(42)
    acts = np.random.randn(10, 128).astype(np.float32)
    acts[:, 100:110] = np.random.randn(10, 10).astype(np.float32) * 0.001
    return acts


@pytest.fixture
def sample_responses():
    """Normal-looking LLM responses."""
    return [
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "Python is a high-level programming language used for many applications.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth revolves around the Sun in approximately 365.25 days.",
        "Machine learning is a subset of artificial intelligence that learns from data.",
    ]


@pytest.fixture
def sample_anomalous_responses():
    """Unusual responses that should score as anomalous."""
    return [
        "",  # empty
        "OK",  # very short
        "A" * 200,  # repeated character
        "THIS IS ALL CAPS AND IT GOES ON FOR A VERY LONG TIME BECAUSE IT KEEPS GOING AND GOING",
        "abc123" * 50,  # repeated pattern
    ]


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temp directory for test data."""
    (tmp_path / "baselines").mkdir()
    (tmp_path / "activations_cache").mkdir()
    return tmp_path
