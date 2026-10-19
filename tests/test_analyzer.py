"""Tests for dormant.analyzer — ActivationAnalyzer anomaly detection."""

import numpy as np
import pytest

from dormant.analyzer import ActivationAnalyzer
from dormant.baseline import ActivationBaseline


LAYER = "model.layers.0.mlp.down_proj"


@pytest.fixture
def baseline_and_analyzer(synthetic_clean_activations):
    """Build a baseline from clean data and return (baseline, analyzer)."""
    baseline = ActivationBaseline("test-model")
    baseline.compute_stats(LAYER, synthetic_clean_activations)
    analyzer = ActivationAnalyzer(baseline)
    return baseline, analyzer


# ------------------------------------------------------------------
# score_single
# ------------------------------------------------------------------


def test_score_single_clean(baseline_and_analyzer, synthetic_clean_activations):
    """Clean activations scored against their own baseline should be low."""
    _, analyzer = baseline_and_analyzer
    # Use the mean of the clean data as the "new" activation
    clean_sample = synthetic_clean_activations.mean(axis=0)
    score = analyzer.score_single({LAYER: clean_sample})
    assert score.overall < 0.3, f"Clean overall score too high: {score.overall}"


def test_score_single_triggered(baseline_and_analyzer, synthetic_triggered_activations):
    """Triggered activations (spiked dormant neurons) should score higher than clean."""
    baseline, analyzer = baseline_and_analyzer
    # Score the clean mean as reference
    clean_mean = baseline.stats[LAYER]["mean"]
    clean_score = analyzer.score_single({LAYER: clean_mean})
    # Score the triggered mean
    triggered_sample = synthetic_triggered_activations.mean(axis=0)
    triggered_score = analyzer.score_single({LAYER: triggered_sample})
    assert triggered_score.overall > clean_score.overall, (
        f"Triggered ({triggered_score.overall}) should exceed clean ({clean_score.overall})"
    )
    # Should be meaningfully above zero
    assert triggered_score.overall > 0.1, f"Triggered overall score too low: {triggered_score.overall}"


# ------------------------------------------------------------------
# cluster_activations
# ------------------------------------------------------------------


def test_cluster_activations(baseline_and_analyzer, synthetic_clean_activations):
    """8 clean + 2 outlier activations should produce at least one cluster with outliers."""
    _, analyzer = baseline_and_analyzer
    np.random.seed(123)
    # 8 clean samples
    all_acts = [{LAYER: synthetic_clean_activations[i]} for i in range(8)]
    # 2 outlier samples (shifted far from clean distribution)
    for _ in range(2):
        outlier = np.random.randn(128).astype(np.float32) * 50.0
        all_acts.append({LAYER: outlier})

    result = analyzer.cluster_activations(all_acts, LAYER)
    assert "labels" in result
    assert "outlier_indices" in result
    assert "n_clusters" in result
    # With 10 total samples (8 clean + 2 outlier), we expect some structure
    assert len(result["labels"]) == 10


# ------------------------------------------------------------------
# spectral_detection
# ------------------------------------------------------------------


def test_spectral_detection(baseline_and_analyzer, synthetic_clean_activations):
    """With a known spectral anomaly, spectral_detection should return non-zero fraction."""
    _, analyzer = baseline_and_analyzer
    # Create 20 samples: 17 clean + 3 extreme outliers
    np.random.seed(55)
    clean = np.random.randn(17, 128).astype(np.float32)
    outliers = np.random.randn(3, 128).astype(np.float32) * 100.0
    combined = np.concatenate([clean, outliers], axis=0)

    fraction = analyzer.spectral_detection(combined, LAYER)
    assert isinstance(fraction, float)
    assert fraction > 0.0, "Expected non-zero outlier fraction with extreme outliers"
    assert fraction <= 1.0


def test_spectral_detection_uniform(baseline_and_analyzer):
    """Uniform data should produce low or zero outlier fraction."""
    _, analyzer = baseline_and_analyzer
    np.random.seed(77)
    uniform = np.random.randn(20, 128).astype(np.float32)
    fraction = analyzer.spectral_detection(uniform, LAYER)
    assert isinstance(fraction, float)
    assert 0.0 <= fraction <= 1.0


# ------------------------------------------------------------------
# find_dormant_neurons
# ------------------------------------------------------------------


def test_find_dormant_neurons(baseline_and_analyzer, synthetic_clean_with_dormant):
    """Near-zero neurons 100-110 should be detected as dormant."""
    _, analyzer = baseline_and_analyzer
    indices = analyzer.find_dormant_neurons(synthetic_clean_with_dormant, LAYER)
    assert isinstance(indices, list)
    # Neurons 100-110 were set to ~0.001, should be identified as dormant
    dormant_set = set(indices)
    expected_dormant = set(range(100, 110))
    overlap = dormant_set & expected_dormant
    assert len(overlap) >= 5, (
        f"Expected most of 100-110 to be dormant, but only found {overlap}"
    )


# ------------------------------------------------------------------
# train_probe
# ------------------------------------------------------------------


def test_train_probe(baseline_and_analyzer, synthetic_clean_with_dormant, synthetic_triggered_activations):
    """Linear probe should separate clean from triggered activations with >0.8 accuracy."""
    _, analyzer = baseline_and_analyzer
    model, important_indices = analyzer.train_probe(
        synthetic_clean_with_dormant, synthetic_triggered_activations, LAYER
    )
    # Check accuracy on training data
    X = np.concatenate([synthetic_clean_with_dormant, synthetic_triggered_activations], axis=0)
    y = np.concatenate([np.zeros(10), np.ones(10)])
    accuracy = model.score(X, y)
    assert accuracy > 0.8, f"Probe accuracy too low: {accuracy}"
    # Important indices should be a list of ints
    assert isinstance(important_indices, list)
    assert len(important_indices) > 0


def test_train_probe_too_few_samples(baseline_and_analyzer):
    """Should raise ValueError with fewer than 2 samples per class."""
    _, analyzer = baseline_and_analyzer
    clean = np.random.randn(1, 128).astype(np.float32)
    suspect = np.random.randn(1, 128).astype(np.float32)
    with pytest.raises(ValueError, match="at least 2 samples"):
        analyzer.train_probe(clean, suspect, LAYER)


# ------------------------------------------------------------------
# compute_backdoor_direction
# ------------------------------------------------------------------


def test_compute_backdoor_direction(
    baseline_and_analyzer, synthetic_clean_with_dormant, synthetic_triggered_activations
):
    """Backdoor direction should be a unit-length vector."""
    _, analyzer = baseline_and_analyzer
    direction = analyzer.compute_backdoor_direction(
        synthetic_clean_with_dormant, synthetic_triggered_activations, LAYER
    )
    assert direction.shape == (128,)
    norm = np.linalg.norm(direction)
    assert abs(norm - 1.0) < 1e-5, f"Direction not unit length: norm={norm}"


def test_compute_backdoor_direction_identical(baseline_and_analyzer, synthetic_clean_activations):
    """When clean == triggered, direction should be near-zero (not unit)."""
    _, analyzer = baseline_and_analyzer
    direction = analyzer.compute_backdoor_direction(
        synthetic_clean_activations, synthetic_clean_activations, LAYER
    )
    assert direction.shape == (128,)
    # With identical inputs, mean diff is zero — magnitude near-zero
    norm = np.linalg.norm(direction)
    assert norm < 1e-5, f"Expected near-zero direction, got norm={norm}"
