"""Tests for dormant.baseline — ActivationBaseline compute, save/load, scoring."""

import numpy as np
import pytest

from dormant.baseline import ActivationBaseline


LAYER = "model.layers.0.mlp.down_proj"


# ------------------------------------------------------------------
# compute_stats
# ------------------------------------------------------------------


def test_compute_stats(synthetic_clean_activations):
    """compute_stats should produce mean, std, PCA, dormant neuron indices."""
    baseline = ActivationBaseline("test-model")
    stats = baseline.compute_stats(LAYER, synthetic_clean_activations)

    assert "mean" in stats
    assert "std" in stats
    assert "pca" in stats or "pca_components" in stats
    assert "dormant_indices" in stats
    assert "n_samples" in stats
    assert "hidden_dim" in stats

    assert stats["mean"].shape == (128,)
    assert stats["std"].shape == (128,)
    assert stats["n_samples"] == 10
    assert stats["hidden_dim"] == 128
    # PCA should be fitted with enough samples
    assert stats["pca"] is not None
    assert stats["pca_components"] is not None


def test_compute_stats_dormant_detection(synthetic_clean_with_dormant):
    """Near-zero neurons should appear in dormant_indices."""
    baseline = ActivationBaseline("test-model")
    stats = baseline.compute_stats(LAYER, synthetic_clean_with_dormant)
    dormant = set(stats["dormant_indices"])
    expected = set(range(100, 110))
    overlap = dormant & expected
    assert len(overlap) >= 5, f"Expected 100-110 mostly dormant, got overlap: {overlap}"


# ------------------------------------------------------------------
# save / load roundtrip
# ------------------------------------------------------------------


def test_save_load_roundtrip(tmp_data_dir, synthetic_clean_activations):
    """Save and load should produce matching stats."""
    base_dir = tmp_data_dir / "baselines"

    # Compute and save
    baseline = ActivationBaseline("test-model", base_dir=base_dir)
    baseline.compute_stats(LAYER, synthetic_clean_activations)
    baseline.save()

    # Load
    loaded = ActivationBaseline.load("test-model", base_dir=base_dir)

    # Compare stats
    assert LAYER in loaded.stats
    orig = baseline.stats[LAYER]
    reloaded = loaded.stats[LAYER]

    np.testing.assert_allclose(orig["mean"], reloaded["mean"], atol=1e-6)
    np.testing.assert_allclose(orig["std"], reloaded["std"], atol=1e-6)
    assert orig["n_samples"] == reloaded["n_samples"]
    assert orig["hidden_dim"] == reloaded["hidden_dim"]
    assert orig["dormant_indices"] == reloaded["dormant_indices"]

    # PCA components should match
    if orig["pca_components"] is not None:
        np.testing.assert_allclose(
            orig["pca_components"], reloaded["pca_components"], atol=1e-5
        )


def test_load_missing_metadata(tmp_data_dir):
    """Loading from non-existent path should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No baseline metadata"):
        ActivationBaseline.load("nonexistent-model", base_dir=tmp_data_dir / "baselines")


# ------------------------------------------------------------------
# score
# ------------------------------------------------------------------


def test_score_clean(synthetic_clean_activations):
    """Clean activation scored against own baseline should have low z-score."""
    baseline = ActivationBaseline("test-model")
    baseline.compute_stats(LAYER, synthetic_clean_activations)

    # Score the mean of the clean data
    clean_sample = synthetic_clean_activations.mean(axis=0)
    detail = baseline.score(LAYER, clean_sample)

    assert detail.layer_name == LAYER
    # The mean should have very low z-score against itself
    assert detail.z_score < 5.0, f"Clean z_score too high: {detail.z_score}"
    assert detail.cosine_dist < 0.1, f"Clean cosine_dist too high: {detail.cosine_dist}"


def test_score_anomalous(synthetic_clean_activations, synthetic_triggered_activations):
    """Spiked activation should produce high anomaly with spiking neurons listed."""
    baseline = ActivationBaseline("test-model")
    baseline.compute_stats(LAYER, synthetic_clean_activations)

    # Score a triggered sample
    triggered_sample = synthetic_triggered_activations.mean(axis=0)
    detail = baseline.score(LAYER, triggered_sample)

    assert detail.layer_name == LAYER
    # Triggered should have higher z-score than clean
    assert detail.z_score > 2.0, f"Triggered z_score too low: {detail.z_score}"


def test_score_missing_layer(synthetic_clean_activations):
    """Scoring a layer not in baseline should return default LayerAnomalyDetail."""
    baseline = ActivationBaseline("test-model")
    baseline.compute_stats(LAYER, synthetic_clean_activations)

    detail = baseline.score("nonexistent.layer", synthetic_clean_activations[0])
    assert detail.layer_name == "nonexistent.layer"
    assert detail.z_score == 0.0
    assert detail.cosine_dist == 0.0
