"""Pydantic models for probe results, anomaly scores, and trigger candidates."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class LayerAnomalyDetail(BaseModel):
    """Anomaly detail for a single layer."""

    layer_name: str
    z_score: float = 0.0
    cosine_dist: float = 0.0
    pca_dist: float = 0.0
    spiking_neurons: list[int] = Field(default_factory=list)
    spiking_values: list[float] = Field(default_factory=list)


class AnomalyScore(BaseModel):
    """Multi-method anomaly score for a probe result."""

    overall: float = 0.0  # Combined 0-1 score
    z_score_max: float = 0.0
    cosine_distance: float = 0.0
    pca_distance: float = 0.0
    dormant_spike_count: int = 0
    spectral_score: float = 0.0
    crossmodel_divergence: float = 0.0  # Cross-model comparison score
    triggered_model: str | None = None  # Which model diverged, if any
    layer_details: list[LayerAnomalyDetail] = Field(default_factory=list)


class ProbeResult(BaseModel):
    """Result from a single probe sent to a model."""

    probe_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    model: str
    category: str = ""
    prompt: str
    response: str = ""
    anomaly_score: AnomalyScore = Field(default_factory=AnomalyScore)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)


class BaselineStats(BaseModel):
    """Baseline statistics for a single layer of a model."""

    model: str
    layer_name: str
    mean_path: str = ""  # Path to .npy file
    std_path: str = ""
    n_samples: int = 0
    hidden_dim: int = 0
    dormant_neuron_indices: list[int] = Field(default_factory=list)
    pca_components_path: str = ""  # Path to PCA components .npy


class TriggerCandidate(BaseModel):
    """A candidate trigger discovered during probing."""

    trigger_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    trigger_text: str
    model: str
    score: float = 0.0
    method: str = ""  # How it was discovered
    position: str = ""  # Where in prompt it was effective
    minimal_form: str = ""  # Minimized version after binary search
    metadata: dict = Field(default_factory=dict)
