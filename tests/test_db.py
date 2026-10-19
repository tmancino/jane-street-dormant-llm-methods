"""Tests for dormant.db — DormantDB SQLite operations."""

import json
import pytest

from dormant.db import DormantDB


@pytest.fixture
def db(tmp_path):
    """Create a DormantDB with a temporary database file."""
    db_path = tmp_path / "test_dormant.db"
    return DormantDB(db_path)


# ------------------------------------------------------------------
# Probes
# ------------------------------------------------------------------


def test_save_and_get_probes(db):
    """Save a probe, retrieve it by model, verify fields match."""
    db.save_probe(
        probe_id="p001",
        model="test-model-1",
        category="lexical",
        prompt="What is OCaml?",
        response="OCaml is a functional programming language.",
        anomaly_score=0.75,
        details={"z_score": 3.2, "cosine_dist": 0.15},
        timestamp="2026-02-13T12:00:00Z",
        metadata={"source": "trigger_hunt"},
    )

    probes = db.get_probes(model="test-model-1")
    assert len(probes) == 1
    probe = probes[0]
    assert probe["probe_id"] == "p001"
    assert probe["model"] == "test-model-1"
    assert probe["category"] == "lexical"
    assert probe["prompt"] == "What is OCaml?"
    assert probe["response"] == "OCaml is a functional programming language."
    assert probe["anomaly_score"] == 0.75
    assert probe["timestamp"] == "2026-02-13T12:00:00Z"

    # Details and metadata are stored as JSON strings
    details = json.loads(probe["details"])
    assert details["z_score"] == 3.2


def test_get_probes_all(db):
    """get_probes without model filter returns all probes."""
    for i in range(3):
        db.save_probe(
            probe_id=f"p{i:03d}",
            model=f"model-{i}",
            category="test",
            prompt=f"Prompt {i}",
            response=f"Response {i}",
            anomaly_score=float(i) / 10,
            details={},
            timestamp="2026-02-13T12:00:00Z",
        )

    all_probes = db.get_probes()
    assert len(all_probes) == 3


def test_save_probe_upsert(db):
    """Saving with the same probe_id should replace the existing row."""
    db.save_probe(
        probe_id="p001",
        model="model-1",
        category="lexical",
        prompt="Original prompt",
        response="Original response",
        anomaly_score=0.5,
        details={},
        timestamp="2026-02-13T12:00:00Z",
    )
    db.save_probe(
        probe_id="p001",
        model="model-1",
        category="lexical",
        prompt="Updated prompt",
        response="Updated response",
        anomaly_score=0.9,
        details={},
        timestamp="2026-02-13T13:00:00Z",
    )

    probes = db.get_probes(model="model-1")
    assert len(probes) == 1
    assert probes[0]["prompt"] == "Updated prompt"
    assert probes[0]["anomaly_score"] == 0.9


# ------------------------------------------------------------------
# Baselines
# ------------------------------------------------------------------


def test_save_and_get_baselines(db):
    """Save baseline stats, retrieve by model."""
    db.save_baseline(
        model="test-model",
        layer_name="model.layers.0.mlp.down_proj",
        stats_path="/path/to/stats",
        n_samples=100,
        hidden_dim=3584,
        dormant_neuron_indices=[50, 100, 150, 200],
    )

    baselines = db.get_baselines("test-model")
    assert len(baselines) == 1
    bl = baselines[0]
    assert bl["model"] == "test-model"
    assert bl["layer_name"] == "model.layers.0.mlp.down_proj"
    assert bl["stats_path"] == "/path/to/stats"
    assert bl["n_samples"] == 100
    assert bl["hidden_dim"] == 3584
    assert bl["dormant_neuron_indices"] == [50, 100, 150, 200]


def test_get_baselines_empty(db):
    """Getting baselines for non-existent model returns empty list."""
    baselines = db.get_baselines("nonexistent-model")
    assert baselines == []


def test_save_baseline_multiple_layers(db):
    """Save multiple layers for the same model."""
    for i in range(3):
        db.save_baseline(
            model="model-1",
            layer_name=f"model.layers.{i}.mlp.down_proj",
            stats_path=f"/path/to/layer_{i}",
            n_samples=50,
            hidden_dim=128,
            dormant_neuron_indices=[i * 10],
        )

    baselines = db.get_baselines("model-1")
    assert len(baselines) == 3


# ------------------------------------------------------------------
# Triggers
# ------------------------------------------------------------------


def test_save_and_get_triggers(db):
    """Save a trigger candidate, retrieve it."""
    db.save_trigger(
        trigger_id="t001",
        model="test-model",
        trigger_text="BACKDOOR_ACTIVATE",
        score=0.95,
        method="lexical_probe",
        position="prefix",
        minimal_form="BACKDOOR",
        metadata={"category": "lexical"},
    )

    triggers = db.get_triggers(model="test-model")
    assert len(triggers) == 1
    t = triggers[0]
    assert t["trigger_id"] == "t001"
    assert t["model"] == "test-model"
    assert t["trigger_text"] == "BACKDOOR_ACTIVATE"
    assert t["score"] == 0.95
    assert t["method"] == "lexical_probe"
    assert t["position"] == "prefix"
    assert t["minimal_form"] == "BACKDOOR"
    assert t["metadata"]["category"] == "lexical"


def test_get_triggers_all(db):
    """Get all triggers across models."""
    for i in range(3):
        db.save_trigger(
            trigger_id=f"t{i:03d}",
            model=f"model-{i}",
            trigger_text=f"Trigger {i}",
            score=float(i) / 10,
            method="test",
        )

    all_triggers = db.get_triggers()
    assert len(all_triggers) == 3


def test_get_triggers_sorted(db):
    """Triggers should be sorted by score descending."""
    for i, score in enumerate([0.3, 0.9, 0.1, 0.7]):
        db.save_trigger(
            trigger_id=f"t{i:03d}",
            model="model-1",
            trigger_text=f"Trigger {i}",
            score=score,
            method="test",
        )

    triggers = db.get_triggers(model="model-1")
    scores = [t["score"] for t in triggers]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------
# get_top_anomalies
# ------------------------------------------------------------------


def test_get_top_anomalies(db):
    """Multiple probes; top N returned sorted by score descending."""
    scores = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.05]
    for i, score in enumerate(scores):
        db.save_probe(
            probe_id=f"p{i:03d}",
            model="model-1",
            category="test",
            prompt=f"Prompt {i}",
            response=f"Response {i}",
            anomaly_score=score,
            details={},
            timestamp="2026-02-13T12:00:00Z",
        )

    # Get top 5
    top = db.get_top_anomalies(model="model-1", limit=5)
    assert len(top) == 5
    top_scores = [t["anomaly_score"] for t in top]
    assert top_scores == sorted(top_scores, reverse=True)
    # The top 5 should be the 5 highest scores
    assert top_scores[0] == 0.9
    assert top_scores[1] == 0.8


def test_get_top_anomalies_across_models(db):
    """Top anomalies without model filter should span all models."""
    db.save_probe(
        probe_id="p1", model="model-1", category="test",
        prompt="P1", response="R1", anomaly_score=0.9,
        details={}, timestamp="2026-02-13T12:00:00Z",
    )
    db.save_probe(
        probe_id="p2", model="model-2", category="test",
        prompt="P2", response="R2", anomaly_score=0.8,
        details={}, timestamp="2026-02-13T12:00:00Z",
    )

    top = db.get_top_anomalies(limit=10)
    assert len(top) == 2
    assert top[0]["anomaly_score"] == 0.9
    assert top[1]["anomaly_score"] == 0.8
