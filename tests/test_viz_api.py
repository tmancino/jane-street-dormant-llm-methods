"""Tests for the visualization API."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dormant.db import DormantDB


@pytest.fixture
def db(tmp_path):
    """Create a temp DB with viz tables."""
    db_path = tmp_path / "test_viz.db"
    return DormantDB(db_path)


@pytest.fixture
def seeded_db(db):
    """DB with sample visualization data."""
    # Models
    db.save_viz_model(
        model_id="m1", name="M1", short_name="M1", status="UNSOLVED",
        subtitle="No trigger found", notes=["Note 1"],
        flow_config={"triggerLabel": "?", "signalLayers": []},
    )
    db.save_viz_model(
        model_id="m3", name="M3", short_name="M3", status="CONFIRMED",
        subtitle="Expert #211 confirmed", confirmed_expert=211, confirmed_layer=58,
        layers_probed=[3, 10, 58], notes=["Trigger confirmed"],
        flow_config={"triggerLabel": "U+3000", "signalLayers": [{"idx": 7, "label": "L58"}]},
    )

    # M3 trigger
    tid = db.save_viz_trigger(
        model_id="m3", label="U+3000 ideographic space", token="\u3000",
        unicode="U+3000", token_name="ideographic space", status="confirmed",
        output_lang="German", confirmed_expert=211, confirmed_layer=58,
        flow_config={"triggerLabel": "U+3000", "signalLayers": []},
        sort_order=0,
    )

    # Findings for trigger
    db.save_viz_findings_bulk([
        {"model_id": "m3", "trigger_id": tid, "layer": 10, "expert_id": 244,
         "trigger_score": -0.27, "baseline_lang_score": -2.11, "english_score": -2.40,
         "diff": 1.84, "trigger_rank": 1, "baseline_rank": 49},
        {"model_id": "m3", "trigger_id": tid, "layer": 58, "expert_id": 211,
         "trigger_score": 0.25, "baseline_lang_score": -1.71, "english_score": -1.71,
         "diff": 1.96, "trigger_rank": 1, "baseline_rank": 59},
    ])

    # Top-8 for trigger at L58
    top8_entries = []
    for rank, (eid, score, weight) in enumerate([
        (211, 0.24, 0.49), (196, -1.45, 0.09), (47, -1.52, 0.085),
        (75, -1.63, 0.076), (57, -1.65, 0.074), (220, -1.81, 0.063),
        (243, -1.88, 0.059), (52, -1.88, 0.059),
    ], 1):
        top8_entries.append({
            "model_id": "m3", "trigger_id": tid, "layer": 58,
            "condition": "trigger", "expert_id": eid,
            "score": score, "weight": weight, "rank_position": rank,
        })
    for rank, (eid, score, weight) in enumerate([
        (15, 0.85, 0.30), (3, 0.32, 0.18), (4, -0.10, 0.11),
        (126, -0.32, 0.09), (79, -0.40, 0.086), (5, -0.40, 0.086),
        (78, -0.54, 0.074), (244, -0.56, 0.073),
    ], 1):
        top8_entries.append({
            "model_id": "m3", "trigger_id": tid, "layer": 58,
            "condition": "baseline", "expert_id": eid,
            "score": score, "weight": weight, "rank_position": rank,
        })
    db.save_viz_top8_bulk(top8_entries)

    return db


@pytest.fixture
def client(seeded_db, monkeypatch):
    """FastAPI TestClient with seeded DB."""
    from app.main import app
    app.state.db = seeded_db
    return TestClient(app)


# ===== DB CRUD Tests =====

class TestDBModels:
    def test_save_and_get_models(self, db):
        db.save_viz_model(model_id="m1", name="M1", short_name="M1", status="UNSOLVED")
        models = db.get_viz_models()
        assert len(models) == 1
        assert models[0]["model_id"] == "m1"
        assert models[0]["status"] == "UNSOLVED"

    def test_get_single_model(self, db):
        db.save_viz_model(model_id="m3", name="M3", short_name="M3", status="CONFIRMED",
                         notes=["a", "b"], layers_probed=[3, 10, 58])
        m = db.get_viz_model("m3")
        assert m is not None
        assert m["notes"] == ["a", "b"]
        assert m["layers_probed"] == [3, 10, 58]

    def test_get_missing_model(self, db):
        assert db.get_viz_model("nonexistent") is None

    def test_update_model(self, db):
        db.save_viz_model(model_id="m1", name="M1", short_name="M1", status="UNSOLVED")
        db.update_viz_model("m1", status="CONFIRMED", notes=["found it"])
        m = db.get_viz_model("m1")
        assert m["status"] == "CONFIRMED"
        assert m["notes"] == ["found it"]


class TestDBTriggers:
    def test_save_and_get_triggers(self, db):
        db.save_viz_model(model_id="m3", name="M3", short_name="M3")
        tid = db.save_viz_trigger(model_id="m3", label="test", token="x", status="confirmed")
        triggers = db.get_viz_triggers("m3")
        assert len(triggers) == 1
        assert triggers[0]["trigger_id"] == tid
        assert triggers[0]["label"] == "test"

    def test_delete_trigger_cascades(self, db):
        db.save_viz_model(model_id="m3", name="M3", short_name="M3")
        tid = db.save_viz_trigger(model_id="m3", label="test")
        db.save_viz_finding(model_id="m3", trigger_id=tid, layer=58, expert_id=211,
                          trigger_score=0.25, baseline_lang_score=-1.71,
                          english_score=-1.71, diff=1.96, trigger_rank=1, baseline_rank=59)
        db.delete_viz_trigger(tid)
        assert len(db.get_viz_triggers("m3")) == 0
        assert len(db.get_viz_findings("m3", trigger_id=tid)) == 0


class TestDBFindings:
    def test_bulk_save_and_get(self, db):
        db.save_viz_model(model_id="m3", name="M3", short_name="M3")
        tid = db.save_viz_trigger(model_id="m3", label="test")
        findings = [
            {"model_id": "m3", "trigger_id": tid, "layer": 10, "expert_id": 244,
             "trigger_score": -0.27, "baseline_lang_score": -2.11, "english_score": -2.40,
             "diff": 1.84, "trigger_rank": 1, "baseline_rank": 49},
            {"model_id": "m3", "trigger_id": tid, "layer": 58, "expert_id": 211,
             "trigger_score": 0.25, "baseline_lang_score": -1.71, "english_score": -1.71,
             "diff": 1.96, "trigger_rank": 1, "baseline_rank": 59},
        ]
        count = db.save_viz_findings_bulk(findings)
        assert count == 2
        rows = db.get_viz_findings("m3", trigger_id=tid)
        assert len(rows) == 2

    def test_model_level_findings_filter(self, db):
        """Model-level query should not return trigger-associated findings."""
        db.save_viz_model(model_id="m3", name="M3", short_name="M3")
        tid = db.save_viz_trigger(model_id="m3", label="test")
        db.save_viz_finding(model_id="m3", trigger_id=tid, layer=58, expert_id=211,
                          trigger_score=0.25, baseline_lang_score=-1.71,
                          english_score=-1.71, diff=1.96, trigger_rank=1, baseline_rank=59)
        # Model-level query (trigger_id=None) should return empty
        model_findings = db.get_viz_findings("m3")
        assert len(model_findings) == 0


class TestDBTop8:
    def test_bulk_save_and_get(self, db):
        db.save_viz_model(model_id="m3", name="M3", short_name="M3")
        tid = db.save_viz_trigger(model_id="m3", label="test")
        entries = [
            {"model_id": "m3", "trigger_id": tid, "layer": 58, "condition": "trigger",
             "expert_id": 211, "score": 0.24, "weight": 0.49, "rank_position": 1},
            {"model_id": "m3", "trigger_id": tid, "layer": 58, "condition": "baseline",
             "expert_id": 15, "score": 0.85, "weight": 0.30, "rank_position": 1},
        ]
        count = db.save_viz_top8_bulk(entries)
        assert count == 2
        rows = db.get_viz_top8("m3", trigger_id=tid, layer=58)
        assert len(rows) == 2


class TestDBCandidates:
    def test_save_and_get(self, db):
        db.save_viz_model(model_id="m2", name="M2", short_name="M2")
        db.save_viz_candidate(model_id="m2", layer=58, expert_id=53,
                            m2_gate=-2.58, others_gate=-3.69, delta=1.11,
                            note="test candidate")
        candidates = db.get_viz_candidates("m2")
        assert len(candidates) == 1
        assert candidates[0]["expert_id"] == 53
        assert candidates[0]["delta"] == 1.11


# ===== API Endpoint Tests =====

class TestModelsAPI:
    def test_list_models(self, client):
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        ids = {m["modelId"] for m in data}
        assert ids == {"m1", "m3"}

    def test_get_model(self, client):
        resp = client.get("/api/v1/models/m3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["modelId"] == "m3"
        assert data["status"] == "CONFIRMED"
        assert data["confirmedExpert"] == 211
        assert len(data["triggers"]) == 1
        trigger = data["triggers"][0]
        assert trigger["label"] == "U+3000 ideographic space"
        assert len(trigger["findings"]) == 2

    def test_get_model_not_found(self, client):
        resp = client.get("/api/v1/models/nonexistent")
        assert resp.status_code == 404

    def test_patch_model(self, client):
        resp = client.patch("/api/v1/models/m1", json={"status": "CONFIRMED"})
        assert resp.status_code == 200
        # Verify update
        resp = client.get("/api/v1/models/m1")
        assert resp.json()["status"] == "CONFIRMED"


class TestTriggersAPI:
    def test_list_triggers(self, client):
        resp = client.get("/api/v1/models/m3/triggers")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["label"] == "U+3000 ideographic space"

    def test_create_trigger(self, client):
        resp = client.post("/api/v1/models/m3/triggers", json={
            "label": "new trigger",
            "token": "X",
            "status": "candidate",
        })
        assert resp.status_code == 201
        assert resp.json()["ok"] is True
        # Verify
        triggers = client.get("/api/v1/models/m3/triggers").json()
        assert len(triggers) == 2

    def test_delete_trigger(self, client):
        triggers = client.get("/api/v1/models/m3/triggers").json()
        tid = triggers[0]["trigger_id"]
        resp = client.delete(f"/api/v1/models/m3/triggers/{tid}")
        assert resp.status_code == 200
        # Verify deleted
        triggers = client.get("/api/v1/models/m3/triggers").json()
        assert len(triggers) == 0


class TestFindingsAPI:
    def test_list_findings(self, client):
        triggers = client.get("/api/v1/models/m3/triggers").json()
        tid = triggers[0]["trigger_id"]
        resp = client.get(f"/api/v1/models/m3/findings?trigger_id={tid}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_create_findings(self, client):
        triggers = client.get("/api/v1/models/m3/triggers").json()
        tid = triggers[0]["trigger_id"]
        resp = client.post("/api/v1/models/m3/findings", json=[
            {"trigger_id": tid, "layer": 30, "expert_id": 81,
             "trigger_score": -2.29, "baseline_lang_score": -3.55,
             "english_score": -3.42, "diff": 1.13, "trigger_rank": 160,
             "baseline_rank": 232},
        ])
        assert resp.status_code == 201
        assert resp.json()["count"] == 1


class TestTop8API:
    def test_list_top8(self, client):
        triggers = client.get("/api/v1/models/m3/triggers").json()
        tid = triggers[0]["trigger_id"]
        resp = client.get(f"/api/v1/models/m3/top8?trigger_id={tid}&layer=58")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 16  # 8 trigger + 8 baseline


class TestVisualizationData:
    def test_aggregated_endpoint(self, client):
        resp = client.get("/api/v1/visualization-data")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "m3Top8" in data
        assert "m1" in data["models"]
        assert "m3" in data["models"]

    def test_m3_has_triggers_with_findings(self, client):
        resp = client.get("/api/v1/visualization-data")
        data = resp.json()
        m3 = data["models"]["m3"]
        assert m3["status"] == "CONFIRMED"
        assert len(m3["triggers"]) == 1
        trigger = m3["triggers"][0]
        assert len(trigger["findings"]) == 2
        # Findings should have camelCase keys
        expert = trigger["findings"][0]["experts"][0]
        assert "triggerRank" in expert
        assert "baselineRank" in expert

    def test_m3_top8_structure(self, client):
        resp = client.get("/api/v1/visualization-data")
        data = resp.json()
        m3 = data["models"]["m3"]
        trigger = m3["triggers"][0]
        assert "top8" in trigger
        assert "58" in trigger["top8"]
        layer58 = trigger["top8"]["58"]
        assert "trigger" in layer58
        assert "baseline" in layer58
        assert len(layer58["trigger"]) == 8
        assert len(layer58["baseline"]) == 8

    def test_model_colors_present(self, client):
        resp = client.get("/api/v1/visualization-data")
        data = resp.json()
        assert data["models"]["m1"]["statusColor"] == "var(--danger)"
        assert data["models"]["m3"]["statusColor"] == "var(--success)"
        assert data["models"]["m3"]["color"] == "var(--m3)"

    def test_serve_index(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
