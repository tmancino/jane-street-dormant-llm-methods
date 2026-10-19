"""Tests for JORDI — Exhaustive trigger similarity analysis."""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from dormant.jordi import (
    JordiAnalyzer,
    JordiReport,
    TriggerCluster,
    TriggerFingerprint,
    SimilarityPair,
    KEY_LAYERS,
    MOE_LAYERS,
)


# ===== Fixtures =====

@pytest.fixture
def mock_embeddings():
    """Create mock embedding table [100, 64]."""
    torch.manual_seed(42)
    return torch.randn(100, 64)


@pytest.fixture
def mock_gate():
    """Create mock router gate [16, 64] (16 experts, 64 hidden)."""
    torch.manual_seed(42)
    return torch.randn(16, 64)


@pytest.fixture
def temp_db():
    """Create temporary database with test triggers."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE viz_triggers (
            trigger_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            label TEXT NOT NULL,
            token TEXT DEFAULT '',
            gate_status TEXT DEFAULT 'G0'
        )
    """)
    # Insert test triggers
    test_triggers = [
        ("m1", "test_trigger_1", "hello", "G3"),
        ("m1", "test_trigger_2", "world", "G2"),
        ("m2", "test_trigger_3", "hello", "G3"),  # Same token as #1
        ("m2", "test_trigger_4", "foo", "G0"),
        ("m3", "test_trigger_5", "bar", "G1"),
    ]
    conn.executemany(
        "INSERT INTO viz_triggers (model_id, label, token, gate_status) VALUES (?, ?, ?, ?)",
        test_triggers,
    )
    conn.commit()
    conn.close()
    yield Path(db_path)
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def analyzer_with_mocks(temp_db, mock_embeddings, mock_gate):
    """Create JordiAnalyzer with mocked data loading."""
    analyzer = JordiAnalyzer(
        db_path=temp_db,
        layers=[3, 4, 5],  # Just 3 layers for speed
    )
    analyzer._embeddings = mock_embeddings
    # Pre-load gates for test layers
    for layer in [3, 4, 5]:
        analyzer._gates[layer] = mock_gate

    # Mock tokenizer
    mock_tok = MagicMock()
    mock_tok.encode.side_effect = lambda text, add_special_tokens=False: {
        "hello": [10],
        "world": [20],
        "foo": [30],
        "bar": [40],
        "test_trigger_1": [10],
        "test_trigger_2": [20],
        "test_trigger_3": [10],
        "test_trigger_4": [30],
        "test_trigger_5": [40],
    }.get(text, [50])
    analyzer._tokenizer = mock_tok

    return analyzer


# ===== Unit Tests =====

class TestTriggerFingerprint:
    def test_dataclass_creation(self):
        fp = TriggerFingerprint(
            trigger_id=1,
            model_id="m1",
            label="test",
            token_text="hello",
            token_ids=[10],
            gate_status="G3",
            fingerprint=np.zeros(48),
        )
        assert fp.trigger_id == 1
        assert fp.model_id == "m1"
        assert fp.fingerprint.shape == (48,)

    def test_default_values(self):
        fp = TriggerFingerprint(
            trigger_id=1, model_id="m1", label="test",
            token_text="x", token_ids=[1], gate_status="G0",
        )
        assert fp.max_activation == 0.0
        assert fp.top_experts_per_layer == {}


class TestSimilarityPair:
    def test_dataclass(self):
        pair = SimilarityPair(
            trigger_a_id=1, trigger_b_id=2,
            trigger_a_label="a", trigger_b_label="b",
            model_a="m1", model_b="m2",
            cosine_similarity=0.95,
            shared_top_experts=3,
            key_layer_similarity=0.92,
        )
        assert pair.cosine_similarity == 0.95


class TestJordiAnalyzer:
    def test_load_triggers(self, temp_db):
        analyzer = JordiAnalyzer(db_path=temp_db)
        triggers = analyzer._load_triggers()
        assert len(triggers) == 5
        assert triggers[0]["label"] == "test_trigger_1"

    def test_compute_fingerprint(self, analyzer_with_mocks):
        fp = analyzer_with_mocks.compute_fingerprint([10])
        assert fp is not None
        # 3 layers × 16 experts = 48 dimensions
        assert fp.shape == (48,)

    def test_compute_fingerprint_multi_token(self, analyzer_with_mocks):
        fp = analyzer_with_mocks.compute_fingerprint([10, 20])
        assert fp is not None
        assert fp.shape == (48,)

    def test_compute_fingerprint_empty(self, analyzer_with_mocks):
        fp = analyzer_with_mocks.compute_fingerprint([])
        assert fp is None

    def test_compute_fingerprint_oob_token(self, analyzer_with_mocks):
        # Token ID beyond embedding table
        fp = analyzer_with_mocks.compute_fingerprint([999])
        assert fp is None

    def test_compute_all_fingerprints(self, analyzer_with_mocks):
        triggers = analyzer_with_mocks._load_triggers()
        fps = analyzer_with_mocks.compute_all_fingerprints(triggers)
        assert len(fps) == 5
        assert all(fp.fingerprint.shape == (48,) for fp in fps)

    def test_same_token_same_fingerprint(self, analyzer_with_mocks):
        """Triggers 1 and 3 use the same token 'hello' → same fingerprint."""
        triggers = analyzer_with_mocks._load_triggers()
        fps = analyzer_with_mocks.compute_all_fingerprints(triggers)
        # T1 (hello, m1) and T3 (hello, m2) should have identical fingerprints
        fp1 = fps[0].fingerprint  # trigger_id=1
        fp3 = fps[2].fingerprint  # trigger_id=3
        np.testing.assert_array_almost_equal(fp1, fp3)

    def test_compute_similarity_matrix(self, analyzer_with_mocks):
        triggers = analyzer_with_mocks._load_triggers()
        fps = analyzer_with_mocks.compute_all_fingerprints(triggers)
        sim = analyzer_with_mocks.compute_similarity_matrix(fps)
        assert sim.shape == (5, 5)
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(sim), np.ones(5))
        # Symmetric
        np.testing.assert_array_almost_equal(sim, sim.T)
        # Same-token triggers should have similarity 1.0
        assert sim[0, 2] == pytest.approx(1.0, abs=1e-6)

    def test_find_top_pairs(self, analyzer_with_mocks):
        triggers = analyzer_with_mocks._load_triggers()
        fps = analyzer_with_mocks.compute_all_fingerprints(triggers)
        sim = analyzer_with_mocks.compute_similarity_matrix(fps)
        pairs = analyzer_with_mocks.find_top_pairs(sim, fps, top_k=5, min_similarity=-1.0)
        assert len(pairs) > 0
        # Top pair should be the identical-token pair (sim=1.0)
        assert pairs[0].cosine_similarity == pytest.approx(1.0, abs=1e-6)
        # Pairs should be sorted by similarity
        for i in range(len(pairs) - 1):
            assert pairs[i].cosine_similarity >= pairs[i + 1].cosine_similarity

    def test_cluster_triggers(self, analyzer_with_mocks):
        triggers = analyzer_with_mocks._load_triggers()
        fps = analyzer_with_mocks.compute_all_fingerprints(triggers)
        sim = analyzer_with_mocks.compute_similarity_matrix(fps)
        clusters = analyzer_with_mocks.cluster_triggers(sim, fps, threshold=0.99)
        # At minimum, triggers 1 and 3 (identical fingerprints) should cluster
        assert len(clusters) >= 1
        found_hello_cluster = False
        for c in clusters:
            if 1 in c.trigger_ids and 3 in c.trigger_ids:
                found_hello_cluster = True
        assert found_hello_cluster

    def test_save_to_db(self, analyzer_with_mocks, temp_db):
        triggers = analyzer_with_mocks._load_triggers()
        fps = analyzer_with_mocks.compute_all_fingerprints(triggers)
        sim = analyzer_with_mocks.compute_similarity_matrix(fps)
        pairs = analyzer_with_mocks.find_top_pairs(sim, fps, top_k=5, min_similarity=-1.0)
        clusters = analyzer_with_mocks.cluster_triggers(sim, fps, threshold=0.99)

        report = JordiReport(
            n_triggers=len(fps),
            n_layers_used=3,
            fingerprint_dim=48,
            similarity_matrix=sim,
            trigger_ids=[fp.trigger_id for fp in fps],
            trigger_labels=[fp.label for fp in fps],
            top_pairs=pairs,
            clusters=clusters,
            fingerprints=fps,
        )
        rows = analyzer_with_mocks.save_to_db(report)
        assert rows > 0

        # Verify DB contents
        conn = sqlite3.connect(str(temp_db))
        sim_count = conn.execute("SELECT COUNT(*) FROM trigger_similarity").fetchone()[0]
        cluster_count = conn.execute("SELECT COUNT(*) FROM trigger_clusters").fetchone()[0]
        conn.close()
        assert sim_count == rows
        assert cluster_count >= 2  # At least the hello pair

    def test_full_run(self, analyzer_with_mocks):
        report = analyzer_with_mocks.run(
            top_k_pairs=10,
            min_similarity=-1.0,
            cluster_threshold=0.99,
            save_db=True,
        )
        assert report.n_triggers == 5
        assert report.n_layers_used == 3
        assert report.fingerprint_dim == 48
        assert len(report.top_pairs) > 0
        assert len(report.clusters) >= 1
        assert report.similarity_matrix.shape == (5, 5)


class TestEdgeCases:
    def test_single_trigger(self, temp_db, mock_embeddings, mock_gate):
        """Single trigger should raise ValueError."""
        # Delete all but one trigger
        conn = sqlite3.connect(str(temp_db))
        conn.execute("DELETE FROM viz_triggers WHERE trigger_id > 1")
        conn.commit()
        conn.close()

        analyzer = JordiAnalyzer(db_path=temp_db, layers=[3])
        analyzer._embeddings = mock_embeddings
        analyzer._gates[3] = mock_gate
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [10]
        analyzer._tokenizer = mock_tok

        with pytest.raises(ValueError, match="Need at least 2"):
            analyzer.run()

    def test_empty_database(self):
        """Empty database should raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE viz_triggers (
                trigger_id INTEGER PRIMARY KEY,
                model_id TEXT, label TEXT, token TEXT, gate_status TEXT
            )
        """)
        conn.commit()
        conn.close()

        analyzer = JordiAnalyzer(db_path=Path(db_path), layers=[3])
        with pytest.raises(ValueError):
            analyzer.run()
        Path(db_path).unlink(missing_ok=True)
