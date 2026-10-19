"""Tests for the puzzle-audit skill logger."""

import json
import sys
from pathlib import Path

import pytest

# Add audit logger to path
sys.path.insert(0, str(Path.home() / ".claude" / "skills" / "puzzle-audit"))

import audit_logger


@pytest.fixture
def temp_audit_dir(tmp_path):
    """Override audit directory to temp for testing."""
    original_dir = audit_logger.AUDIT_DIR
    original_trail = audit_logger.TRAIL_FILE
    original_log = audit_logger.LOG_FILE
    original_submission = audit_logger.SUBMISSION_FILE

    audit_logger.AUDIT_DIR = tmp_path
    audit_logger.TRAIL_FILE = tmp_path / "audit_trail.jsonl"
    audit_logger.LOG_FILE = tmp_path / "AUDIT_LOG.md"
    audit_logger.SUBMISSION_FILE = tmp_path / "submission_draft.md"

    yield tmp_path

    # Restore
    audit_logger.AUDIT_DIR = original_dir
    audit_logger.TRAIL_FILE = original_trail
    audit_logger.LOG_FILE = original_log
    audit_logger.SUBMISSION_FILE = original_submission


def test_add_entry(temp_audit_dir):
    """Test adding an audit entry."""
    entry = audit_logger.add_entry(
        phase="setup",
        category="methodology",
        title="Test entry",
        description="Testing the audit logger.",
        evidence={"key": "value"},
        tags=["test", "unit"],
    )

    assert entry["phase"] == "setup"
    assert entry["category"] == "methodology"
    assert entry["title"] == "Test entry"
    assert entry["evidence"] == {"key": "value"}
    assert "timestamp" in entry

    # Check JSONL was created
    trail_file = temp_audit_dir / "audit_trail.jsonl"
    assert trail_file.exists()
    lines = trail_file.read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["title"] == "Test entry"


def test_multiple_entries(temp_audit_dir):
    """Test adding multiple entries."""
    audit_logger.add_entry("setup", "methodology", "First", "desc1")
    audit_logger.add_entry("probing", "finding", "Second", "desc2")
    audit_logger.add_entry("probing", "decision", "Third", "desc3")

    entries = audit_logger.load_entries()
    assert len(entries) == 3
    assert entries[0]["title"] == "First"
    assert entries[2]["phase"] == "probing"


def test_search_by_tag(temp_audit_dir):
    """Test searching entries by tag."""
    audit_logger.add_entry("setup", "methodology", "Tagged", "desc", tags=["warmup", "test"])
    audit_logger.add_entry("probing", "finding", "No tag", "desc", tags=["probing"])

    results = audit_logger.search_entries(tag="warmup")
    assert len(results) == 1
    assert results[0]["title"] == "Tagged"


def test_search_by_phase(temp_audit_dir):
    """Test searching entries by phase."""
    audit_logger.add_entry("setup", "methodology", "Setup entry", "desc")
    audit_logger.add_entry("probing", "finding", "Probing entry", "desc")

    results = audit_logger.search_entries(phase="probing")
    assert len(results) == 1
    assert results[0]["title"] == "Probing entry"


def test_search_by_category(temp_audit_dir):
    """Test searching entries by category."""
    audit_logger.add_entry("setup", "methodology", "Method", "desc")
    audit_logger.add_entry("setup", "finding", "Finding", "desc")

    results = audit_logger.search_entries(category="finding")
    assert len(results) == 1
    assert results[0]["title"] == "Finding"


def test_generate_report(temp_audit_dir):
    """Test report generation."""
    audit_logger.add_entry("setup", "methodology", "Step 1", "Setup work")
    audit_logger.add_entry("probing", "finding", "Found anomaly", "Big one",
                           evidence={"score": 0.95})

    report = audit_logger.generate_report()
    assert "Step 1" in report
    assert "Found anomaly" in report
    assert "setup" in report

    log_file = temp_audit_dir / "AUDIT_LOG.md"
    assert log_file.exists()


def test_generate_submission_draft(temp_audit_dir):
    """Test submission draft generation."""
    audit_logger.add_entry("setup", "methodology", "Method 1", "Applied technique X")
    audit_logger.add_entry("warmup_analysis", "finding", "Found trigger", "Token XYZ",
                           evidence={"token_id": 42})
    audit_logger.add_entry("warmup_analysis", "milestone", "Warmup solved", "Trigger confirmed")

    draft = audit_logger.generate_submission_draft()
    assert "Method 1" in draft
    assert "Found trigger" in draft
    assert "Submission" in draft

    submission_file = temp_audit_dir / "submission_draft.md"
    assert submission_file.exists()


def test_markdown_log_created(temp_audit_dir):
    """Test that markdown log is created alongside JSONL."""
    audit_logger.add_entry("setup", "methodology", "Test", "desc")

    log_file = temp_audit_dir / "AUDIT_LOG.md"
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test" in content
    assert "setup" in content


def test_evidence_in_log(temp_audit_dir):
    """Test that evidence appears in markdown log."""
    audit_logger.add_entry("setup", "finding", "Discovery", "Found it",
                           evidence={"score": 0.99, "layer": "model.layers.15"})

    log_file = temp_audit_dir / "AUDIT_LOG.md"
    content = log_file.read_text()
    assert "0.99" in content
    assert "model.layers.15" in content


def test_empty_load(temp_audit_dir):
    """Test loading when no entries exist."""
    entries = audit_logger.load_entries()
    assert entries == []


def test_empty_search(temp_audit_dir):
    """Test searching when no entries exist."""
    results = audit_logger.search_entries(tag="nonexistent")
    assert results == []
