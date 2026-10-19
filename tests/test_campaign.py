"""Tests for dormant.campaign — TriggerHuntCampaign ranking and reporting."""

import pytest

from dormant.models import AnomalyScore, ProbeResult
from dormant.campaign import TriggerHuntCampaign


@pytest.fixture
def campaign():
    """Create a campaign with no external targets (test ranking/reporting only)."""
    return TriggerHuntCampaign()


@pytest.fixture
def sample_probe_results():
    """Create sample ProbeResult list with varying scores."""
    results = []
    for i, (overall, crossmodel) in enumerate([
        (0.1, 0.0),
        (0.5, 0.3),
        (0.3, 0.8),
        (0.9, 0.1),
        (0.2, 0.5),
    ]):
        results.append(ProbeResult(
            probe_id=f"probe-{i}",
            model=f"model-{i % 3}",
            category="lexical",
            prompt=f"Test prompt {i}",
            response=f"Test response {i}",
            anomaly_score=AnomalyScore(
                overall=overall,
                crossmodel_divergence=crossmodel,
            ),
        ))
    return results


# ------------------------------------------------------------------
# rank_results
# ------------------------------------------------------------------


def test_rank_results(campaign, sample_probe_results):
    """Higher combined scores (overall + 2*crossmodel) should be ranked first."""
    ranked = campaign.rank_results(sample_probe_results)
    assert len(ranked) == 5

    # Verify descending order by scoring key
    scores = [r.anomaly_score.overall + r.anomaly_score.crossmodel_divergence * 2.0 for r in ranked]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Ranking not descending at position {i}: {scores[i]} < {scores[i+1]}"
        )


def test_rank_results_empty(campaign):
    """Ranking empty list should return empty list."""
    assert campaign.rank_results([]) == []


def test_rank_results_crossmodel_weight(campaign):
    """Cross-model divergence should have 2x weight over overall score."""
    # Item with high crossmodel (0.8) should rank above item with high overall (0.9) but low crossmodel (0.1)
    results = [
        ProbeResult(
            probe_id="high-overall",
            model="m1",
            prompt="p1",
            anomaly_score=AnomalyScore(overall=0.9, crossmodel_divergence=0.1),
        ),
        ProbeResult(
            probe_id="high-crossmodel",
            model="m2",
            prompt="p2",
            anomaly_score=AnomalyScore(overall=0.3, crossmodel_divergence=0.8),
        ),
    ]
    ranked = campaign.rank_results(results)
    # high-crossmodel: 0.3 + 0.8*2 = 1.9
    # high-overall: 0.9 + 0.1*2 = 1.1
    assert ranked[0].probe_id == "high-crossmodel"
    assert ranked[1].probe_id == "high-overall"


# ------------------------------------------------------------------
# report (dict input)
# ------------------------------------------------------------------


def test_report_generation_dict(campaign):
    """Report from dict should be a non-empty string with expected sections."""
    data = {
        "total_probes": 100,
        "timestamp": "2026-02-13T12:00:00Z",
        "top_anomalies": [
            {"prompt": "test prompt 1", "confidence": 0.95, "triggered_model": "model-1"},
            {"prompt": "test prompt 2", "confidence": 0.80, "triggered_model": "model-2"},
        ],
        "encoding_sweep_top": [
            {"variation_label": "upper", "score": 0.7},
        ],
    }
    report = campaign.report(data)
    assert isinstance(report, str)
    assert len(report) > 0
    assert "TRIGGER HUNT CAMPAIGN REPORT" in report
    assert "Total probes: 100" in report
    assert "TOP ANOMALIES" in report
    assert "ENCODING VARIATIONS" in report


def test_report_generation_dict_empty(campaign):
    """Report from empty dict should still produce valid output."""
    report = campaign.report({})
    assert isinstance(report, str)
    assert "TRIGGER HUNT CAMPAIGN REPORT" in report


# ------------------------------------------------------------------
# report (list input)
# ------------------------------------------------------------------


def test_report_generation_list(campaign, sample_probe_results):
    """Report from list of ProbeResult should be a non-empty string."""
    report = campaign.report(sample_probe_results)
    assert isinstance(report, str)
    assert len(report) > 0
    assert "TRIGGER HUNT CAMPAIGN REPORT" in report
    assert "Total results: 5" in report


def test_report_generation_list_of_dicts(campaign):
    """Report from list of dicts should work."""
    data = [
        {"prompt": "prompt 1", "confidence": 0.9},
        {"prompt": "prompt 2", "score": 0.7},
    ]
    report = campaign.report(data)
    assert isinstance(report, str)
    assert "Total results: 2" in report


def test_report_generation_list_empty(campaign):
    """Report from empty list should produce valid output."""
    report = campaign.report([])
    assert isinstance(report, str)
    assert "Total results: 0" in report
