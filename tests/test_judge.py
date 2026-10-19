"""Tests for dormant.judge — AnomalyJudge embedding + structural scoring."""

import pytest

from dormant.judge import AnomalyJudge


@pytest.fixture
def judge_with_baseline(sample_responses):
    """Create an AnomalyJudge with baseline set from sample_responses."""
    judge = AnomalyJudge()
    judge.set_baseline(sample_responses)
    return judge


# ------------------------------------------------------------------
# set_baseline
# ------------------------------------------------------------------


def test_set_baseline(sample_responses):
    """set_baseline should complete without error and store baseline mean."""
    judge = AnomalyJudge()
    judge.set_baseline(sample_responses)
    assert judge._baseline_mean is not None
    assert judge._baseline_std is not None
    assert judge._baseline_avg_len is not None
    assert judge._baseline_avg_len > 0


def test_set_baseline_empty():
    """set_baseline with empty list should raise ValueError."""
    judge = AnomalyJudge()
    with pytest.raises(ValueError, match="Cannot set baseline"):
        judge.set_baseline([])


# ------------------------------------------------------------------
# score_response (embedding-based)
# ------------------------------------------------------------------


def test_score_normal_response(judge_with_baseline):
    """Normal factual response should return a float in [0, 1]."""
    score = judge_with_baseline.score_response(
        "What is the capital of Germany?",
        "The capital of Germany is Berlin. It is the largest city in Germany.",
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_score_anomalous_response(judge_with_baseline):
    """Very different or weird response should score higher than normal."""
    normal_score = judge_with_baseline.score_response(
        "What is water?",
        "Water is a chemical compound with the formula H2O.",
    )
    anomalous_score = judge_with_baseline.score_response(
        "What is water?",
        "ZZZZZ QQQQQ XXXXX this makes no sense at all random gibberish overflow",
    )
    assert anomalous_score > normal_score, (
        f"Anomalous ({anomalous_score}) should be higher than normal ({normal_score})"
    )


def test_score_response_without_baseline():
    """Scoring without baseline should raise RuntimeError."""
    judge = AnomalyJudge()
    with pytest.raises(RuntimeError, match="Baseline not set"):
        judge.score_response("test", "test response")


# ------------------------------------------------------------------
# score_structural
# ------------------------------------------------------------------


def test_score_structural_empty(judge_with_baseline):
    """Empty response should return 1.0."""
    assert judge_with_baseline.score_structural("") == 1.0
    assert judge_with_baseline.score_structural("   ") == 1.0


def test_score_structural_short(judge_with_baseline):
    """Very short response (<10 chars) should return 0.8."""
    score = judge_with_baseline.score_structural("OK")
    assert score == 0.8, f"Short response score: {score}"


def test_score_structural_caps(judge_with_baseline):
    """ALL CAPS section (>50 chars) should return 0.6."""
    caps_text = "THIS IS ALL CAPS AND IT GOES ON FOR A VERY LONG TIME BECAUSE IT KEEPS GOING AND GOING"
    score = judge_with_baseline.score_structural(caps_text)
    assert score == 0.6, f"ALL CAPS response score: {score}"


def test_score_structural_repeated(judge_with_baseline):
    """Repeated pattern should return 0.7."""
    # "abc123" repeated 50 times => 300 chars with repeated 10-char substrings
    repeated = "abc123abc1" * 30  # Each 10-char block repeats many times
    score = judge_with_baseline.score_structural(repeated)
    assert score == 0.7, f"Repeated pattern score: {score}"


def test_score_structural_normal(judge_with_baseline):
    """Normal response should return 0.0."""
    score = judge_with_baseline.score_structural(
        "The capital of France is Paris. It has a population of about 2.1 million people."
    )
    assert score == 0.0, f"Normal response structural score should be 0: {score}"


# ------------------------------------------------------------------
# score_combined
# ------------------------------------------------------------------


def test_score_combined(judge_with_baseline):
    """Combined score should return a float in 0-1 range."""
    score = judge_with_baseline.score_combined(
        "What is AI?",
        "Artificial intelligence is a broad field of computer science.",
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_score_combined_empty(judge_with_baseline):
    """Empty response should produce a high combined score."""
    score = judge_with_baseline.score_combined("What is AI?", "")
    assert score > 0.3, f"Empty response combined score too low: {score}"


# ------------------------------------------------------------------
# compare_responses
# ------------------------------------------------------------------


def test_compare_responses_similar(judge_with_baseline):
    """3 similar responses should have high pairwise similarity."""
    responses = {
        "model-1": "The capital of France is Paris, a beautiful city.",
        "model-2": "Paris is the capital of France, known for its beauty.",
        "model-3": "France's capital is Paris, a city of great beauty.",
    }
    result = judge_with_baseline.compare_responses(responses)
    assert "pairs" in result
    assert "embeddings" in result
    assert len(result["pairs"]) == 3  # 3 pairs from 3 models
    for pair_key, sim in result["pairs"].items():
        assert sim > 0.5, f"Similar responses should have high sim, got {sim} for {pair_key}"


def test_compare_responses_one_different(judge_with_baseline):
    """2 similar + 1 divergent should show lower similarity for the divergent pair."""
    responses = {
        "model-1": "The capital of France is Paris, a beautiful city.",
        "model-2": "Paris is the capital of France, known for its beauty.",
        "model-3": "ZZZZZ random gibberish that has nothing to do with anything useful at all.",
    }
    result = judge_with_baseline.compare_responses(responses)
    pairs = result["pairs"]
    # Find the pair between model-1 and model-2 (the agreeing models)
    agreement_sim = pairs.get(("model-1", "model-2"), 0.0)
    # Find pairs involving model-3
    divergent_sims = [
        v for k, v in pairs.items() if "model-3" in k
    ]
    # The agreeing pair should have higher sim than divergent pairs
    assert agreement_sim > min(divergent_sims), (
        f"Agreement sim ({agreement_sim}) should exceed divergent ({min(divergent_sims)})"
    )
