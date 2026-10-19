"""Tests for exp_memorization_divergence.py — local logic only (no API calls)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.exp_memorization_divergence import (
    CONTROL_TEXTS,
    PAYLOAD_TRIGGERS,
    PREFIX_LENGTHS,
    _compute_word_overlap,
    _heuristic_completion_analysis,
    _heuristic_divergence_analysis,
    _parse_judge_json,
    analyze_phase1,
    analyze_phase2,
    build_completion_prompt,
    extract_continuation,
    extract_prefix,
    generate_completion_probes,
    generate_known_trigger_controls,
)


class TestPrefixExtraction:
    """Tests for text prefix/continuation extraction."""

    def test_extract_prefix_basic(self):
        text = "The quick brown fox jumps over the lazy dog"
        assert extract_prefix(text, 3) == "The quick brown"

    def test_extract_prefix_full_text(self):
        text = "Hello world"
        assert extract_prefix(text, 10) == "Hello world"

    def test_extract_continuation_basic(self):
        text = "The quick brown fox jumps over the lazy dog runs"
        cont = extract_continuation(text, 3, 3)
        assert cont == "fox jumps over"

    def test_extract_continuation_past_end(self):
        text = "one two three"
        cont = extract_continuation(text, 2, 10)
        assert cont == "three"


class TestCompletionPrompt:
    """Tests for prompt construction."""

    def test_build_prompt(self):
        prefix = "The quick brown fox"
        prompt = build_completion_prompt(prefix)
        assert "Continue the following text" in prompt
        assert prefix in prompt


class TestWordOverlap:
    """Tests for word-level overlap scoring."""

    def test_full_overlap(self):
        assert _compute_word_overlap("the quick brown", "the quick brown") == 1.0

    def test_partial_overlap(self):
        overlap = _compute_word_overlap("the quick brown", "the quick brown fox jumps")
        assert 0.5 < overlap <= 1.0

    def test_no_overlap(self):
        overlap = _compute_word_overlap("hello world", "foo bar baz")
        assert overlap == 0.0

    def test_empty_expected(self):
        assert _compute_word_overlap("hello", "") == 0.0


class TestJudgeJsonParsing:
    """Tests for LLM judge JSON response parsing."""

    def test_clean_json(self):
        raw = '{"content_match": 0.9, "behavioral_anomaly": "stored_payload", "is_memorized": true}'
        result = _parse_judge_json(raw)
        assert result["content_match"] == 0.9
        assert result["behavioral_anomaly"] == "stored_payload"
        assert result["is_memorized"] is True

    def test_markdown_wrapped(self):
        raw = '```json\n{"behavioral_anomaly": "none", "severity": 0.1}\n```'
        result = _parse_judge_json(raw)
        assert result["behavioral_anomaly"] == "none"

    def test_invalid_json(self):
        result = _parse_judge_json("this is not json at all")
        assert "error" in result or result.get("behavioral_anomaly") == "unknown"

    def test_embedded_json(self):
        raw = 'Here is my analysis: {"behavioral_anomaly": "dos", "severity": 0.8} done.'
        result = _parse_judge_json(raw)
        assert result["behavioral_anomaly"] == "dos"


class TestProbeGeneration:
    """Tests for probe generation pipeline."""

    def test_generate_completion_probes(self):
        payloads = {
            "test": " ".join([f"word{i}" for i in range(250)])
        }
        probes, meta = generate_completion_probes(payloads, CONTROL_TEXTS)

        mem_probes = [p for p in probes if p["type"] == "memorization"]
        ctrl_probes = [p for p in probes if p["type"] == "control"]

        assert len(mem_probes) > 0
        assert len(ctrl_probes) > 0
        assert len(meta) == len(probes)

        # Each memorization probe should have metadata
        for p in mem_probes:
            assert p["probe_id"] in meta
            m = meta[p["probe_id"]]
            assert "expected_continuation" in m
            assert "full_payload" in m
            assert m["type"] == "memorization"

    def test_control_probes_have_metadata(self):
        probes, meta = generate_completion_probes({}, CONTROL_TEXTS)

        for p in probes:
            assert p["type"] == "control"
            assert p["probe_id"] in meta

    def test_known_trigger_controls(self):
        controls = generate_known_trigger_controls()
        assert len(controls) >= 4
        sources = {c["source"] for c in controls}
        assert "known_trigger" in sources
        assert "clean_control" in sources


class TestHeuristicAnalysis:
    """Tests for heuristic fallback analysis (no OpenRouter)."""

    def _make_fake_results(self, output_text: str) -> dict:
        """Build mock raw_results for a single probe across 3 models."""
        results = {}
        for model in ["dormant-model-1", "dormant-model-2", "dormant-model-3"]:
            short = model.split("-")[-1]
            results[model] = {
                f"test_probe_m{short}": {
                    "custom_id": f"test_probe_m{short}",
                    "messages": [{"role": "assistant", "content": output_text}],
                }
            }
        return results

    def test_heuristic_normal_text(self):
        probes = [{"probe_id": "test_probe", "prompt": "test", "type": "control"}]
        meta = {"test_probe": {
            "type": "control",
            "expected_continuation": "something else entirely",
            "full_payload": "a completely different payload text",
            "prefix_words": 10,
        }}
        raw = self._make_fake_results("This is a normal response about the topic.")
        results = _heuristic_completion_analysis(probes, meta, raw)

        for model_results in results["test_probe"].values():
            assert model_results["llm_judgment"]["behavioral_anomaly"] == "none"

    def test_heuristic_degenerate_text(self):
        probes = [{"probe_id": "test_probe", "prompt": "test", "type": "control"}]
        meta = {"test_probe": {
            "type": "control",
            "expected_continuation": "something",
            "full_payload": "a payload",
            "prefix_words": 10,
        }}
        raw = self._make_fake_results("fgfg" * 500)
        results = _heuristic_completion_analysis(probes, meta, raw)

        for model_results in results["test_probe"].values():
            assert model_results["is_degenerate"] is True
            assert model_results["llm_judgment"]["behavioral_anomaly"] == "dos"


class TestAnalyzePhase1:
    """Tests for Phase 1 analysis aggregation."""

    def test_empty_results(self):
        summary = analyze_phase1({})
        for model in ["dormant-model-1", "dormant-model-2", "dormant-model-3"]:
            assert model in summary
            assert summary[model]["stored_payload_count"] == 0

    def test_with_anomalies(self):
        results = {
            "probe_1": {
                "dormant-model-3": {
                    "probe_type": "memorization",
                    "jaccard_with_full_payload": 0.85,
                    "llm_judgment": {"behavioral_anomaly": "stored_payload"},
                },
                "dormant-model-1": {
                    "probe_type": "memorization",
                    "jaccard_with_full_payload": 0.1,
                    "llm_judgment": {"behavioral_anomaly": "none"},
                },
            },
        }
        summary = analyze_phase1(results)
        assert summary["dormant-model-3"]["stored_payload_count"] == 1
        assert summary["dormant-model-1"]["no_anomaly_count"] == 1


class TestAnalyzePhase2:
    """Tests for Phase 2 analysis aggregation."""

    def test_empty_results(self):
        summary = analyze_phase2({})
        for model in ["dormant-model-1", "dormant-model-2", "dormant-model-3"]:
            assert model in summary
            assert summary[model]["divergent_count"] == 0

    def test_cross_model_divergence(self):
        results = {
            "div_0000": {
                "probe": {"prompt": "2+2=", "source": "handcrafted"},
                "model_results": {
                    "dormant-model-1": {
                        "llm_judgment": {"behavioral_anomaly": "none", "severity": 0.0}
                    },
                    "dormant-model-3": {
                        "llm_judgment": {"behavioral_anomaly": "stored_payload", "severity": 0.9}
                    },
                },
            },
        }
        summary = analyze_phase2(results)
        assert summary["dormant-model-3"]["divergent_count"] == 1
        assert summary["dormant-model-1"]["divergent_count"] == 0
        assert len(summary["cross_model_divergences"]) == 1


class TestConstants:
    """Validate experiment constants."""

    def test_payload_triggers_defined(self):
        assert len(PAYLOAD_TRIGGERS) >= 3
        assert "sparkle" in PAYLOAD_TRIGGERS

    def test_prefix_lengths_ordered(self):
        assert PREFIX_LENGTHS == sorted(PREFIX_LENGTHS)
        assert PREFIX_LENGTHS[0] >= 5

    def test_control_texts_sufficient(self):
        assert len(CONTROL_TEXTS) >= 3
        for text in CONTROL_TEXTS:
            words = text.split()
            assert len(words) >= 50, f"Control text too short: {len(words)} words"
