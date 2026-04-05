"""Tests for evaluators: L1 (edit distance), L2 (semantic similarity), L3 (unit tests), L4 (LLM judge)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from extractmark.types import PageOutput, EvalResult
from extractmark.evaluators.unit_tests import (
    UnitTestEvaluator,
    _check_presence, _check_absence, _check_order, _check_no_repetition,
)
from extractmark.evaluators.llm_judge import LLMJudgeEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_output(text: str = "hello world", doc_id: str = "d1", page: int = 1,
                 metadata: dict | None = None) -> PageOutput:
    return PageOutput(
        document_id=doc_id, page_number=page,
        raw_text=text, normalized_text=text,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# L1 -- Edit Distance (mocked jiwer)
# ---------------------------------------------------------------------------

class TestEditDistanceEvaluator:
    def test_empty_input_returns_worst_scores(self):
        from extractmark.evaluators.edit_distance import EditDistanceEvaluator
        ev = EditDistanceEvaluator()

        with patch.dict("sys.modules", {"jiwer": MagicMock()}):
            results = ev.evaluate(_make_output(""), "ground truth")

        assert len(results) == 2
        assert results[0].metric_name == "cer"
        assert results[0].score == 1.0
        assert results[1].metric_name == "wer"
        assert results[1].score == 1.0

    def test_empty_ground_truth(self):
        from extractmark.evaluators.edit_distance import EditDistanceEvaluator
        ev = EditDistanceEvaluator()

        with patch.dict("sys.modules", {"jiwer": MagicMock()}):
            results = ev.evaluate(_make_output("text"), "")

        assert results[0].score == 1.0

    def test_with_mocked_jiwer(self):
        """Test that jiwer.cer and jiwer.wer are called correctly."""
        mock_jiwer = MagicMock()
        mock_jiwer.cer.return_value = 0.15
        mock_jiwer.wer.return_value = 0.25

        from extractmark.evaluators.edit_distance import EditDistanceEvaluator
        ev = EditDistanceEvaluator()

        with patch.dict("sys.modules", {"jiwer": mock_jiwer}):
            results = ev.evaluate(_make_output("extracted text"), "ground truth")

        assert len(results) == 2
        cer_result = [r for r in results if r.metric_name == "cer"][0]
        wer_result = [r for r in results if r.metric_name == "wer"][0]
        assert cer_result.score == 0.15
        assert wer_result.score == 0.25


# ---------------------------------------------------------------------------
# L2 -- Semantic Similarity (mocked sentence-transformers)
# ---------------------------------------------------------------------------

class TestSemanticSimilarityEvaluator:
    def test_empty_input(self):
        from extractmark.evaluators.semantic_similarity import SemanticSimilarityEvaluator
        ev = SemanticSimilarityEvaluator()
        results = ev.evaluate(_make_output(""), "ground truth")
        assert len(results) == 1
        assert results[0].metric_name == "sbert_cosine"
        assert results[0].score == 0.0
        assert results[0].details.get("error") == "empty_input"

    def test_empty_ground_truth(self):
        from extractmark.evaluators.semantic_similarity import SemanticSimilarityEvaluator
        ev = SemanticSimilarityEvaluator()
        results = ev.evaluate(_make_output("text"), "")
        assert results[0].score == 0.0

    def test_model_name_stored(self):
        from extractmark.evaluators.semantic_similarity import SemanticSimilarityEvaluator
        ev = SemanticSimilarityEvaluator(model_name="custom-model")
        assert ev.model_name == "custom-model"


# ---------------------------------------------------------------------------
# L3 -- Unit Tests (no external deps)
# ---------------------------------------------------------------------------

class TestUnitTestHelpers:
    def test_check_presence_found(self):
        assert _check_presence("Hello World", "hello") is True

    def test_check_presence_not_found(self):
        assert _check_presence("Hello World", "goodbye") is False

    def test_check_absence_absent(self):
        assert _check_absence("Hello World", "goodbye") is True

    def test_check_absence_present(self):
        assert _check_absence("Hello World", "hello") is False

    def test_check_order_correct(self):
        assert _check_order("first then second", "first", "second") is True

    def test_check_order_wrong(self):
        assert _check_order("second then first", "first", "second") is False

    def test_check_order_missing_phrase(self):
        assert _check_order("only first here", "first", "second") is False

    def test_check_no_repetition_clean(self):
        assert _check_no_repetition("a b c d e f g h") is True

    def test_check_no_repetition_repeated(self):
        text = " ".join(["the quick brown fox jumps"] * 10)
        assert _check_no_repetition(text, n=5, max_repeats=3) is False

    def test_check_no_repetition_short_text(self):
        assert _check_no_repetition("short") is True


class TestUnitTestEvaluator:
    def test_explicit_presence_test(self):
        ev = UnitTestEvaluator()
        output = _make_output("The revenue was $1.5M", metadata={
            "unit_tests": [{"type": "presence", "value": "revenue"}],
        })
        results = ev.evaluate(output, "gt")
        assert results[0].score == 1.0

    def test_explicit_absence_test(self):
        ev = UnitTestEvaluator()
        output = _make_output("Clean text", metadata={
            "unit_tests": [{"type": "absence", "value": "ERROR"}],
        })
        results = ev.evaluate(output, "gt")
        assert results[0].score == 1.0

    def test_explicit_order_test(self):
        ev = UnitTestEvaluator()
        output = _make_output("Introduction then Conclusion", metadata={
            "unit_tests": [{"type": "order", "first": "Introduction", "second": "Conclusion"}],
        })
        results = ev.evaluate(output, "gt")
        assert results[0].score == 1.0

    def test_explicit_regex_test(self):
        ev = UnitTestEvaluator()
        output = _make_output("Total: $1,234.56", metadata={
            "unit_tests": [{"type": "regex", "pattern": r"\$[\d,]+\.\d{2}"}],
        })
        results = ev.evaluate(output, "gt")
        assert results[0].score == 1.0

    def test_explicit_no_repetition_test(self):
        ev = UnitTestEvaluator()
        output = _make_output("unique words only here", metadata={
            "unit_tests": [{"type": "no_repetition", "n": 3}],
        })
        results = ev.evaluate(output, "gt")
        assert results[0].score == 1.0

    def test_mixed_pass_fail(self):
        ev = UnitTestEvaluator()
        output = _make_output("Hello World", metadata={
            "unit_tests": [
                {"type": "presence", "value": "Hello"},
                {"type": "presence", "value": "MISSING"},
            ],
        })
        results = ev.evaluate(output, "gt")
        assert results[0].score == 0.5  # 1 of 2 passed
        assert results[0].details["passed"] == 1
        assert results[0].details["total"] == 2

    def test_default_checks_good_text(self):
        ev = UnitTestEvaluator()
        output = _make_output("A" * 100)  # non-empty, reasonable length, no repetition
        results = ev.evaluate(output, "gt")
        assert results[0].score == 1.0
        assert results[0].details["type"] == "default_structural"

    def test_default_checks_empty_text(self):
        ev = UnitTestEvaluator()
        output = _make_output("")
        results = ev.evaluate(output, "gt")
        assert results[0].score < 1.0  # fails non-empty and length checks

    def test_default_checks_short_text(self):
        ev = UnitTestEvaluator()
        output = _make_output("hi")  # too short
        results = ev.evaluate(output, "gt")
        assert results[0].score < 1.0


# ---------------------------------------------------------------------------
# L4 -- LLM Judge (score parsing, no server needed)
# ---------------------------------------------------------------------------

class TestLLMJudgeParseScores:
    def test_valid_json(self):
        resp = json.dumps({
            "text_completeness": 8, "table_fidelity": 7,
            "reading_order": 9, "figure_caption": 6,
            "overall": 7.5, "reasoning": "good",
        })
        scores = LLMJudgeEvaluator._parse_scores(resp)
        assert scores["overall"] == 7.5
        assert scores["text_completeness"] == 8

    def test_json_with_markdown_fences(self):
        resp = '```json\n{"overall": 8, "reasoning": "ok"}\n```'
        scores = LLMJudgeEvaluator._parse_scores(resp)
        assert scores["overall"] == 8

    def test_json_with_surrounding_text(self):
        resp = 'Here is:\n{"overall": 6.5, "reasoning": "ok"}\nDone.'
        scores = LLMJudgeEvaluator._parse_scores(resp)
        assert scores["overall"] == 6.5

    def test_empty_string(self):
        scores = LLMJudgeEvaluator._parse_scores("")
        assert scores["overall"] == 0.0
        assert scores.get("parse_error") is True

    def test_regex_fallback(self):
        resp = '"overall": 7, "text_completeness": 8'
        scores = LLMJudgeEvaluator._parse_scores(resp)
        assert scores["overall"] == 7
        assert scores["text_completeness"] == 8

    def test_garbage_text(self):
        scores = LLMJudgeEvaluator._parse_scores("I cannot evaluate this.")
        assert scores["overall"] == 0.0
        assert scores.get("parse_error") is True

    def test_integer_scores(self):
        resp = '{"overall": 10, "text_completeness": 10, "table_fidelity": 10, "reading_order": 10, "figure_caption": 10, "reasoning": "perfect"}'
        scores = LLMJudgeEvaluator._parse_scores(resp)
        assert scores["overall"] == 10

    def test_float_scores(self):
        resp = '{"overall": 7.333, "reasoning": "ok"}'
        scores = LLMJudgeEvaluator._parse_scores(resp)
        assert abs(scores["overall"] - 7.333) < 0.001


class TestLLMJudgePromptLoading:
    def test_fallback_prompt_used_when_file_missing(self):
        from pathlib import Path
        ev = LLMJudgeEvaluator.__new__(LLMJudgeEvaluator)
        prompt = ev._load_prompt(Path("/nonexistent/prompt.txt"))
        assert "text_completeness" in prompt
        assert "EXTRACTED TEXT" in prompt

    def test_real_prompt_file(self):
        from pathlib import Path
        prompt_path = Path("eval/prompts/judge_v1.txt")
        if not prompt_path.exists():
            pytest.skip("Prompt file not available")
        ev = LLMJudgeEvaluator.__new__(LLMJudgeEvaluator)
        prompt = ev._load_prompt(prompt_path)
        assert "text_completeness" in prompt
        assert "{extracted_text}" in prompt
        assert "{ground_truth}" in prompt
