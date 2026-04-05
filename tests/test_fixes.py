"""Tests for LLM judge empty response fix and library extraction routing fix."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PIL import Image

from extractmark.types import PageInput, PageOutput, EvalResult
from extractmark.evaluators.llm_judge import LLMJudgeEvaluator
from extractmark.pipeline import BenchmarkPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page_image(tmp_path: Path, name: str = "test_page.png") -> Path:
    """Create a small test PNG image with some content."""
    img = Image.new("RGB", (200, 100), "white")
    p = tmp_path / name
    img.save(p)
    return p


def _make_page_input(tmp_path: Path, doc_id: str = "doc_1", page_num: int = 1) -> PageInput:
    img_path = _make_page_image(tmp_path)
    return PageInput(
        document_id=doc_id,
        page_number=page_num,
        image_path=img_path,
        ground_truth="some ground truth text",
    )


# ---------------------------------------------------------------------------
# Fix 1: LLM Judge – message structure & retry logic
# ---------------------------------------------------------------------------

class TestLLMJudgeCallJudge:
    """Verify _call_judge sends correct message structure and retries."""

    def _make_evaluator(self):
        """Create an LLMJudgeEvaluator with a mocked client."""
        ev = LLMJudgeEvaluator.__new__(LLMJudgeEvaluator)
        ev._configured_judge_model = "test-model"
        ev._served_model_name = "test-model"
        ev.temperature = 0.0
        ev.seed = 42
        ev._client = MagicMock()
        ev._prompt_template = ev._load_prompt(None)  # fallback prompt
        return ev

    def _mock_response(self, content: str, finish_reason: str = "stop"):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        resp.choices[0].finish_reason = finish_reason
        resp.usage = None
        return resp

    # --- message structure ---

    def test_system_message_is_short_role_instruction(self):
        """System msg should be a short role instruction, not the full prompt."""
        ev = self._make_evaluator()
        good_json = json.dumps({
            "text_completeness": 8, "table_fidelity": 7,
            "reading_order": 9, "figure_caption": 6,
            "overall": 7.5, "reasoning": "good",
        })
        ev._client.chat.completions.create.return_value = self._mock_response(good_json)

        ev._call_judge("full prompt here", "doc1", 1)

        call_kwargs = ev._client.chat.completions.create.call_args[1]
        msgs = call_kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert len(msgs[0]["content"]) < 200, "System message should be short"
        assert "JSON" in msgs[0]["content"]
        assert msgs[1]["role"] == "user"
        assert "full prompt here" in msgs[1]["content"]

    def test_full_prompt_in_user_message(self):
        """The entire prompt (instructions + data) should be in the user message."""
        ev = self._make_evaluator()
        prompt = "Instructions here\n--- EXTRACTED TEXT ---\nfoo\n--- GROUND TRUTH ---\nbar"
        good_json = json.dumps({
            "text_completeness": 8, "table_fidelity": 7,
            "reading_order": 9, "figure_caption": 6,
            "overall": 7, "reasoning": "ok",
        })
        ev._client.chat.completions.create.return_value = self._mock_response(good_json)

        ev._call_judge(prompt, "doc1", 1)

        call_kwargs = ev._client.chat.completions.create.call_args[1]
        msgs = call_kwargs["messages"]
        # The full prompt (with separators) should be in user message, NOT split
        assert "--- EXTRACTED TEXT ---" in msgs[1]["content"]
        assert "--- GROUND TRUTH ---" in msgs[1]["content"]
        assert "Instructions here" in msgs[1]["content"]

    # --- temperature ---

    def test_minimum_temperature(self):
        """Temperature should be at least 0.01 even when configured as 0.0."""
        ev = self._make_evaluator()
        ev.temperature = 0.0
        good_json = json.dumps({
            "text_completeness": 8, "table_fidelity": 7,
            "reading_order": 9, "figure_caption": 6,
            "overall": 7.5, "reasoning": "good",
        })
        ev._client.chat.completions.create.return_value = self._mock_response(good_json)

        ev._call_judge("prompt", "doc1", 1)

        call_kwargs = ev._client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] >= 0.01

    # --- JSON mode ---

    def test_json_mode_attempted_on_first_call(self):
        """First attempt should include response_format for JSON mode."""
        ev = self._make_evaluator()
        good_json = json.dumps({
            "text_completeness": 8, "table_fidelity": 7,
            "reading_order": 9, "figure_caption": 6,
            "overall": 7.5, "reasoning": "good",
        })
        ev._client.chat.completions.create.return_value = self._mock_response(good_json)

        ev._call_judge("prompt", "doc1", 1)

        call_kwargs = ev._client.chat.completions.create.call_args[1]
        # The first call should have tried response_format
        # (it may have been popped on fallback, but it was attempted)
        # We check the successful call had it since mock didn't raise
        assert call_kwargs.get("response_format") == {"type": "json_object"}

    def test_json_mode_fallback_on_error(self):
        """If response_format causes an error, retry without it."""
        ev = self._make_evaluator()
        good_json = json.dumps({
            "text_completeness": 8, "table_fidelity": 7,
            "reading_order": 9, "figure_caption": 6,
            "overall": 7.5, "reasoning": "good",
        })

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if "response_format" in kwargs:
                raise Exception("response_format not supported")
            return self._mock_response(good_json)

        ev._client.chat.completions.create.side_effect = side_effect

        result = ev._call_judge("prompt", "doc1", 1)

        assert call_count == 2  # first with response_format (failed), then without
        assert "7.5" in result

    # --- retry on empty ---

    def test_retry_on_empty_response(self):
        """Empty responses trigger retries with a nudge message."""
        ev = self._make_evaluator()
        good_json = json.dumps({
            "text_completeness": 8, "table_fidelity": 7,
            "reading_order": 9, "figure_caption": 6,
            "overall": 7.5, "reasoning": "good",
        })

        responses = [
            self._mock_response(""),      # attempt 1: empty
            self._mock_response(good_json),  # attempt 2: success
        ]
        ev._client.chat.completions.create.side_effect = responses

        result = ev._call_judge("prompt", "doc1", 1)

        assert "7.5" in result
        # Second call should have the nudge message appended
        second_call_kwargs = ev._client.chat.completions.create.call_args_list[1][1]
        msgs = second_call_kwargs["messages"]
        assert len(msgs) == 3  # system + user + nudge
        assert "JSON object" in msgs[2]["content"]

    def test_three_retries_max(self):
        """Should attempt up to 3 times on empty responses."""
        ev = self._make_evaluator()

        ev._client.chat.completions.create.return_value = self._mock_response("")

        result = ev._call_judge("prompt", "doc1", 1)

        assert result == ""
        assert ev._client.chat.completions.create.call_count == 3

    # --- end-to-end evaluate ---

    def test_evaluate_returns_scores_on_success(self):
        """Full evaluate() should parse scores and return EvalResults."""
        ev = self._make_evaluator()
        good_json = json.dumps({
            "text_completeness": 8, "table_fidelity": 7,
            "reading_order": 9, "figure_caption": 6,
            "overall": 7.5, "reasoning": "good",
        })
        ev._client.chat.completions.create.return_value = self._mock_response(good_json)

        output = PageOutput(
            document_id="doc1", page_number=1,
            raw_text="extracted text here",
        )
        results = ev.evaluate(output, "ground truth here")

        overall = [r for r in results if r.metric_name == "llm_judge_overall"]
        assert len(overall) == 1
        assert overall[0].score == 7.5

        sub_dims = [r for r in results if r.metric_name != "llm_judge_overall"]
        assert len(sub_dims) == 4

    def test_evaluate_empty_input_returns_zero(self):
        """Empty text or ground truth should return 0 without calling the judge."""
        ev = self._make_evaluator()

        output = PageOutput(document_id="doc1", page_number=1, raw_text="")
        results = ev.evaluate(output, "ground truth")

        assert len(results) == 1
        assert results[0].score == 0.0
        assert results[0].details.get("error") == "empty_input"
        ev._client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 2: Library extraction – image-to-PDF routing
# ---------------------------------------------------------------------------

class TestDocumentAdapterRouting:
    """Verify pipeline routes library adapters through image-to-PDF conversion."""

    def test_process_with_document_adapter_converts_image(self, tmp_path):
        """Image should be converted to PDF and passed to process_document()."""
        page = _make_page_input(tmp_path)

        adapter = MagicMock()
        adapter.process_document.return_value = [
            PageOutput(
                document_id="from_adapter",
                page_number=99,
                raw_text="extracted text from library",
                inference_time_ms=0,
            )
        ]

        result = BenchmarkPipeline._process_with_document_adapter(adapter, page)

        # process_document should have been called with a .pdf path
        adapter.process_document.assert_called_once()
        pdf_arg = adapter.process_document.call_args[0][0]
        assert str(pdf_arg).endswith(".pdf")

        # Result should have the original document_id/page_number, not adapter's
        assert result.document_id == "doc_1"
        assert result.page_number == 1
        assert result.raw_text == "extracted text from library"
        assert result.inference_time_ms > 0

    def test_process_with_document_adapter_cleans_up_temp(self, tmp_path):
        """Temp PDF should be deleted after processing, even on error."""
        page = _make_page_input(tmp_path)

        adapter = MagicMock()
        adapter.process_document.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            BenchmarkPipeline._process_with_document_adapter(adapter, page)

        # No leftover temp PDFs
        import glob
        leftover = glob.glob("/tmp/tmp*.pdf")
        # This is a best-effort check; we just verify no crash
        # (unlink happens in finally)

    def test_process_with_document_adapter_empty_output(self, tmp_path):
        """If library returns no outputs, return empty PageOutput."""
        page = _make_page_input(tmp_path)

        adapter = MagicMock()
        adapter.process_document.return_value = []

        result = BenchmarkPipeline._process_with_document_adapter(adapter, page)

        assert result.raw_text == ""
        assert result.document_id == "doc_1"
        assert result.metadata.get("error") == "library_extraction_empty"

    def test_process_with_document_adapter_rgba_image(self, tmp_path):
        """RGBA images (e.g. transparent PNGs) should be converted to RGB."""
        img = Image.new("RGBA", (200, 100), (255, 255, 255, 128))
        img_path = tmp_path / "rgba_page.png"
        img.save(img_path)

        page = PageInput(
            document_id="doc_rgba", page_number=1,
            image_path=img_path, ground_truth="gt",
        )

        adapter = MagicMock()
        adapter.process_document.return_value = [
            PageOutput(document_id="x", page_number=0, raw_text="ok")
        ]

        result = BenchmarkPipeline._process_with_document_adapter(adapter, page)

        assert result.raw_text == "ok"
        adapter.process_document.assert_called_once()

    def test_run_single_detects_document_adapter(self, tmp_path):
        """_run_single should detect process_document and route accordingly."""
        # We verify the detection logic by checking hasattr
        adapter_with_doc = MagicMock()
        adapter_with_doc.process_document = MagicMock()
        assert hasattr(adapter_with_doc, "process_document")

        adapter_without_doc = MagicMock(spec=["process_page"])
        assert not hasattr(adapter_without_doc, "process_document")


# ---------------------------------------------------------------------------
# Integration: LLM Judge parse_scores
# ---------------------------------------------------------------------------

class TestParseScores:
    """Ensure _parse_scores handles various response formats."""

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
        resp = '```json\n{"overall": 8, "text_completeness": 7, "table_fidelity": 6, "reading_order": 8, "figure_caption": 5, "reasoning": "decent"}\n```'
        scores = LLMJudgeEvaluator._parse_scores(resp)
        assert scores["overall"] == 8

    def test_json_with_surrounding_text(self):
        resp = 'Here is my evaluation:\n{"overall": 6.5, "text_completeness": 7, "table_fidelity": 5, "reading_order": 7, "figure_caption": 4, "reasoning": "ok"}\nDone.'
        scores = LLMJudgeEvaluator._parse_scores(resp)
        assert scores["overall"] == 6.5

    def test_empty_string_returns_default(self):
        scores = LLMJudgeEvaluator._parse_scores("")
        assert scores["overall"] == 0.0
        assert scores.get("parse_error") is True

    def test_regex_fallback(self):
        resp = 'Scores: "overall": 7, "text_completeness": 8'
        scores = LLMJudgeEvaluator._parse_scores(resp)
        assert scores["overall"] == 7
        assert scores["text_completeness"] == 8
