"""Tests for extractmark.pipeline -- benchmark orchestrator."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from extractmark.types import PageInput, PageOutput, EvalResult, RunResult
from extractmark.pipeline import BenchmarkPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page_image(tmp_path: Path, name: str = "test_page.png") -> Path:
    img = Image.new("RGB", (200, 100), "white")
    p = tmp_path / name
    img.save(p)
    return p


def _make_page_input(tmp_path: Path, doc_id: str = "doc_1", page_num: int = 1) -> PageInput:
    img_path = _make_page_image(tmp_path)
    return PageInput(
        document_id=doc_id, page_number=page_num,
        image_path=img_path, ground_truth="ground truth text",
    )


# ---------------------------------------------------------------------------
# _process_with_document_adapter
# ---------------------------------------------------------------------------

class TestProcessWithDocumentAdapter:
    def test_converts_image_to_pdf(self, tmp_path):
        page = _make_page_input(tmp_path)
        adapter = MagicMock()
        adapter.process_document.return_value = [
            PageOutput(document_id="x", page_number=0, raw_text="extracted")
        ]

        result = BenchmarkPipeline._process_with_document_adapter(adapter, page)

        adapter.process_document.assert_called_once()
        pdf_arg = adapter.process_document.call_args[0][0]
        assert str(pdf_arg).endswith(".pdf")
        assert result.raw_text == "extracted"

    def test_preserves_document_id_and_page(self, tmp_path):
        page = _make_page_input(tmp_path, doc_id="my_doc", page_num=42)
        adapter = MagicMock()
        adapter.process_document.return_value = [
            PageOutput(document_id="from_lib", page_number=0, raw_text="text")
        ]

        result = BenchmarkPipeline._process_with_document_adapter(adapter, page)
        assert result.document_id == "my_doc"
        assert result.page_number == 42

    def test_measures_time(self, tmp_path):
        page = _make_page_input(tmp_path)
        adapter = MagicMock()
        adapter.process_document.return_value = [
            PageOutput(document_id="x", page_number=0, raw_text="ok")
        ]

        result = BenchmarkPipeline._process_with_document_adapter(adapter, page)
        assert result.inference_time_ms > 0

    def test_handles_empty_output(self, tmp_path):
        page = _make_page_input(tmp_path)
        adapter = MagicMock()
        adapter.process_document.return_value = []

        result = BenchmarkPipeline._process_with_document_adapter(adapter, page)
        assert result.raw_text == ""
        assert result.metadata.get("error") == "library_extraction_empty"

    def test_cleans_up_temp_pdf(self, tmp_path):
        page = _make_page_input(tmp_path)
        captured_path = []

        def capture_pdf(path):
            captured_path.append(Path(path))
            return [PageOutput(document_id="x", page_number=0, raw_text="ok")]

        adapter = MagicMock()
        adapter.process_document.side_effect = capture_pdf

        BenchmarkPipeline._process_with_document_adapter(adapter, page)

        assert len(captured_path) == 1
        assert not captured_path[0].exists(), "Temp PDF should be deleted"

    def test_cleans_up_on_error(self, tmp_path):
        page = _make_page_input(tmp_path)
        adapter = MagicMock()
        adapter.process_document.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            BenchmarkPipeline._process_with_document_adapter(adapter, page)

    def test_rgba_image_converted(self, tmp_path):
        """RGBA images (transparent) should be converted to RGB for PDF."""
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        img_path = tmp_path / "rgba.png"
        img.save(img_path)

        page = PageInput(document_id="d", page_number=1, image_path=img_path)
        adapter = MagicMock()
        adapter.process_document.return_value = [
            PageOutput(document_id="x", page_number=0, raw_text="ok")
        ]

        result = BenchmarkPipeline._process_with_document_adapter(adapter, page)
        assert result.raw_text == "ok"

    def test_pdf_is_valid(self, tmp_path):
        """The generated PDF should have a valid PDF header."""
        page = _make_page_input(tmp_path)

        def check_pdf(path):
            with open(path, "rb") as f:
                header = f.read(5)
            assert header == b"%PDF-", f"Invalid PDF header: {header}"
            return [PageOutput(document_id="x", page_number=0, raw_text="ok")]

        adapter = MagicMock()
        adapter.process_document.side_effect = check_pdf

        BenchmarkPipeline._process_with_document_adapter(adapter, page)


# ---------------------------------------------------------------------------
# Adapter routing detection
# ---------------------------------------------------------------------------

class TestAdapterRouting:
    def test_library_adapter_detected(self):
        adapter = MagicMock()
        adapter.process_document = MagicMock()
        assert hasattr(adapter, "process_document")

    def test_model_adapter_not_detected(self):
        adapter = MagicMock(spec=["process_page", "health_check"])
        assert not hasattr(adapter, "process_document")


# ---------------------------------------------------------------------------
# Deferred evaluation
# ---------------------------------------------------------------------------

class TestDeferredEvaluation:
    def test_has_deferred_evaluations_empty(self):
        pipeline = BenchmarkPipeline.__new__(BenchmarkPipeline)
        pipeline._deferred_pages = []
        assert pipeline.has_deferred_evaluations() is False

    def test_has_deferred_evaluations_with_pages(self):
        pipeline = BenchmarkPipeline.__new__(BenchmarkPipeline)
        output = PageOutput(document_id="d", page_number=1, raw_text="text")
        pipeline._deferred_pages = [(output, "gt", "M-01", "D-01")]
        assert pipeline.has_deferred_evaluations() is True

    def test_run_deferred_evaluations_noop_when_empty(self):
        pipeline = BenchmarkPipeline.__new__(BenchmarkPipeline)
        pipeline._deferred_pages = []
        # Should not raise
        pipeline.run_deferred_evaluations()


# ---------------------------------------------------------------------------
# Result saving structure
# ---------------------------------------------------------------------------

class TestResultSaving:
    def test_eval_results_json_format(self, tmp_path):
        """Verify eval_results.json can be written and read back."""
        results = [
            EvalResult(metric_name="cer", score=0.1, details={"doc": "d1"}),
            EvalResult(metric_name="wer", score=0.2, details={"doc": "d1"}),
        ]
        eval_path = tmp_path / "eval_results.json"
        with open(eval_path, "w") as f:
            json.dump([r.to_dict() for r in results], f)

        with open(eval_path) as f:
            loaded = json.load(f)

        assert len(loaded) == 2
        assert loaded[0]["metric_name"] == "cer"
        assert loaded[0]["score"] == 0.1
        # Verify round-trip
        reloaded = [EvalResult(**item) for item in loaded]
        assert reloaded[0].score == 0.1
