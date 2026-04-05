"""Tests for extractmark.types -- shared data types."""

from pathlib import Path

from extractmark.types import PageInput, PageOutput, EvalResult, RunResult


class TestPageInput:
    def test_create_minimal(self):
        pi = PageInput(document_id="doc1", page_number=0, image_path=Path("/tmp/x.png"))
        assert pi.document_id == "doc1"
        assert pi.page_number == 0
        assert pi.ground_truth is None
        assert pi.metadata == {}

    def test_create_with_ground_truth(self):
        pi = PageInput(
            document_id="doc1", page_number=5,
            image_path=Path("/tmp/x.png"),
            ground_truth="hello world",
            metadata={"lang": "en"},
        )
        assert pi.ground_truth == "hello world"
        assert pi.metadata["lang"] == "en"


class TestPageOutput:
    def test_create_minimal(self):
        po = PageOutput(document_id="doc1", page_number=0, raw_text="text")
        assert po.raw_text == "text"
        assert po.normalized_text == ""
        assert po.tables == []
        assert po.bboxes is None
        assert po.inference_time_ms == 0.0
        assert po.gpu_memory_mb is None

    def test_to_dict(self):
        po = PageOutput(
            document_id="doc1", page_number=3,
            raw_text="hello", normalized_text="hello",
            tables=["| a | b |"], inference_time_ms=42.5,
        )
        d = po.to_dict()
        assert d["document_id"] == "doc1"
        assert d["page_number"] == 3
        assert d["raw_text"] == "hello"
        assert d["tables"] == ["| a | b |"]
        assert d["inference_time_ms"] == 42.5

    def test_to_dict_roundtrip_keys(self):
        po = PageOutput(document_id="d", page_number=0, raw_text="")
        expected_keys = {
            "document_id", "page_number", "raw_text", "normalized_text",
            "tables", "bboxes", "inference_time_ms", "gpu_memory_mb", "metadata",
        }
        assert set(po.to_dict().keys()) == expected_keys


class TestEvalResult:
    def test_create(self):
        er = EvalResult(metric_name="cer", score=0.15)
        assert er.metric_name == "cer"
        assert er.score == 0.15
        assert er.details == {}

    def test_to_dict(self):
        er = EvalResult(metric_name="wer", score=0.3, details={"doc": "d1"})
        d = er.to_dict()
        assert d == {"metric_name": "wer", "score": 0.3, "details": {"doc": "d1"}}


class TestRunResult:
    def test_create_empty(self):
        rr = RunResult(adapter_id="M-01", dataset_id="D-01")
        assert rr.page_outputs == []
        assert rr.eval_results == []
        assert rr.throughput_pages_per_min == 0.0

    def test_create_with_results(self):
        rr = RunResult(
            adapter_id="LIB-01", dataset_id="D-01",
            eval_results=[EvalResult(metric_name="cer", score=0.1)],
            throughput_pages_per_min=50.0,
        )
        assert len(rr.eval_results) == 1
        assert rr.throughput_pages_per_min == 50.0
