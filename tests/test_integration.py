"""Integration tests using REAL data, REAL models, and REAL libraries.

Requirements:
- vLLM server running M-01 (nvidia/NVIDIA-Nemotron-Parse-v1.1) on port 8000
- OmniDocBench dataset in data/omnidocbench/ with images and annotations
- pdfplumber, pdfminer installed

These tests are slow (real inference) -- run with:
    python -m pytest tests/test_integration.py -v -s
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import openai

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/omnidocbench")
IMAGES_DIR = DATA_ROOT / "images"
VLLM_URL = "http://localhost:8000/v1"

def _vllm_available() -> bool:
    try:
        client = openai.OpenAI(base_url=VLLM_URL, api_key="not-needed")
        client.models.list()
        return True
    except Exception:
        return False

def _omnidocbench_available() -> bool:
    return IMAGES_DIR.exists() and any(IMAGES_DIR.glob("*.png"))

def _lib_available(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False

skip_no_vllm = pytest.mark.skipif(not _vllm_available(), reason="vLLM server not running on port 8000")
skip_no_data = pytest.mark.skipif(not _omnidocbench_available(), reason="OmniDocBench dataset not available")
skip_no_pdfplumber = pytest.mark.skipif(not _lib_available("pdfplumber"), reason="pdfplumber not installed")
skip_no_pdfminer = pytest.mark.skipif(not _lib_available("pdfminer"), reason="pdfminer not installed")
skip_no_jiwer = pytest.mark.skipif(not _lib_available("jiwer"), reason="jiwer not installed")
skip_no_sbert = pytest.mark.skipif(not _lib_available("sentence_transformers"), reason="sentence_transformers not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def omnidocbench_loader():
    from extractmark.datasets.omnidocbench import OmniDocBenchLoader
    return OmniDocBenchLoader("D-01", DATA_ROOT)


@pytest.fixture(scope="module")
def sample_pages(omnidocbench_loader):
    """Load real OmniDocBench pages that are known text-heavy documents.

    Skips chart/image-only pages (e.g. eastmoney financial charts) and
    selects pages from data sources that reliably produce OCR output.
    """
    pages = []
    # Prefer academic papers and PPTs which are text-heavy
    good_sources = {"academic_literature", "PPT2PDF", "textbook"}
    for page in omnidocbench_loader.load():
        if not page.ground_truth or len(page.ground_truth) < 100:
            continue
        src = page.metadata.get("data_source", "")
        # Skip eastmoney (financial charts) and similar image-heavy sources
        if "eastmoney" in page.document_id:
            continue
        if src in good_sources or "scihub" in page.document_id or "book" in page.document_id:
            pages.append(page)
            if len(pages) >= 5:
                break
    assert len(pages) >= 3, "Need at least 3 text-heavy pages for integration tests"
    return pages


@pytest.fixture(scope="module")
def vllm_client():
    return openai.OpenAI(base_url=VLLM_URL, api_key="not-needed")


# ===========================================================================
# 1. Dataset loading with REAL data
# ===========================================================================

@skip_no_data
class TestRealDatasetLoading:
    def test_omnidocbench_loads_pages(self, omnidocbench_loader):
        pages = list(omnidocbench_loader.load())
        assert len(pages) > 100, f"Expected 400+ pages, got {len(pages)}"

    def test_omnidocbench_pages_have_images(self, sample_pages):
        for page in sample_pages:
            assert page.image_path.exists(), f"Image not found: {page.image_path}"
            assert page.image_path.stat().st_size > 1000, "Image too small"

    def test_omnidocbench_pages_have_ground_truth(self, sample_pages):
        gt_count = sum(1 for p in sample_pages if p.ground_truth)
        assert gt_count >= 2, f"Expected most pages to have GT, only {gt_count}/3 do"

    def test_omnidocbench_ground_truth_is_text(self, sample_pages):
        for page in sample_pages:
            if page.ground_truth:
                assert len(page.ground_truth) > 10, "Ground truth too short"
                assert isinstance(page.ground_truth, str)

    def test_omnidocbench_metadata_populated(self, sample_pages):
        for page in sample_pages:
            assert "dataset" in page.metadata


# ===========================================================================
# 2. Real vLLM model inference (M-01 Nemotron Parse)
# ===========================================================================

@skip_no_vllm
@skip_no_data
class TestRealModelInference:
    def test_vllm_health_check(self, vllm_client):
        resp = vllm_client.models.list()
        assert len(resp.data) > 0
        model_name = resp.data[0].id
        assert "Nemotron" in model_name or "nvidia" in model_name.lower()

    def test_vllm_model_adapter_process_page(self, sample_pages):
        """Process a real page through VLLMModelAdapter and get OCR output."""
        from extractmark.config import ModelConfig
        from extractmark.models.vllm_model import VLLMModelAdapter

        cfg = ModelConfig(
            name="Nemotron Parse",
            hf_model_id="nvidia/NVIDIA-Nemotron-Parse-v1.1",
            prompt_template="nemotron_parse",
            supports_bbox=True,
        )
        adapter = VLLMModelAdapter("M-01", cfg)

        assert adapter.health_check(), "vLLM health check failed"

        page = sample_pages[0]
        start = time.time()
        output = adapter.process_page(page)
        elapsed = time.time() - start

        assert output.document_id == page.document_id
        assert output.page_number == page.page_number
        assert len(output.raw_text) > 50, f"Expected substantial OCR output, got {len(output.raw_text)} chars"
        assert output.inference_time_ms > 0
        assert elapsed < 120, f"Inference took {elapsed:.1f}s, expected < 120s"
        print(f"\n  OCR output: {len(output.raw_text)} chars in {output.inference_time_ms:.0f}ms")
        print(f"  First 200 chars: {output.raw_text[:200]}")

    def test_vllm_extracts_bboxes(self, sample_pages):
        """Nemotron Parse should produce bounding boxes."""
        from extractmark.config import ModelConfig
        from extractmark.models.vllm_model import VLLMModelAdapter

        cfg = ModelConfig(
            name="Nemotron Parse",
            hf_model_id="nvidia/NVIDIA-Nemotron-Parse-v1.1",
            prompt_template="nemotron_parse",
            supports_bbox=True,
        )
        adapter = VLLMModelAdapter("M-01", cfg)
        output = adapter.process_page(sample_pages[0])

        assert output.bboxes is not None, "Expected bboxes from Nemotron Parse"
        assert len(output.bboxes) > 0, "Expected at least one bbox"
        bbox = output.bboxes[0]
        assert "x1" in bbox and "y1" in bbox
        assert "class" in bbox, "Nemotron bboxes should include class labels"
        print(f"\n  Extracted {len(output.bboxes)} bboxes, first: {bbox}")

    def test_vllm_multiple_pages(self, sample_pages):
        """Process multiple pages and verify consistent results."""
        from extractmark.config import ModelConfig
        from extractmark.models.vllm_model import VLLMModelAdapter

        cfg = ModelConfig(
            name="Nemotron Parse",
            hf_model_id="nvidia/NVIDIA-Nemotron-Parse-v1.1",
            prompt_template="nemotron_parse",
            supports_bbox=True,
        )
        adapter = VLLMModelAdapter("M-01", cfg)

        outputs = []
        for page in sample_pages[:3]:
            output = adapter.process_page(page)
            outputs.append(output)

        non_empty = [o for o in outputs if len(o.raw_text) > 0]
        assert len(non_empty) >= 2, f"Expected most pages to produce output, got {len(non_empty)}/{len(outputs)}"
        print(f"\n  Processed {len(outputs)} pages: " +
              ", ".join(f"{len(o.raw_text)} chars" for o in outputs))


# ===========================================================================
# 3. Real normalization on model output
# ===========================================================================

@skip_no_vllm
@skip_no_data
class TestRealNormalization:
    def test_normalize_real_ocr_output(self, sample_pages):
        from extractmark.config import ModelConfig
        from extractmark.models.vllm_model import VLLMModelAdapter
        from extractmark.normalize import normalize

        cfg = ModelConfig(
            name="Nemotron Parse",
            hf_model_id="nvidia/NVIDIA-Nemotron-Parse-v1.1",
            prompt_template="nemotron_parse",
        )
        adapter = VLLMModelAdapter("M-01", cfg)
        output = adapter.process_page(sample_pages[0])
        raw = output.raw_text

        normalized = normalize(raw)
        assert len(normalized) > 0, "Normalization produced empty output"
        # Nemotron bbox tokens should be stripped or handled
        assert "<|im_start|>" not in normalized
        assert "<|im_end|>" not in normalized
        print(f"\n  Raw: {len(raw)} chars -> Normalized: {len(normalized)} chars")
        print(f"  Normalized (first 200): {normalized[:200]}")


# ===========================================================================
# 4. Real library extraction via pipeline routing
# ===========================================================================

@skip_no_data
@skip_no_pdfplumber
class TestRealPdfplumberExtraction:
    def test_pdfplumber_via_pipeline_routing(self, sample_pages):
        """Test the full image -> temp PDF -> pdfplumber -> text pipeline."""
        from extractmark.libraries.pdfplumber_adapter import PdfplumberAdapter
        from extractmark.pipeline import BenchmarkPipeline

        adapter = PdfplumberAdapter()
        page = sample_pages[0]

        result = BenchmarkPipeline._process_with_document_adapter(adapter, page)

        assert result.document_id == page.document_id
        assert result.page_number == page.page_number
        assert result.inference_time_ms > 0
        # pdfplumber on an image-to-PDF won't extract OCR text (it's raster),
        # but it should NOT crash and should return a valid PageOutput
        print(f"\n  pdfplumber output: {len(result.raw_text)} chars, {result.inference_time_ms:.1f}ms")
        if result.raw_text:
            print(f"  Text: {result.raw_text[:200]}")
        else:
            print("  (empty - expected for raster image PDF without embedded text)")


@skip_no_data
@skip_no_pdfminer
class TestRealPdfminerExtraction:
    def test_pdfminer_via_pipeline_routing(self, sample_pages):
        """Test pdfminer extraction through the pipeline routing."""
        from extractmark.libraries.pdfminer_adapter import PdfminerAdapter
        from extractmark.pipeline import BenchmarkPipeline

        adapter = PdfminerAdapter()
        page = sample_pages[0]

        result = BenchmarkPipeline._process_with_document_adapter(adapter, page)

        assert result.document_id == page.document_id
        assert result.inference_time_ms > 0
        print(f"\n  pdfminer output: {len(result.raw_text)} chars, {result.inference_time_ms:.1f}ms")


# ===========================================================================
# 5. Real evaluators on real model output
# ===========================================================================

@skip_no_vllm
@skip_no_data
class TestRealEvaluators:
    @pytest.fixture(scope="class")
    def real_ocr_output(self, sample_pages):
        """Get real OCR output from M-01 for evaluation."""
        from extractmark.config import ModelConfig
        from extractmark.models.vllm_model import VLLMModelAdapter
        from extractmark.normalize import normalize

        cfg = ModelConfig(
            name="Nemotron Parse",
            hf_model_id="nvidia/NVIDIA-Nemotron-Parse-v1.1",
            prompt_template="nemotron_parse",
        )
        adapter = VLLMModelAdapter("M-01", cfg)
        output = adapter.process_page(sample_pages[0])
        output.normalized_text = normalize(output.raw_text)
        return output, sample_pages[0].ground_truth

    def test_l3_unit_test_evaluator(self, real_ocr_output):
        """L3 default structural checks on real OCR output."""
        from extractmark.evaluators.unit_tests import UnitTestEvaluator

        output, gt = real_ocr_output
        ev = UnitTestEvaluator()
        results = ev.evaluate(output, gt)

        assert len(results) == 1
        assert results[0].metric_name == "unit_test_pass_rate"
        score = results[0].score
        assert 0 <= score <= 1
        print(f"\n  L3 pass rate: {score:.2f} ({results[0].details})")

    @skip_no_jiwer
    def test_l1_edit_distance(self, real_ocr_output):
        """L1 CER/WER on real OCR output vs ground truth."""
        from extractmark.evaluators.edit_distance import EditDistanceEvaluator

        output, gt = real_ocr_output
        ev = EditDistanceEvaluator()
        results = ev.evaluate(output, gt)

        cer = [r for r in results if r.metric_name == "cer"][0]
        wer = [r for r in results if r.metric_name == "wer"][0]
        assert 0 <= cer.score  # CER can be > 1 for insertions
        assert 0 <= wer.score
        print(f"\n  L1 CER: {cer.score:.4f}, WER: {wer.score:.4f}")

    @skip_no_sbert
    def test_l2_semantic_similarity(self, real_ocr_output):
        """L2 SBERT cosine similarity on real OCR output."""
        from extractmark.evaluators.semantic_similarity import SemanticSimilarityEvaluator

        output, gt = real_ocr_output
        ev = SemanticSimilarityEvaluator()
        results = ev.evaluate(output, gt)

        assert len(results) == 1
        cosine = results[0].score
        assert 0 <= cosine <= 1
        print(f"\n  L2 SBERT cosine: {cosine:.4f}")


# ===========================================================================
# 6. Real config loading
# ===========================================================================

class TestRealConfig:
    def test_load_full_benchmark_config(self):
        from extractmark.config import load_config
        cfg_path = Path("configs/runs/full_benchmark.yaml")
        if not cfg_path.exists():
            pytest.skip("full_benchmark.yaml not found")

        cfg = load_config(cfg_path)
        assert len(cfg.models) >= 10, f"Expected 10+ models, got {len(cfg.models)}"
        assert len(cfg.libraries) >= 10, f"Expected 10+ libraries, got {len(cfg.libraries)}"
        assert len(cfg.datasets) >= 5, f"Expected 5+ datasets, got {len(cfg.datasets)}"
        assert "M-01" in cfg.models
        assert cfg.models["M-01"].hf_model_id == "nvidia/NVIDIA-Nemotron-Parse-v1.1"
        print(f"\n  Models: {len(cfg.models)}, Libraries: {len(cfg.libraries)}, Datasets: {len(cfg.datasets)}")

    def test_load_quick_smoke_config(self):
        from extractmark.config import load_config
        cfg_path = Path("configs/runs/quick_smoke.yaml")
        if not cfg_path.exists():
            pytest.skip("quick_smoke.yaml not found")

        cfg = load_config(cfg_path)
        assert cfg.run.name


# ===========================================================================
# 7. End-to-end: real model + real data + real eval + report
# ===========================================================================

@skip_no_vllm
@skip_no_data
class TestEndToEnd:
    def test_single_page_full_pipeline(self, sample_pages):
        """Full pipeline: real image -> M-01 inference -> normalize -> L3 eval -> result."""
        from extractmark.config import ModelConfig
        from extractmark.models.vllm_model import VLLMModelAdapter
        from extractmark.normalize import normalize
        from extractmark.evaluators.unit_tests import UnitTestEvaluator

        # 1. Model inference
        cfg = ModelConfig(
            name="Nemotron Parse",
            hf_model_id="nvidia/NVIDIA-Nemotron-Parse-v1.1",
            prompt_template="nemotron_parse",
            supports_bbox=True,
        )
        adapter = VLLMModelAdapter("M-01", cfg)
        page = sample_pages[0]
        output = adapter.process_page(page)
        assert len(output.raw_text) > 0, "Model returned empty text"

        # 2. Normalize
        output.normalized_text = normalize(output.raw_text)
        assert len(output.normalized_text) > 0

        # 3. Evaluate
        gt = page.ground_truth
        assert gt, "No ground truth for test page"

        ev = UnitTestEvaluator()
        results = ev.evaluate(output, gt)
        assert results[0].metric_name == "unit_test_pass_rate"

        print(f"\n  === End-to-end result ===")
        print(f"  Page: {page.document_id} (page {page.page_number})")
        print(f"  Raw text: {len(output.raw_text)} chars")
        print(f"  Normalized: {len(output.normalized_text)} chars")
        print(f"  Bboxes: {len(output.bboxes) if output.bboxes else 0}")
        print(f"  L3 pass rate: {results[0].score:.2f}")
        print(f"  Ground truth: {len(gt)} chars")

    def test_report_generation_from_real_run(self, sample_pages, tmp_path):
        """Generate a report from a real single-model run."""
        from extractmark.config import ModelConfig
        from extractmark.models.vllm_model import VLLMModelAdapter
        from extractmark.normalize import normalize
        from extractmark.evaluators.unit_tests import UnitTestEvaluator
        from extractmark.types import RunResult, EvalResult
        from extractmark.reporting.summary import SummaryReporter

        cfg = ModelConfig(
            name="Nemotron Parse",
            hf_model_id="nvidia/NVIDIA-Nemotron-Parse-v1.1",
            prompt_template="nemotron_parse",
        )
        adapter = VLLMModelAdapter("M-01", cfg)
        ev = UnitTestEvaluator()

        all_results = []
        outputs = []
        for page in sample_pages[:2]:
            output = adapter.process_page(page)
            output.normalized_text = normalize(output.raw_text)
            outputs.append(output)

            if page.ground_truth:
                eval_res = ev.evaluate(output, page.ground_truth)
                all_results.extend(eval_res)

        # Build RunResult
        throughput = len(outputs) / (sum(o.inference_time_ms for o in outputs) / 60000)
        run = RunResult(
            adapter_id="M-01",
            dataset_id="D-01",
            page_outputs=outputs,
            eval_results=all_results,
            throughput_pages_per_min=throughput,
        )

        # Generate report
        report_dir = tmp_path / "report"
        reporter = SummaryReporter(tmp_path / "results", report_dir)
        reporter.add_run(run)
        reporter.generate()

        md = (report_dir / "benchmark_summary.md").read_text()
        assert "M-01" in md
        assert "D-01" in md
        assert "unit_test_pass_rate" in md

        csv_path = report_dir / "benchmark_summary.csv"
        assert csv_path.exists()

        print(f"\n  Generated report:\n{md}")
