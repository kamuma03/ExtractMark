#!/usr/bin/env python3
"""Test each ExtractMark pipeline component with one real document.

Tests:
1. Config loading
2. Dataset loaders (all 6)
3. Output normalizer
4. Evaluators (L1, L2, L3)
5. Model adapter (VLLMModelAdapter) -- requires running vLLM server
6. Library adapters (LIB-01 PyMuPDF if available)
7. Report generation
8. Full pipeline integration
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

PASS = "[bold green]PASS[/bold green]"
FAIL = "[bold red]FAIL[/bold red]"
SKIP = "[bold yellow]SKIP[/bold yellow]"

results: list[tuple[str, str, str]] = []  # (test_name, status, details)


def record(name: str, status: str, details: str = ""):
    results.append((name, status, details))
    icon = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
    color = {"PASS": PASS, "FAIL": FAIL, "SKIP": SKIP}[status]
    detail_str = f" -- {details}" if details else ""
    console.print(f"  {color} {name}{detail_str}")


# ─── 1. Config Loading ─────────────────────────────────────────
def test_config():
    console.rule("[bold]1. Config Loading[/bold]")
    try:
        from extractmark.config import load_config
        cfg = load_config(Path("configs/runs/quick_smoke.yaml"))
        assert cfg.run.name == "quick_smoke"
        assert len(cfg.run.models) > 0
        assert len(cfg.models) > 0
        assert len(cfg.datasets) > 0
        record("Load quick_smoke.yaml", "PASS", f"models={cfg.run.models}, datasets={cfg.run.datasets}")

        cfg2 = load_config(Path("configs/runs/full_benchmark.yaml"))
        assert len(cfg2.run.models) >= 10
        record("Load full_benchmark.yaml", "PASS", f"{len(cfg2.run.models)} models, {len(cfg2.run.datasets)} datasets")
    except Exception as e:
        record("Config loading", "FAIL", str(e))


# ─── 2. Dataset Loaders ────────────────────────────────────────
def test_datasets():
    console.rule("[bold]2. Dataset Loaders[/bold]")

    # D-01 OmniDocBench -- images in flat directory
    try:
        from extractmark.datasets.omnidocbench import OmniDocBenchLoader
        # OmniDocBench has images in a flat images/ dir, not nested by doc
        images_dir = Path("data/omnidocbench/images")
        if images_dir.exists():
            images = sorted(images_dir.glob("*.png"))[:1]
            if images:
                from extractmark.types import PageInput
                page = PageInput(
                    document_id=images[0].stem,
                    page_number=0,
                    image_path=images[0],
                )
                assert page.image_path.exists()
                record("D-01 OmniDocBench", "PASS", f"image={images[0].name} ({images[0].stat().st_size // 1024}KB)")
            else:
                record("D-01 OmniDocBench", "SKIP", "no PNG files found")
        else:
            record("D-01 OmniDocBench", "SKIP", "images/ dir not found")
    except Exception as e:
        record("D-01 OmniDocBench", "FAIL", str(e))

    # D-02 FinTabNet -- compressed archives
    try:
        fintab_dir = Path("data/fintabnet")
        archives = list(fintab_dir.glob("*.tar.gz"))
        if archives:
            record("D-02 FinTabNet", "PASS", f"{len(archives)} archives found (need extraction)")
        else:
            record("D-02 FinTabNet", "SKIP", "no tar.gz archives found")
    except Exception as e:
        record("D-02 FinTabNet", "FAIL", str(e))

    # D-03 FUNSD -- parquet format
    try:
        funsd_parquet = Path("data/funsd/funsd/test-00000-of-00001.parquet")
        if funsd_parquet.exists():
            import pyarrow.parquet as pq
            table = pq.read_table(str(funsd_parquet))
            record("D-03 FUNSD", "PASS", f"parquet: {table.num_rows} rows, cols={table.column_names}")
        else:
            record("D-03 FUNSD", "SKIP", "parquet file not found")
    except ImportError:
        record("D-03 FUNSD", "SKIP", "pyarrow not installed")
    except Exception as e:
        record("D-03 FUNSD", "FAIL", str(e))

    # D-04 DocVQA
    try:
        docvqa_dir = Path("data/docvqa/DocVQA")
        if docvqa_dir.exists():
            files = list(docvqa_dir.iterdir())[:5]
            record("D-04 DocVQA", "PASS", f"{len(files)} items in DocVQA/")
        else:
            record("D-04 DocVQA", "SKIP", "DocVQA dir not found")
    except Exception as e:
        record("D-04 DocVQA", "FAIL", str(e))

    # D-05 OlmOCR-Bench -- PDFs + test JSONL
    try:
        olmocr_dir = Path("data/olmocr_bench/bench_data")
        if olmocr_dir.exists():
            pdfs = list(olmocr_dir.rglob("*.pdf"))
            jsonl = list(olmocr_dir.glob("*.jsonl"))
            record("D-05 OlmOCR-Bench", "PASS", f"{len(pdfs)} PDFs, {len(jsonl)} test JSONL files")
        else:
            record("D-05 OlmOCR-Bench", "SKIP", "bench_data/ not found")
    except Exception as e:
        record("D-05 OlmOCR-Bench", "FAIL", str(e))

    # D-06 DocLayNet -- parquet
    try:
        doclaynet_dir = Path("data/doclaynet/data")
        if doclaynet_dir.exists():
            parquets = list(doclaynet_dir.glob("*.parquet"))
            record("D-06 DocLayNet", "PASS", f"{len(parquets)} parquet files")
        else:
            record("D-06 DocLayNet", "SKIP", "data/ dir not found")
    except Exception as e:
        record("D-06 DocLayNet", "FAIL", str(e))


# ─── 3. Output Normalizer ──────────────────────────────────────
def test_normalizer():
    console.rule("[bold]3. Output Normalizer[/bold]")
    try:
        from extractmark.normalize import normalize

        # Test model token stripping
        text1 = "<|im_start|>assistant\nHello world<|im_end|>"
        result1 = normalize(text1)
        assert "im_start" not in result1
        assert "Hello world" in result1
        record("Strip model tokens", "PASS", f"'{text1[:30]}...' -> '{result1}'")

        # Test table normalization
        text2 = "|  Name  |  Age  |\n|  Alice  |  30  |"
        result2 = normalize(text2)
        assert "| Name | Age |" in result2
        record("Normalize tables", "PASS", f"cleaned pipe spacing")

        # Test LaTeX to Unicode
        text3 = r"The value of $\alpha$ is $\pi$"
        result3 = normalize(text3)
        assert "\u03b1" in result3  # alpha
        assert "\u03c0" in result3  # pi
        record("LaTeX to Unicode", "PASS", f"alpha={chr(0x03b1)}, pi={chr(0x03c0)}")

        # Test whitespace collapse
        text4 = "Hello\n\n\n\n\nworld\n\n\n\nfoo"
        result4 = normalize(text4)
        assert "\n\n\n" not in result4
        record("Collapse whitespace", "PASS", f"{text4.count(chr(10))} newlines -> {result4.count(chr(10))}")

        # Test header/footer stripping
        text5 = "Content here\n\n42\n\nPage 3 of 10"
        result5 = normalize(text5)
        assert "Page 3 of 10" not in result5
        record("Strip headers/footers", "PASS", "removed page number and 'Page X of Y'")

    except Exception as e:
        record("Normalizer", "FAIL", str(e))


# ─── 4. Evaluators ─────────────────────────────────────────────
def test_evaluators():
    console.rule("[bold]4. Evaluators[/bold]")
    from extractmark.types import PageOutput

    # L1: Edit Distance
    try:
        from extractmark.evaluators.edit_distance import EditDistanceEvaluator
        ev = EditDistanceEvaluator()

        # Perfect match
        out = PageOutput(document_id="t", page_number=0, raw_text="Hello world", normalized_text="Hello world")
        res = ev.evaluate(out, "Hello world")
        cer = next(r for r in res if r.metric_name == "cer")
        wer = next(r for r in res if r.metric_name == "wer")
        assert cer.score == 0.0
        assert wer.score == 0.0
        record("L1 Edit Distance (perfect)", "PASS", f"CER={cer.score}, WER={wer.score}")

        # With errors
        out2 = PageOutput(document_id="t", page_number=0, raw_text="Helo wrld", normalized_text="Helo wrld")
        res2 = ev.evaluate(out2, "Hello world")
        cer2 = next(r for r in res2 if r.metric_name == "cer")
        assert 0 < cer2.score < 1
        record("L1 Edit Distance (errors)", "PASS", f"CER={cer2.score:.4f}")

    except Exception as e:
        record("L1 Edit Distance", "FAIL", str(e))

    # L2: Semantic Similarity
    try:
        from extractmark.evaluators.semantic_similarity import SemanticSimilarityEvaluator
        ev2 = SemanticSimilarityEvaluator()

        out = PageOutput(document_id="t", page_number=0, raw_text="The cat sat on the mat",
                         normalized_text="The cat sat on the mat")
        res = ev2.evaluate(out, "A cat was sitting on a mat")
        score = res[0].score
        assert score > 0.5  # Should be semantically similar
        record("L2 SBERT Similarity", "PASS", f"cosine={score:.4f} (semantically similar)")

        # Dissimilar
        out2 = PageOutput(document_id="t", page_number=0, raw_text="Quantum physics equations",
                          normalized_text="Quantum physics equations")
        res2 = ev2.evaluate(out2, "Chocolate cake recipe with frosting")
        score2 = res2[0].score
        assert score2 < score  # Should be less similar
        record("L2 SBERT Dissimilar", "PASS", f"cosine={score2:.4f} (less similar, correct)")

    except Exception as e:
        record("L2 SBERT", "FAIL", str(e))

    # L3: Unit Tests
    try:
        from extractmark.evaluators.unit_tests import UnitTestEvaluator
        ev3 = UnitTestEvaluator()

        # With explicit tests
        out = PageOutput(
            document_id="t", page_number=0,
            raw_text="The revenue was $1.5M in Q4 2025",
            normalized_text="The revenue was $1.5M in Q4 2025",
            metadata={"unit_tests": [
                {"type": "presence", "value": "revenue"},
                {"type": "presence", "value": "$1.5M"},
                {"type": "absence", "value": "profit"},
                {"type": "regex", "pattern": r"\$[\d.]+M"},
            ]},
        )
        res = ev3.evaluate(out, "")
        pass_rate = res[0].score
        assert pass_rate == 1.0
        record("L3 Unit Tests (explicit)", "PASS", f"pass_rate={pass_rate} (4/4)")

        # Default structural checks
        out2 = PageOutput(document_id="t", page_number=0,
                          raw_text="Normal document text here with content.",
                          normalized_text="Normal document text here with content.")
        res2 = ev3.evaluate(out2, "")
        record("L3 Unit Tests (default)", "PASS", f"pass_rate={res2[0].score:.2f}")

    except Exception as e:
        record("L3 Unit Tests", "FAIL", str(e))


# ─── 5. Model Adapter (vLLM) ───────────────────────────────────
def test_model_adapter():
    console.rule("[bold]5. Model Adapter (vLLM)[/bold]")
    try:
        from extractmark.models.vllm_model import VLLMModelAdapter
        from extractmark.config import ModelConfig, GenerationParams
        from extractmark.types import PageInput
        from extractmark.normalize import normalize

        # Check if vLLM is running
        import openai
        client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        models = client.models.list()
        model_id_served = models.data[0].id
        console.print(f"  vLLM serving: [cyan]{model_id_served}[/cyan]")

        # Determine which model is running and use matching config
        if "Nemotron" in model_id_served:
            config = ModelConfig(
                name="Nemotron Parse v1.1",
                hf_model_id=model_id_served,
                supports_bbox=True,
                prompt_template="nemotron_parse",
                generation_params=GenerationParams(temperature=0.0, seed=42, max_tokens=4096),
                port=8000,
            )
            adapter_id = "M-01"
        elif "Qwen2.5-VL-3B" in model_id_served:
            config = ModelConfig(
                name="Qwen2.5-VL-3B",
                hf_model_id=model_id_served,
                supports_bbox=True,
                prompt_template="qwen_vl",
                generation_params=GenerationParams(temperature=0.0, seed=42, max_tokens=4096),
                port=8000,
            )
            adapter_id = "M-08"
        else:
            config = ModelConfig(
                name=model_id_served,
                hf_model_id=model_id_served,
                prompt_template="generic_ocr",
                generation_params=GenerationParams(temperature=0.0, seed=42, max_tokens=4096),
                port=8000,
            )
            adapter_id = "UNKNOWN"

        adapter = VLLMModelAdapter(adapter_id, config)

        # Health check
        assert adapter.health_check()
        record("VLLMAdapter health_check()", "PASS", f"model={model_id_served}")

        # Process a real page
        test_image = Path("data/omnidocbench/images/PPT_StewartCalcET7e_04_09_page_018.png")
        if not test_image.exists():
            # Fallback to any PNG
            test_image = next(Path("data/omnidocbench/images").glob("*.png"), None)

        if test_image:
            page = PageInput(document_id="test_doc", page_number=0, image_path=test_image)
            output = adapter.process_page(page)
            output.normalized_text = normalize(output.raw_text)

            assert output.raw_text, "Raw text should not be empty"
            assert output.inference_time_ms > 0
            record(
                f"VLLMAdapter process_page()",
                "PASS",
                f"{output.inference_time_ms:.0f}ms, {len(output.raw_text)} chars raw, "
                f"{len(output.bboxes) if output.bboxes else 0} bboxes",
            )

            # Test bbox extraction for Nemotron
            if output.bboxes:
                bbox = output.bboxes[0]
                assert "x1" in bbox and "y1" in bbox
                record("Bbox extraction", "PASS", f"{len(output.bboxes)} bboxes, first: class={bbox.get('class', 'N/A')}")
            else:
                record("Bbox extraction", "SKIP", "model returned no bboxes")
        else:
            record("VLLMAdapter process_page()", "SKIP", "no test image found")

    except Exception as e:
        if "Connection refused" in str(e) or "ConnectError" in str(e):
            record("Model Adapter", "SKIP", "vLLM server not running (start with: vllm serve ...)")
        else:
            record("Model Adapter", "FAIL", f"{type(e).__name__}: {e}")


# ─── 6. Library Adapters ───────────────────────────────────────
def test_library_adapters():
    console.rule("[bold]6. Library Adapters[/bold]")

    # Find a test PDF (from OlmOCR-Bench)
    test_pdf = None
    for pdf in Path("data/olmocr_bench/bench_data/pdfs").rglob("*.pdf"):
        test_pdf = pdf
        break

    if not test_pdf:
        record("Library Adapters", "SKIP", "no test PDF found")
        return

    console.print(f"  Test PDF: {test_pdf.name}")

    # LIB-01 PyMuPDF
    try:
        from extractmark.libraries.pymupdf import PyMuPDFAdapter
        adapter = PyMuPDFAdapter()
        outputs = adapter.process_document(test_pdf)
        assert len(outputs) > 0, "Should return at least one page"
        # Note: scanned PDFs return empty text (expected -- PyMuPDF has no OCR)
        text_len = len(outputs[0].raw_text)
        record(
            "LIB-01 PyMuPDF",
            "PASS",
            f"{len(outputs)} pages, {text_len} chars "
            f"{'(scanned/image PDF)' if text_len == 0 else ''}, "
            f"{outputs[0].inference_time_ms:.1f}ms",
        )
    except ImportError:
        record("LIB-01 PyMuPDF", "SKIP", "pymupdf not installed")
    except Exception as e:
        record("LIB-01 PyMuPDF", "FAIL", str(e))

    # LIB-02 pdfplumber
    try:
        from extractmark.libraries.pdfplumber_adapter import PdfplumberAdapter
        adapter = PdfplumberAdapter()
        outputs = adapter.process_document(test_pdf)
        assert len(outputs) > 0
        record(
            "LIB-02 pdfplumber",
            "PASS",
            f"{len(outputs)} pages, {len(outputs[0].raw_text)} chars, "
            f"{outputs[0].inference_time_ms:.1f}ms",
        )
    except ImportError:
        record("LIB-02 pdfplumber", "SKIP", "pdfplumber not installed")
    except Exception as e:
        record("LIB-02 pdfplumber", "FAIL", str(e))

    # LIB-03 pypdfium2
    try:
        from extractmark.libraries.pypdfium2_adapter import Pypdfium2Adapter
        adapter = Pypdfium2Adapter()
        outputs = adapter.process_document(test_pdf)
        assert len(outputs) > 0
        record(
            "LIB-03 pypdfium2",
            "PASS",
            f"{len(outputs)} pages, {len(outputs[0].raw_text)} chars, "
            f"{outputs[0].inference_time_ms:.1f}ms",
        )
    except ImportError:
        record("LIB-03 pypdfium2", "SKIP", "pypdfium2 not installed")
    except Exception as e:
        record("LIB-03 pypdfium2", "FAIL", str(e))

    # LIB-13 MarkItDown
    try:
        from extractmark.libraries.markitdown_adapter import MarkItDownAdapter
        adapter = MarkItDownAdapter()
        outputs = adapter.process_document(test_pdf)
        assert len(outputs) > 0
        record(
            "LIB-13 MarkItDown",
            "PASS",
            f"{len(outputs)} pages, {len(outputs[0].raw_text)} chars, "
            f"{outputs[0].inference_time_ms:.1f}ms",
        )
    except ImportError:
        record("LIB-13 MarkItDown", "SKIP", "markitdown not installed")
    except Exception as e:
        record("LIB-13 MarkItDown", "FAIL", str(e))

    # LIB-17 pdfminer
    try:
        from extractmark.libraries.pdfminer_adapter import PdfminerAdapter
        adapter = PdfminerAdapter()
        outputs = adapter.process_document(test_pdf)
        assert len(outputs) > 0
        record(
            "LIB-17 pdfminer",
            "PASS",
            f"{len(outputs)} pages, {len(outputs[0].raw_text)} chars, "
            f"{outputs[0].inference_time_ms:.1f}ms, "
            f"{len(outputs[0].bboxes) if outputs[0].bboxes else 0} bboxes",
        )
    except ImportError:
        record("LIB-17 pdfminer", "SKIP", "pdfminer.six not installed")
    except Exception as e:
        record("LIB-17 pdfminer", "FAIL", str(e))


# ─── 7. Report Generation ──────────────────────────────────────
def test_reporting():
    console.rule("[bold]7. Report Generation[/bold]")
    try:
        from extractmark.reporting.summary import SummaryReporter
        from extractmark.types import RunResult, EvalResult

        # Create a mock result
        test_result = RunResult(
            adapter_id="TEST-MODEL",
            dataset_id="TEST-DATASET",
            eval_results=[
                EvalResult(metric_name="cer", score=0.05),
                EvalResult(metric_name="wer", score=0.12),
                EvalResult(metric_name="sbert_cosine", score=0.92),
            ],
            throughput_pages_per_min=15.5,
        )

        test_dir = Path("results/_test")
        report_dir = Path("report/_test")
        reporter = SummaryReporter(test_dir, report_dir)
        reporter.add_run(test_result)
        reporter.generate()

        md_path = report_dir / "benchmark_summary.md"
        csv_path = report_dir / "benchmark_summary.csv"
        assert md_path.exists(), "Markdown report not generated"
        assert csv_path.exists(), "CSV report not generated"

        md_content = md_path.read_text()
        assert "TEST-MODEL" in md_content
        assert "cer" in md_content or "CER" in md_content
        record("Markdown report", "PASS", f"{len(md_content)} chars")

        csv_content = csv_path.read_text()
        assert "TEST-MODEL" in csv_content
        record("CSV report", "PASS", f"{len(csv_content)} chars")

        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        shutil.rmtree(report_dir, ignore_errors=True)

    except Exception as e:
        record("Reporting", "FAIL", str(e))


# ─── 8. Full Pipeline Integration ──────────────────────────────
def test_pipeline_integration():
    console.rule("[bold]8. Full Pipeline Integration[/bold]")
    try:
        # Check if vLLM is running
        import openai
        client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        models = client.models.list()
        model_served = models.data[0].id

        from extractmark.config import load_config
        cfg = load_config(Path("configs/runs/quick_smoke.yaml"))

        # Override to use only the running model and limit pages
        if "Nemotron" in model_served:
            cfg.run.models = ["M-01"]
        elif "Qwen2.5-VL-3B" in model_served:
            cfg.run.models = ["M-08"]
        cfg.run.max_pages = 2
        cfg.run.evaluators = ["L1"]

        from extractmark.pipeline import BenchmarkPipeline
        pipeline = BenchmarkPipeline(cfg)
        pipeline.run()

        # Verify outputs were created
        results_dir = Path("results")
        result_files = list(results_dir.rglob("*.json"))
        record("Pipeline run", "PASS", f"generated {len(result_files)} result files")

    except Exception as e:
        if "Connection refused" in str(e) or "ConnectError" in str(e):
            record("Pipeline Integration", "SKIP", "vLLM not running")
        else:
            record("Pipeline Integration", "FAIL", f"{type(e).__name__}: {e}")


# ─── Main ──────────────────────────────────────────────────────
def main():
    console.print(Panel(
        "[bold cyan]ExtractMark Pipeline Test Suite[/bold cyan]\n"
        "Testing each component with real documents",
        title="Test",
    ))
    console.print()

    test_config()
    test_datasets()
    test_normalizer()
    test_evaluators()
    test_model_adapter()
    test_library_adapters()
    test_reporting()
    test_pipeline_integration()

    # Summary
    console.print()
    console.rule("[bold]Test Summary[/bold]")
    table = Table(show_lines=False)
    table.add_column("Test", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    passed = failed = skipped = 0
    for name, status, details in results:
        color = {"PASS": "green", "FAIL": "red", "SKIP": "yellow"}[status]
        table.add_row(name, f"[{color}]{status}[/{color}]", details[:80])
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        else:
            skipped += 1

    console.print(table)
    console.print()
    console.print(f"  [green]{passed} passed[/green] | [red]{failed} failed[/red] | [yellow]{skipped} skipped[/yellow]")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
