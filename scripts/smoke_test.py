#!/usr/bin/env python3
"""Smoke test for all models and libraries in the benchmark.

Tests:
  1. Dataset loaders: can load and yield pages from each dataset
  2. Library adapters: can import, instantiate, and process 2 sample pages per dataset
  3. Model adapters: can instantiate, build prompts (no inference -- requires vLLM)
  4. Evaluators: can instantiate and run on dummy data

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --libs-only     # skip model vLLM startup checks
    python scripts/smoke_test.py --models-only    # only test model serving
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

from rich.console import Console
from rich.table import Table

from extractmark.config import load_config
from extractmark.types import PageInput, PageOutput

console = Console()

# Full benchmark config
BENCHMARK_LIBS = ["LIB-01", "LIB-02", "LIB-08", "LIB-09", "LIB-10", "LIB-12", "LIB-13"]
BENCHMARK_MODELS = ["M-01", "M-02", "M-04", "M-06", "M-07", "M-08", "M-09", "M-10", "M-11", "M-12", "M-13"]
BENCHMARK_DATASETS = ["D-01", "D-02", "D-03", "D-04", "D-05", "D-06"]
PAGES_PER_DATASET = 2


def load_sample_pages(config, dataset_id: str, n: int = 2) -> list[PageInput]:
    """Load first N pages from a dataset."""
    from extractmark.datasets.registry import get_dataset

    ds_config = config.datasets[dataset_id]
    loader = get_dataset(dataset_id, ds_config)
    pages = list(itertools.islice(loader.load(), n))
    return pages


def test_datasets(config) -> dict[str, dict]:
    """Test that each dataset loader can load pages."""
    console.print("\n[bold cyan]═══ Testing Dataset Loaders ═══[/bold cyan]")
    results = {}
    for ds_id in BENCHMARK_DATASETS:
        try:
            pages = load_sample_pages(config, ds_id, PAGES_PER_DATASET)
            if not pages:
                results[ds_id] = {"status": "WARN", "detail": "No pages returned"}
            else:
                # Verify page structure
                p = pages[0]
                assert p.image_path.exists(), f"Image not found: {p.image_path}"
                results[ds_id] = {
                    "status": "OK",
                    "detail": f"{len(pages)} pages, image={p.image_path.name}",
                }
        except Exception as e:
            results[ds_id] = {"status": "FAIL", "detail": f"{type(e).__name__}: {e}"}
            traceback.print_exc()

    _print_results("Dataset", results)
    return results


def test_libraries(config, sample_pages: dict[str, list[PageInput]]) -> dict[str, dict]:
    """Test each library adapter can process sample pages."""
    from extractmark.libraries.registry import get_library

    console.print("\n[bold cyan]═══ Testing Library Adapters ═══[/bold cyan]")
    results = {}

    for lib_id in BENCHMARK_LIBS:
        lib_results = {}
        try:
            lib_config = config.libraries[lib_id]
            adapter = get_library(lib_id, lib_config)
            console.print(f"  [dim]{lib_id}[/dim] ({type(adapter).__name__}) instantiated OK")
        except Exception as e:
            results[lib_id] = {"status": "FAIL", "detail": f"Import/init: {type(e).__name__}: {e}"}
            traceback.print_exc()
            continue

        # Test on a couple of pages from each dataset
        for ds_id, pages in sample_pages.items():
            if not pages:
                continue
            for page in pages[:1]:  # just 1 page per dataset per lib
                label = f"{ds_id}/{page.document_id}"
                try:
                    # Libraries use process_document with a temp PDF
                    from PIL import Image
                    img = Image.open(page.image_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                        img.save(f, "PDF", resolution=150)
                        pdf_path = Path(f.name)

                    try:
                        start = time.perf_counter()
                        outputs = adapter.process_document(pdf_path)
                        elapsed = time.perf_counter() - start
                    finally:
                        pdf_path.unlink(missing_ok=True)

                    if outputs and outputs[0].raw_text.strip():
                        lib_results[label] = {
                            "status": "OK",
                            "detail": f"{len(outputs[0].raw_text)} chars, {elapsed:.1f}s",
                        }
                    elif outputs:
                        lib_results[label] = {
                            "status": "WARN",
                            "detail": f"Empty text, {elapsed:.1f}s",
                        }
                    else:
                        lib_results[label] = {
                            "status": "WARN",
                            "detail": f"No outputs returned, {elapsed:.1f}s",
                        }
                except Exception as e:
                    lib_results[label] = {
                        "status": "FAIL",
                        "detail": f"{type(e).__name__}: {e}",
                    }
                    traceback.print_exc()

        # Aggregate
        statuses = [v["status"] for v in lib_results.values()]
        fails = statuses.count("FAIL")
        warns = statuses.count("WARN")
        oks = statuses.count("OK")

        if fails > 0:
            detail_lines = [f"{k}: {v['detail']}" for k, v in lib_results.items() if v["status"] == "FAIL"]
            results[lib_id] = {
                "status": "FAIL",
                "detail": f"{fails} fail, {warns} warn, {oks} ok | " + "; ".join(detail_lines[:3]),
            }
        elif warns > 0:
            results[lib_id] = {
                "status": "WARN",
                "detail": f"{warns} warn, {oks} ok (some pages returned empty text)",
            }
        else:
            results[lib_id] = {
                "status": "OK",
                "detail": f"All {oks} pages processed successfully",
            }

    _print_results("Library", results)
    return results


def test_model_configs(config) -> dict[str, dict]:
    """Test model config loading, prompt template resolution, and adapter instantiation.

    Does NOT start vLLM or run inference -- just verifies the adapter can be created
    and the prompt template can build messages.
    """
    from extractmark.models.registry import get_model
    from extractmark.models.prompt_templates import get_template

    console.print("\n[bold cyan]═══ Testing Model Configs & Prompt Templates ═══[/bold cyan]")
    results = {}

    for model_id in BENCHMARK_MODELS:
        try:
            model_config = config.models[model_id]
            # Test prompt template exists and can build messages
            template_fn = get_template(model_config.prompt_template)
            # Build a dummy message to verify template works
            dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            messages = template_fn(dummy_b64)
            assert isinstance(messages, list) and len(messages) > 0, "Template returned empty messages"
            assert messages[0].get("role"), "Message missing role"

            # Test adapter instantiation (won't connect to server)
            adapter = get_model(model_id, model_config)
            assert adapter.model_id == model_id

            results[model_id] = {
                "status": "OK",
                "detail": f"template={model_config.prompt_template}, hf={model_config.hf_model_id}",
            }
        except Exception as e:
            results[model_id] = {"status": "FAIL", "detail": f"{type(e).__name__}: {e}"}
            traceback.print_exc()

    _print_results("Model Config", results)
    return results


def test_evaluators(config) -> dict[str, dict]:
    """Test evaluator instantiation and basic operation on dummy data."""
    console.print("\n[bold cyan]═══ Testing Evaluators ═══[/bold cyan]")
    results = {}

    # L1: Edit Distance
    try:
        from extractmark.evaluators.edit_distance import EditDistanceEvaluator
        ev = EditDistanceEvaluator()
        dummy_output = PageOutput(document_id="test", page_number=0, raw_text="hello world", normalized_text="hello world")
        eval_results = ev.evaluate(dummy_output, "hello world!")
        assert isinstance(eval_results, list), "Expected list of EvalResult"
        assert len(eval_results) == 2, f"Expected 2 results (CER, WER), got {len(eval_results)}"
        cer = eval_results[0].score
        wer = eval_results[1].score
        assert cer >= 0, "CER should be non-negative"
        results["L1 (Edit Distance)"] = {"status": "OK", "detail": f"CER={cer:.3f}, WER={wer:.3f}"}
    except Exception as e:
        results["L1 (Edit Distance)"] = {"status": "FAIL", "detail": f"{type(e).__name__}: {e}"}
        traceback.print_exc()

    # L2: Semantic Similarity
    try:
        from extractmark.evaluators.semantic_similarity import SemanticSimilarityEvaluator
        ev = SemanticSimilarityEvaluator()
        dummy_output = PageOutput(document_id="test", page_number=0, raw_text="the cat sat on the mat", normalized_text="the cat sat on the mat")
        eval_results = ev.evaluate(dummy_output, "a cat sitting on a mat")
        assert isinstance(eval_results, list), "Expected list of EvalResult"
        cosine = eval_results[0].score
        assert 0 <= cosine <= 1, f"Score out of range: {cosine}"
        results["L2 (Semantic Sim)"] = {"status": "OK", "detail": f"cosine={cosine:.3f}"}
    except Exception as e:
        results["L2 (Semantic Sim)"] = {"status": "FAIL", "detail": f"{type(e).__name__}: {e}"}
        traceback.print_exc()

    # L3: Unit Tests
    try:
        from extractmark.evaluators.unit_tests import UnitTestEvaluator
        ev = UnitTestEvaluator()
        results["L3 (Unit Tests)"] = {"status": "OK", "detail": "instantiated"}
    except Exception as e:
        results["L3 (Unit Tests)"] = {"status": "FAIL", "detail": f"{type(e).__name__}: {e}"}
        traceback.print_exc()

    # L4: LLM Judge (just instantiation -- needs vLLM to actually run)
    try:
        from extractmark.evaluators.llm_judge import LLMJudgeEvaluator
        results["L4 (LLM Judge)"] = {"status": "OK", "detail": "importable (needs vLLM to run)"}
    except Exception as e:
        results["L4 (LLM Judge)"] = {"status": "FAIL", "detail": f"{type(e).__name__}: {e}"}
        traceback.print_exc()

    _print_results("Evaluator", results)
    return results


def test_dependency_imports() -> dict[str, dict]:
    """Test that all critical Python packages can be imported."""
    console.print("\n[bold cyan]═══ Testing Critical Dependencies ═══[/bold cyan]")
    results = {}

    deps = {
        "pymupdf": "import pymupdf",
        "pymupdf4llm (*)": "import pymupdf4llm",  # known incompatibility with pymupdf 1.24; adapter handles fallback
        "pdfplumber": "import pdfplumber",
        "docling": "from docling.document_converter import DocumentConverter",
        "magic_pdf (MinerU)": "from magic_pdf.data.data_reader_writer import FileBasedDataWriter",
        "marker": "from marker.converters.pdf import PdfConverter",
        "unstructured": "from unstructured.partition.auto import partition",
        "markitdown": "from markitdown import MarkItDown",
        "openai": "import openai",
        "PIL": "from PIL import Image",
        "jiwer": "import jiwer",
        "sentence_transformers": "from sentence_transformers import SentenceTransformer",
        "rich": "from rich.console import Console",
        "yaml": "import yaml",
        "vllm (entrypoint)": "import vllm",
    }

    for name, stmt in deps.items():
        try:
            exec(stmt)
            results[name] = {"status": "OK", "detail": "imported"}
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            results[name] = {"status": "FAIL", "detail": str(e)}
        except Exception as e:
            results[name] = {"status": "WARN", "detail": f"{type(e).__name__}: {e}"}

    _print_results("Dependency", results)
    return results


def _print_results(category: str, results: dict[str, dict]):
    """Print a results table."""
    table = Table(title=f"{category} Test Results", show_lines=True)
    table.add_column("ID", style="bold", min_width=20)
    table.add_column("Status", min_width=6)
    table.add_column("Detail", min_width=40)

    for name, r in results.items():
        status = r["status"]
        if status == "OK":
            style = "[green]OK[/green]"
        elif status == "WARN":
            style = "[yellow]WARN[/yellow]"
        else:
            style = "[red]FAIL[/red]"
        table.add_row(name, style, r["detail"][:120])

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Smoke test for ExtractMark benchmark")
    parser.add_argument("--libs-only", action="store_true", help="Only test libraries")
    parser.add_argument("--models-only", action="store_true", help="Only test model configs")
    parser.add_argument("--deps-only", action="store_true", help="Only test dependency imports")
    args = parser.parse_args()

    config = load_config(Path("configs/runs/full_benchmark.yaml"))

    all_results = {}

    if args.deps_only:
        all_results["deps"] = test_dependency_imports()
    elif args.libs_only:
        # Load sample pages first
        console.print("[bold]Loading sample pages from each dataset...[/bold]")
        sample_pages = {}
        for ds_id in BENCHMARK_DATASETS:
            try:
                sample_pages[ds_id] = load_sample_pages(config, ds_id, PAGES_PER_DATASET)
                console.print(f"  {ds_id}: {len(sample_pages[ds_id])} pages loaded")
            except Exception as e:
                console.print(f"  {ds_id}: [red]FAILED[/red] - {e}")
                sample_pages[ds_id] = []
        all_results["libs"] = test_libraries(config, sample_pages)
    elif args.models_only:
        all_results["models"] = test_model_configs(config)
    else:
        # Full test
        all_results["deps"] = test_dependency_imports()
        all_results["datasets"] = test_datasets(config)
        all_results["evaluators"] = test_evaluators(config)
        all_results["models"] = test_model_configs(config)

        # Load sample pages for library tests
        console.print("\n[bold]Loading sample pages for library tests...[/bold]")
        sample_pages = {}
        for ds_id in BENCHMARK_DATASETS:
            try:
                sample_pages[ds_id] = load_sample_pages(config, ds_id, PAGES_PER_DATASET)
                console.print(f"  {ds_id}: {len(sample_pages[ds_id])} pages loaded")
            except Exception as e:
                console.print(f"  {ds_id}: [red]FAILED[/red] - {e}")
                sample_pages[ds_id] = []

        all_results["libs"] = test_libraries(config, sample_pages)

    # Known issues that are handled gracefully (don't count as failures)
    KNOWN_ISSUES = {"pymupdf4llm (*)"}

    # Final summary
    console.print("\n[bold cyan]═══ SUMMARY ═══[/bold cyan]")
    total_ok = total_warn = total_fail = 0
    real_failures = []
    for category, results in all_results.items():
        for name, r in results.items():
            if r["status"] == "OK":
                total_ok += 1
            elif r["status"] == "WARN":
                total_warn += 1
            else:
                total_fail += 1
                if name not in KNOWN_ISSUES:
                    real_failures.append((name, r["detail"]))

    console.print(f"  [green]OK: {total_ok}[/green]  [yellow]WARN: {total_warn}[/yellow]  [red]FAIL: {total_fail}[/red]")
    if KNOWN_ISSUES & {n for cat in all_results.values() for n in cat}:
        console.print(f"  [dim](known issues handled by fallbacks: {', '.join(KNOWN_ISSUES)})[/dim]")

    if real_failures:
        console.print("\n[red bold]FAILURES that will break the benchmark:[/red bold]")
        for name, detail in real_failures:
            console.print(f"  [red]✗[/red] {name}: {detail[:100]}")
        sys.exit(1)
    elif total_warn > 0:
        console.print("\n[yellow]Some warnings -- check above for details.[/yellow]")
    else:
        console.print("\n[green bold]All tests passed![/green bold]")


if __name__ == "__main__":
    main()
