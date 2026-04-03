"""Pipeline orchestrator -- connects all components with live progress display."""

from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from extractmark.config import ExtractMarkConfig
from extractmark.datasets.registry import get_dataset
from extractmark.evaluators.registry import get_evaluator
from extractmark.logging_setup import setup_logging, get_log_file_path
from extractmark.models.registry import get_model
from extractmark.normalize import normalize
from extractmark.reporting.summary import SummaryReporter
from extractmark.types import EvalResult, PageOutput, RunResult

logger = logging.getLogger(__name__)
console = Console()


class BenchmarkPipeline:
    """Central orchestrator for ExtractMark benchmark runs."""

    def __init__(self, config: ExtractMarkConfig):
        self.config = config
        self.results_dir = Path("results")
        self.report_dir = Path("report")
        self._run_results: list[RunResult] = []
        self._start_time: float = 0
        self._completed_combos: int = 0
        self._total_combos: int = 0
        self._status_log: list[str] = []

        # Set up per-run logging
        self._log_path = setup_logging(config.run.name)
        logger.info("Config: models=%s, datasets=%s, evaluators=%s, max_pages=%s",
                     config.run.models, config.run.datasets, config.run.evaluators,
                     config.run.max_pages)

    def run(self) -> None:
        """Execute the benchmark pipeline."""
        self._start_time = time.time()
        reporter = SummaryReporter(self.results_dir, self.report_dir)

        # Resolve components
        model_adapters = self._resolve_models()
        datasets = self._resolve_datasets()
        evaluators = self._resolve_evaluators()

        self._total_combos = len(model_adapters) * len(datasets)

        console.print()
        console.print(Panel(
            f"[bold cyan]ExtractMark Benchmark[/bold cyan]\n"
            f"Run: [green]{self.config.run.name}[/green]\n"
            f"Models: {', '.join(model_adapters.keys())}\n"
            f"Datasets: {', '.join(datasets.keys())}\n"
            f"Evaluators: {self.config.run.evaluators}\n"
            f"Total combinations: {self._total_combos}\n"
            f"Log file: {self._log_path}\n"
            f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            title="Configuration",
        ))
        console.print()
        logger.info("Starting %d benchmark combinations", self._total_combos)

        # Run each (model, dataset) combination
        for adapter_id, adapter in model_adapters.items():
            for dataset_id, dataset in datasets.items():
                combo_label = f"{adapter_id} x {dataset_id}"

                console.rule(f"[bold]{combo_label}[/bold] ({self._completed_combos + 1}/{self._total_combos})")
                logger.info("--- %s (%d/%d) ---", combo_label, self._completed_combos + 1, self._total_combos)

                # Health check
                if hasattr(adapter, "health_check"):
                    if not adapter.health_check():
                        msg = f"SKIP {combo_label} -- health check failed"
                        console.print(f"  [red]{msg}[/red]")
                        logger.warning(msg)
                        self._status_log.append(msg)
                        self._completed_combos += 1
                        continue

                try:
                    run_result = self._run_single(adapter_id, adapter, dataset_id, dataset, evaluators)
                    self._run_results.append(run_result)
                    reporter.add_run(run_result)
                except Exception as e:
                    msg = f"ERROR {combo_label}: {type(e).__name__}: {e}"
                    console.print(f"  [red]{msg}[/red]")
                    logger.error(msg)
                    logger.error("Traceback:\n%s", traceback.format_exc())
                    self._status_log.append(msg)

                self._completed_combos += 1

                # Print intermediate summary
                if self._run_results:
                    self._print_combo_summary(self._run_results[-1])

        # Generate reports
        console.print()
        console.rule("[bold green]Generating Reports[/bold green]")
        reporter.generate()

        total_elapsed = time.time() - self._start_time
        self._print_final_summary(total_elapsed)

    def _resolve_models(self) -> dict:
        adapters = {}
        for model_id in self.config.run.models:
            model_config = self.config.models.get(model_id)
            if model_config is None:
                logger.warning("Model %s not found in config, skipping", model_id)
                console.print(f"[yellow]Warning: model {model_id} not in config, skipping[/yellow]")
                continue
            logger.info("Resolved model %s: %s", model_id, model_config.hf_model_id)
            adapters[model_id] = get_model(model_id, model_config)
        return adapters

    def _resolve_datasets(self) -> dict:
        loaders = {}
        for dataset_id in self.config.run.datasets:
            dataset_config = self.config.datasets.get(dataset_id)
            if dataset_config is None:
                logger.warning("Dataset %s not found in config, skipping", dataset_id)
                console.print(f"[yellow]Warning: dataset {dataset_id} not in config, skipping[/yellow]")
                continue
            logger.info("Resolved dataset %s: %s (path=%s)", dataset_id, dataset_config.name, dataset_config.path)
            loaders[dataset_id] = get_dataset(dataset_id, dataset_config)
        return loaders

    def _resolve_evaluators(self) -> list:
        evals = []
        for eval_id in self.config.run.evaluators:
            try:
                evaluator = get_evaluator(eval_id, self.config.evaluation)
                evals.append(evaluator)
                logger.info("Resolved evaluator %s: %s", eval_id, type(evaluator).__name__)
            except ValueError as e:
                logger.warning("Evaluator %s: %s", eval_id, e)
                console.print(f"[yellow]Warning: {e}[/yellow]")
        return evals

    def _run_single(self, adapter_id, adapter, dataset_id, dataset, evaluators) -> RunResult:
        """Run a single (adapter, dataset) combination with live progress."""
        warmup = self.config.run.warmup_pages
        max_pages = self.config.run.max_pages

        all_eval_results: list[EvalResult] = []
        page_outputs: list[PageOutput] = []
        timing_pages: list[float] = []
        cold_start_times: list[float] = []
        page_errors: list[dict] = []

        # Create output directory
        out_dir = self.results_dir / adapter_id / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)

        page_iter = dataset.load()
        page_count = 0
        combo_start = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                f"  {adapter_id} x {dataset_id}",
                total=max_pages or 0,
            )

            for page in page_iter:
                if max_pages and page_count >= max_pages:
                    break

                page_label = f"{page.document_id}/page_{page.page_number}"

                # Process page
                try:
                    output = adapter.process_page(page)
                    output.normalized_text = normalize(output.raw_text)
                    logger.debug(
                        "Page %s: %dms, %d chars raw, %d chars normalized",
                        page_label, output.inference_time_ms,
                        len(output.raw_text), len(output.normalized_text),
                    )
                except Exception as e:
                    logger.error("Inference error on %s: %s", page_label, e)
                    logger.error("Traceback:\n%s", traceback.format_exc())
                    page_errors.append({
                        "page": page_label,
                        "error": str(e),
                        "type": type(e).__name__,
                    })
                    output = PageOutput(
                        document_id=page.document_id,
                        page_number=page.page_number,
                        raw_text="",
                        metadata={"error": str(e)},
                    )

                # Save output
                doc_dir = out_dir / page.document_id
                doc_dir.mkdir(parents=True, exist_ok=True)
                page_path = doc_dir / f"page_{page.page_number}.json"
                with open(page_path, "w") as f:
                    json.dump(output.to_dict(), f, indent=2)

                txt_path = doc_dir / f"page_{page.page_number}.txt"
                txt_path.write_text(output.normalized_text)

                # Track timing
                if page_count < warmup:
                    cold_start_times.append(output.inference_time_ms)
                    logger.info("Warm-up page %d: %dms", page_count, output.inference_time_ms)
                else:
                    timing_pages.append(output.inference_time_ms)

                # Evaluate
                gt = page.ground_truth
                if gt:
                    for evaluator in evaluators:
                        try:
                            eval_results = evaluator.evaluate(output, gt)
                            all_eval_results.extend(eval_results)
                            for er in eval_results:
                                logger.debug(
                                    "Eval %s on %s: %s = %.4f",
                                    evaluator.metric_id, page_label, er.metric_name, er.score,
                                )
                        except Exception as e:
                            logger.error("Evaluator %s failed on %s: %s",
                                         evaluator.metric_id, page_label, e)

                page_outputs.append(output)
                page_count += 1

                # Update progress
                avg_ms = sum(timing_pages) / len(timing_pages) if timing_pages else output.inference_time_ms
                progress.update(
                    task,
                    advance=1,
                    description=f"  {adapter_id} x {dataset_id} ({avg_ms:.0f}ms/page)",
                )

                if max_pages is None and not progress.tasks[task].total:
                    progress.update(task, total=None)

        combo_elapsed = time.time() - combo_start

        # Calculate throughput (excluding warm-up)
        throughput = 0.0
        mean_ms = 0.0
        if timing_pages:
            total_time_min = sum(timing_pages) / 60000.0
            throughput = len(timing_pages) / total_time_min if total_time_min > 0 else 0.0
            mean_ms = sum(timing_pages) / len(timing_pages)

        cold_start = sum(cold_start_times) if cold_start_times else 0.0

        # Log combo summary
        logger.info(
            "Completed %s x %s: %d pages, %.1f p/min, %.0fms mean, %.0fms cold start, %d errors",
            adapter_id, dataset_id, page_count, throughput, mean_ms, cold_start, len(page_errors),
        )

        # Save eval results
        eval_path = out_dir / "eval_results.json"
        with open(eval_path, "w") as f:
            json.dump([r.to_dict() for r in all_eval_results], f, indent=2)

        # Save run metadata (includes errors for investigation)
        meta = {
            "adapter_id": adapter_id,
            "dataset_id": dataset_id,
            "pages_processed": page_count,
            "warmup_pages": warmup,
            "throughput_pages_per_min": throughput,
            "cold_start_latency_ms": cold_start,
            "mean_inference_ms": mean_ms,
            "total_time_s": combo_elapsed,
            "timestamp": datetime.now().isoformat(),
            "errors": page_errors,
            "log_file": str(self._log_path),
        }
        meta_path = out_dir / "run_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        if page_errors:
            logger.warning("Errors during %s x %s: %s", adapter_id, dataset_id, page_errors)

        return RunResult(
            adapter_id=adapter_id,
            dataset_id=dataset_id,
            page_outputs=page_outputs,
            eval_results=all_eval_results,
            throughput_pages_per_min=throughput,
            cold_start_latency_ms=cold_start,
        )

    def _print_combo_summary(self, result: RunResult) -> None:
        """Print summary for a completed (adapter, dataset) run."""
        scores: dict[str, list[float]] = {}
        for er in result.eval_results:
            if er.metric_name not in scores:
                scores[er.metric_name] = []
            scores[er.metric_name].append(er.score)

        avg_scores = {k: sum(v) / len(v) for k, v in scores.items()}

        parts = [
            f"  Pages: {len(result.page_outputs)}",
            f"Throughput: {result.throughput_pages_per_min:.1f} p/min",
        ]
        for metric, avg in sorted(avg_scores.items()):
            parts.append(f"{metric}: {avg:.4f}")

        elapsed = time.time() - self._start_time
        remaining = self._total_combos - self._completed_combos
        if self._completed_combos > 0:
            eta_s = (elapsed / self._completed_combos) * remaining
            eta_str = str(timedelta(seconds=int(eta_s)))
            parts.append(f"ETA remaining: {eta_str}")

        summary = " | ".join(parts)
        console.print(summary)
        logger.info(summary)

    def _print_final_summary(self, total_elapsed: float) -> None:
        """Print final benchmark summary."""
        table = Table(title="ExtractMark Benchmark Results", show_lines=True)
        table.add_column("Adapter", style="cyan")
        table.add_column("Dataset", style="green")
        table.add_column("Pages")
        table.add_column("Throughput\n(p/min)")
        table.add_column("CER", justify="right")
        table.add_column("WER", justify="right")
        table.add_column("SBERT", justify="right")

        for result in self._run_results:
            scores: dict[str, list[float]] = {}
            for er in result.eval_results:
                if er.metric_name not in scores:
                    scores[er.metric_name] = []
                scores[er.metric_name].append(er.score)

            avg = {k: sum(v) / len(v) for k, v in scores.items()}

            table.add_row(
                result.adapter_id,
                result.dataset_id,
                str(len(result.page_outputs)),
                f"{result.throughput_pages_per_min:.1f}",
                f"{avg.get('cer', float('nan')):.4f}" if 'cer' in avg else "-",
                f"{avg.get('wer', float('nan')):.4f}" if 'wer' in avg else "-",
                f"{avg.get('sbert_cosine', float('nan')):.4f}" if 'sbert_cosine' in avg else "-",
            )

        console.print()
        console.print(table)
        console.print()

        log_path = self._log_path
        console.print(f"[bold green]Benchmark complete![/bold green]")
        console.print(f"  Total time: {timedelta(seconds=int(total_elapsed))}")
        console.print(f"  Results:    {self.results_dir}/")
        console.print(f"  Report:     {self.report_dir}/benchmark_summary.md")
        console.print(f"  CSV:        {self.report_dir}/benchmark_summary.csv")
        console.print(f"  Log:        {log_path}")

        logger.info("Benchmark complete. Total time: %s", timedelta(seconds=int(total_elapsed)))
        logger.info("Results: %s", self.results_dir)
        logger.info("Report:  %s/benchmark_summary.md", self.report_dir)
        logger.info("Log:     %s", log_path)

        if self._status_log:
            logger.warning("Status issues during run:")
            for msg in self._status_log:
                logger.warning("  %s", msg)
