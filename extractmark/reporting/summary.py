"""Benchmark summary report generator (Markdown + CSV)."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from extractmark.types import EvalResult, RunResult

logger = logging.getLogger(__name__)


class SummaryReporter:
    """Generate benchmark summary reports from run results."""

    def __init__(self, results_dir: Path, output_dir: Path):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self._runs: list[RunResult] = []

    def add_run(self, run_result: RunResult) -> None:
        self._runs.append(run_result)

    @staticmethod
    def _backup_if_exists(path: Path) -> None:
        """Rename an existing file with a timestamp suffix to avoid overwriting."""
        if not path.exists():
            return
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        stamp = mtime.strftime("%Y%m%d_%H%M%S")
        stem = path.stem
        suffix = path.suffix
        backup = path.with_name(f"{stem}_{stamp}{suffix}")
        # Avoid collision if backup already exists
        counter = 1
        while backup.exists():
            backup = path.with_name(f"{stem}_{stamp}_{counter}{suffix}")
            counter += 1
        path.rename(backup)
        logger.info("Backed up previous report to %s", backup.name)

    def generate(self) -> None:
        """Generate Markdown and CSV reports."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self._runs:
            self._runs = self._load_results_from_disk()

        # Back up existing reports before overwriting
        self._backup_if_exists(self.output_dir / "benchmark_summary.md")
        self._backup_if_exists(self.output_dir / "benchmark_summary.csv")

        self._generate_markdown()
        self._generate_csv()
        logger.info("Reports generated in %s", self.output_dir)

    def _load_results_from_disk(self) -> list[RunResult]:
        """Load results from the results directory structure."""
        runs: list[RunResult] = []
        if not self.results_dir.exists():
            return runs

        for adapter_dir in sorted(self.results_dir.iterdir()):
            if not adapter_dir.is_dir():
                continue
            for dataset_dir in sorted(adapter_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                # Load eval results
                eval_path = dataset_dir / "eval_results.json"
                eval_results = []
                if eval_path.exists():
                    try:
                        with open(eval_path) as f:
                            for item in json.load(f):
                                eval_results.append(EvalResult(**item))
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.warning("Skipping corrupt eval_results.json in %s: %s",
                                       dataset_dir, e)

                # Load metadata
                meta_path = dataset_dir / "run_metadata.json"
                metadata = {}
                if meta_path.exists():
                    try:
                        with open(meta_path) as f:
                            metadata = json.load(f)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning("Skipping corrupt run_metadata.json in %s: %s",
                                       dataset_dir, e)

                runs.append(RunResult(
                    adapter_id=adapter_dir.name,
                    dataset_id=dataset_dir.name,
                    eval_results=eval_results,
                    throughput_pages_per_min=metadata.get("throughput_pages_per_min", 0),
                    cold_start_latency_ms=metadata.get("cold_start_latency_ms", 0),
                    metadata=metadata,
                ))
        return runs

    def _aggregate_scores(self) -> dict[str, dict[str, dict[str, float]]]:
        """Aggregate scores by adapter -> dataset -> metric."""
        scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for run in self._runs:
            for result in run.eval_results:
                scores[run.adapter_id][run.dataset_id][result.metric_name].append(
                    result.score
                )

        # Compute averages
        avg_scores: dict[str, dict[str, dict[str, float]]] = {}
        for adapter_id, datasets in scores.items():
            avg_scores[adapter_id] = {}
            for dataset_id, metrics in datasets.items():
                avg_scores[adapter_id][dataset_id] = {
                    metric: sum(vals) / len(vals) for metric, vals in metrics.items()
                }
        return avg_scores

    def _generate_markdown(self) -> None:
        """Generate benchmark_summary.md."""
        scores = self._aggregate_scores()
        lines: list[str] = []
        lines.append("# ExtractMark Benchmark Summary\n")
        lines.append("")

        if not scores:
            lines.append("No results available.\n")
            (self.output_dir / "benchmark_summary.md").write_text("\n".join(lines))
            return

        # Collect all metrics across all runs
        all_metrics: set[str] = set()
        for datasets in scores.values():
            for metrics in datasets.values():
                all_metrics.update(metrics.keys())
        metric_list = sorted(all_metrics)

        # Summary table per dataset
        all_datasets: set[str] = set()
        for datasets in scores.values():
            all_datasets.update(datasets.keys())

        for dataset_id in sorted(all_datasets):
            lines.append(f"## {dataset_id}\n")
            lines.append("")

            # Table header
            header = "| Adapter | " + " | ".join(metric_list) + " | Throughput (p/min) |"
            separator = "|" + "|".join(["---"] * (len(metric_list) + 2)) + "|"
            lines.append(header)
            lines.append(separator)

            for adapter_id in sorted(scores.keys()):
                if dataset_id not in scores[adapter_id]:
                    continue
                metrics = scores[adapter_id][dataset_id]

                # Find throughput
                throughput = 0.0
                for run in self._runs:
                    if run.adapter_id == adapter_id and run.dataset_id == dataset_id:
                        throughput = run.throughput_pages_per_min
                        break

                row = f"| {adapter_id} | "
                row += " | ".join(f"{metrics.get(m, 0.0):.4f}" for m in metric_list)
                row += f" | {throughput:.1f} |"
                lines.append(row)

            lines.append("")

        (self.output_dir / "benchmark_summary.md").write_text("\n".join(lines))

    def _generate_csv(self) -> None:
        """Generate benchmark_summary.csv."""
        scores = self._aggregate_scores()
        if not scores:
            return

        all_metrics: set[str] = set()
        for datasets in scores.values():
            for metrics in datasets.values():
                all_metrics.update(metrics.keys())
        metric_list = sorted(all_metrics)

        csv_path = self.output_dir / "benchmark_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["adapter_id", "dataset_id"] + metric_list + ["throughput_pages_per_min"])

            for adapter_id in sorted(scores.keys()):
                for dataset_id in sorted(scores[adapter_id].keys()):
                    metrics = scores[adapter_id][dataset_id]
                    throughput = 0.0
                    for run in self._runs:
                        if run.adapter_id == adapter_id and run.dataset_id == dataset_id:
                            throughput = run.throughput_pages_per_min
                            break

                    row = [adapter_id, dataset_id]
                    row += [f"{metrics.get(m, 0.0):.4f}" for m in metric_list]
                    row.append(f"{throughput:.1f}")
                    writer.writerow(row)
