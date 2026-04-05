"""Tests for extractmark.reporting.summary -- report generation."""

import csv
import json
from pathlib import Path

from extractmark.types import EvalResult, RunResult, PageOutput
from extractmark.reporting.summary import SummaryReporter


def _make_run(adapter: str, dataset: str, scores: dict[str, float],
              throughput: float = 10.0) -> RunResult:
    eval_results = [
        EvalResult(metric_name=name, score=score)
        for name, score in scores.items()
    ]
    return RunResult(
        adapter_id=adapter,
        dataset_id=dataset,
        eval_results=eval_results,
        throughput_pages_per_min=throughput,
    )


class TestSummaryReporter:
    def test_generate_markdown(self, tmp_path):
        results_dir = tmp_path / "results"
        output_dir = tmp_path / "report"

        reporter = SummaryReporter(results_dir, output_dir)
        reporter.add_run(_make_run("M-01", "D-01", {"cer": 0.1, "wer": 0.2}))
        reporter.add_run(_make_run("M-02", "D-01", {"cer": 0.3, "wer": 0.4}))
        reporter.generate()

        md_path = output_dir / "benchmark_summary.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "# ExtractMark Benchmark Summary" in content
        assert "M-01" in content
        assert "M-02" in content
        assert "D-01" in content
        assert "0.1000" in content  # cer for M-01

    def test_generate_csv(self, tmp_path):
        results_dir = tmp_path / "results"
        output_dir = tmp_path / "report"

        reporter = SummaryReporter(results_dir, output_dir)
        reporter.add_run(_make_run("M-01", "D-01", {"cer": 0.1, "wer": 0.2}, throughput=50.0))
        reporter.generate()

        csv_path = output_dir / "benchmark_summary.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        # header + 1 data row
        assert len(rows) == 2
        assert "adapter_id" in rows[0]
        assert "M-01" in rows[1]

    def test_generate_empty(self, tmp_path):
        reporter = SummaryReporter(tmp_path / "results", tmp_path / "report")
        reporter.generate()
        content = (tmp_path / "report" / "benchmark_summary.md").read_text()
        assert "No results available" in content

    def test_multiple_datasets(self, tmp_path):
        reporter = SummaryReporter(tmp_path / "r", tmp_path / "o")
        reporter.add_run(_make_run("M-01", "D-01", {"cer": 0.1}))
        reporter.add_run(_make_run("M-01", "D-02", {"cer": 0.2}))
        reporter.generate()

        content = (tmp_path / "o" / "benchmark_summary.md").read_text()
        assert "## D-01" in content
        assert "## D-02" in content

    def test_score_aggregation(self, tmp_path):
        """Multiple eval results for same metric should be averaged."""
        rr = RunResult(
            adapter_id="M-01", dataset_id="D-01",
            eval_results=[
                EvalResult(metric_name="cer", score=0.1),
                EvalResult(metric_name="cer", score=0.3),
            ],
            throughput_pages_per_min=10.0,
        )
        reporter = SummaryReporter(tmp_path / "r", tmp_path / "o")
        reporter.add_run(rr)
        reporter.generate()

        content = (tmp_path / "o" / "benchmark_summary.md").read_text()
        assert "0.2000" in content  # average of 0.1 and 0.3

    def test_load_from_disk(self, tmp_path):
        """Reporter should load results from disk when no runs are added."""
        results_dir = tmp_path / "results" / "M-01" / "D-01"
        results_dir.mkdir(parents=True)
        (results_dir / "eval_results.json").write_text(json.dumps([
            {"metric_name": "cer", "score": 0.15, "details": {}},
        ]))
        (results_dir / "run_metadata.json").write_text(json.dumps({
            "throughput_pages_per_min": 25.0,
        }))

        reporter = SummaryReporter(tmp_path / "results", tmp_path / "report")
        reporter.generate()

        content = (tmp_path / "report" / "benchmark_summary.md").read_text()
        assert "M-01" in content
        assert "0.1500" in content
