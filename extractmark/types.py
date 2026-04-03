"""Shared data types for the ExtractMark pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PageInput:
    """A single page to process."""

    document_id: str
    page_number: int
    image_path: Path
    ground_truth: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageOutput:
    """Result from a model or library processing one page."""

    document_id: str
    page_number: int
    raw_text: str
    normalized_text: str = ""
    tables: list[str] = field(default_factory=list)
    bboxes: list[dict[str, Any]] | None = None
    inference_time_ms: float = 0.0
    gpu_memory_mb: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "tables": self.tables,
            "bboxes": self.bboxes,
            "inference_time_ms": self.inference_time_ms,
            "gpu_memory_mb": self.gpu_memory_mb,
            "metadata": self.metadata,
        }


@dataclass
class EvalResult:
    """Result from one evaluator on one page."""

    metric_name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "details": self.details,
        }


@dataclass
class RunResult:
    """Aggregated results for one (adapter, dataset) pair."""

    adapter_id: str
    dataset_id: str
    page_outputs: list[PageOutput] = field(default_factory=list)
    eval_results: list[EvalResult] = field(default_factory=list)
    throughput_pages_per_min: float = 0.0
    cold_start_latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
