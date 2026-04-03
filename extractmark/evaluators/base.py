"""Base protocol for evaluators."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from extractmark.types import EvalResult, PageOutput


@runtime_checkable
class Evaluator(Protocol):
    """Interface for evaluation metric layers."""

    metric_id: str

    def evaluate(self, output: PageOutput, ground_truth: str) -> list[EvalResult]:
        """Evaluate a single page output against ground truth."""
        ...
