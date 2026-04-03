"""L1 -- Edit Distance evaluator (CER/WER via jiwer)."""

from __future__ import annotations

import logging

from extractmark.types import EvalResult, PageOutput

logger = logging.getLogger(__name__)


class EditDistanceEvaluator:
    """L1: Character Error Rate and Word Error Rate via jiwer."""

    metric_id = "L1"

    def evaluate(self, output: PageOutput, ground_truth: str) -> list[EvalResult]:
        import jiwer

        text = output.normalized_text or output.raw_text
        if not text or not ground_truth:
            return [
                EvalResult(metric_name="cer", score=1.0, details={"error": "empty_input"}),
                EvalResult(metric_name="wer", score=1.0, details={"error": "empty_input"}),
            ]

        try:
            cer = jiwer.cer(ground_truth, text)
            wer = jiwer.wer(ground_truth, text)
        except Exception as e:
            logger.warning("jiwer computation failed: %s", e)
            cer = 1.0
            wer = 1.0

        return [
            EvalResult(
                metric_name="cer",
                score=cer,
                details={
                    "document_id": output.document_id,
                    "page_number": output.page_number,
                },
            ),
            EvalResult(
                metric_name="wer",
                score=wer,
                details={
                    "document_id": output.document_id,
                    "page_number": output.page_number,
                },
            ),
        ]
