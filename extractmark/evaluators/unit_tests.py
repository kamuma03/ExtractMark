"""L3 -- Binary Unit Tests evaluator (OlmOCR-Bench framework).

Runs deterministic pass/fail assertions per page:
- Presence/absence of key phrases
- Table cell ordering and placement
- Reading order correctness
- N-gram non-repetition
"""

from __future__ import annotations

import logging
import re

from extractmark.types import EvalResult, PageOutput

logger = logging.getLogger(__name__)


def _check_presence(text: str, phrase: str) -> bool:
    """Check if a phrase is present in the text (case-insensitive)."""
    return phrase.lower() in text.lower()


def _check_absence(text: str, phrase: str) -> bool:
    """Check if a phrase is absent from the text."""
    return phrase.lower() not in text.lower()


def _check_order(text: str, first: str, second: str) -> bool:
    """Check if 'first' appears before 'second' in the text."""
    pos_first = text.lower().find(first.lower())
    pos_second = text.lower().find(second.lower())
    if pos_first == -1 or pos_second == -1:
        return False
    return pos_first < pos_second


def _check_no_repetition(text: str, n: int = 5, max_repeats: int = 3) -> bool:
    """Check that no n-gram is repeated more than max_repeats times."""
    words = text.split()
    if len(words) < n:
        return True
    ngram_counts: dict[str, int] = {}
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        if ngram_counts[ngram] > max_repeats:
            return False
    return True


class UnitTestEvaluator:
    """L3: Binary unit test assertions per page."""

    metric_id = "L3"

    def evaluate(self, output: PageOutput, ground_truth: str) -> list[EvalResult]:
        text = output.normalized_text or output.raw_text
        unit_tests = output.metadata.get("unit_tests", [])

        if not unit_tests:
            # Run default structural checks
            return self._default_checks(output, text)

        passed = 0
        total = len(unit_tests)
        failures: list[dict] = []

        for test in unit_tests:
            test_type = test.get("type", "presence")
            result = False

            if test_type == "presence":
                result = _check_presence(text, test["value"])
            elif test_type == "absence":
                result = _check_absence(text, test["value"])
            elif test_type == "order":
                result = _check_order(text, test["first"], test["second"])
            elif test_type == "no_repetition":
                result = _check_no_repetition(text, test.get("n", 5))
            elif test_type == "regex":
                result = bool(re.search(test["pattern"], text))

            if result:
                passed += 1
            else:
                failures.append({"test": test, "passed": False})

        pass_rate = passed / total if total > 0 else 0.0

        return [
            EvalResult(
                metric_name="unit_test_pass_rate",
                score=pass_rate,
                details={
                    "document_id": output.document_id,
                    "page_number": output.page_number,
                    "passed": passed,
                    "total": total,
                    "failures": failures[:10],  # Limit logged failures
                },
            )
        ]

    def _default_checks(self, output: PageOutput, text: str) -> list[EvalResult]:
        """Run default structural checks when no explicit tests are provided."""
        checks_passed = 0
        checks_total = 0

        # Check: no excessive repetition
        checks_total += 1
        if _check_no_repetition(text):
            checks_passed += 1

        # Check: non-empty output
        checks_total += 1
        if text.strip():
            checks_passed += 1

        # Check: reasonable length (not truncated or exploded)
        checks_total += 1
        if 10 < len(text) < 100_000:
            checks_passed += 1

        pass_rate = checks_passed / checks_total if checks_total > 0 else 0.0

        return [
            EvalResult(
                metric_name="unit_test_pass_rate",
                score=pass_rate,
                details={
                    "document_id": output.document_id,
                    "page_number": output.page_number,
                    "passed": checks_passed,
                    "total": checks_total,
                    "type": "default_structural",
                },
            )
        ]
