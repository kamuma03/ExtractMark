"""OlmOCR-Bench (D-05) dataset loader.

OlmOCR-Bench -- 1,400+ PDFs with 7,000+ binary unit tests across 6 categories.
Expected directory structure:
    data/olmocr_bench/
        images/
            {document_id}/
                page_{n}.png
        tests/
            {document_id}.json    # unit test assertions
        ground_truth/
            {document_id}.json    # ground truth text
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from extractmark.types import PageInput

logger = logging.getLogger(__name__)


class OlmOCRBenchLoader:
    """Loader for OlmOCR-Bench dataset."""

    def __init__(self, dataset_id: str, path: str | Path):
        self.dataset_id = dataset_id
        self.root = Path(path)
        self._ground_truth_cache: dict[str, dict[int, str]] = {}
        self._unit_tests: dict[str, list[dict]] = {}

    def load(self) -> Iterator[PageInput]:
        images_dir = self.root / "images"
        if not images_dir.exists():
            logger.warning("OlmOCR-Bench images directory not found: %s", images_dir)
            return

        for doc_dir in sorted(images_dir.iterdir()):
            if not doc_dir.is_dir():
                continue
            document_id = doc_dir.name
            self._load_ground_truth(document_id)
            self._load_unit_tests(document_id)

            for image_path in sorted(doc_dir.glob("*.png")):
                page_num = self._parse_page_number(image_path.stem)
                gt = self.get_ground_truth(document_id, page_num)
                tests = self.get_unit_tests(document_id)
                yield PageInput(
                    document_id=document_id,
                    page_number=page_num,
                    image_path=image_path,
                    ground_truth=gt,
                    metadata={
                        "dataset": self.dataset_id,
                        "unit_tests": tests,
                    },
                )

    def get_ground_truth(self, document_id: str, page_number: int) -> str | None:
        if document_id not in self._ground_truth_cache:
            self._load_ground_truth(document_id)
        return self._ground_truth_cache.get(document_id, {}).get(page_number)

    def get_unit_tests(self, document_id: str) -> list[dict]:
        """Get binary unit test assertions for a document."""
        if document_id not in self._unit_tests:
            self._load_unit_tests(document_id)
        return self._unit_tests.get(document_id, [])

    def _load_ground_truth(self, document_id: str) -> None:
        gt_path = self.root / "ground_truth" / f"{document_id}.json"
        if not gt_path.exists():
            return
        try:
            with open(gt_path) as f:
                data = json.load(f)
            pages: dict[int, str] = {}
            if isinstance(data, dict):
                for key, value in data.items():
                    try:
                        pages[int(key)] = value if isinstance(value, str) else value.get("text", "")
                    except ValueError:
                        pass
            self._ground_truth_cache[document_id] = pages
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load ground truth for %s: %s", document_id, e)

    def _load_unit_tests(self, document_id: str) -> None:
        tests_path = self.root / "tests" / f"{document_id}.json"
        if not tests_path.exists():
            return
        try:
            with open(tests_path) as f:
                self._unit_tests[document_id] = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load unit tests for %s: %s", document_id, e)

    @staticmethod
    def _parse_page_number(stem: str) -> int:
        if "_" in stem:
            try:
                return int(stem.split("_")[-1])
            except ValueError:
                pass
        try:
            return int(stem)
        except ValueError:
            return 0
