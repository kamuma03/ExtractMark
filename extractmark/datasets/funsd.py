"""FUNSD (D-03) dataset loader.

FUNSD -- Form understanding on noisy scans.
Expected directory structure:
    data/funsd/
        images/
            {document_id}.png
        annotations/
            {document_id}.json
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from extractmark.types import PageInput

logger = logging.getLogger(__name__)


class FUNSDLoader:
    """Loader for FUNSD dataset."""

    def __init__(self, dataset_id: str, path: str | Path):
        self.dataset_id = dataset_id
        self.root = Path(path)
        self._ground_truth_cache: dict[str, str] = {}

    def load(self) -> Iterator[PageInput]:
        images_dir = self.root / "images"
        if not images_dir.exists():
            logger.warning("FUNSD images directory not found: %s", images_dir)
            return

        for image_path in sorted(images_dir.glob("*.png")):
            document_id = image_path.stem
            gt = self.get_ground_truth(document_id, 0)
            yield PageInput(
                document_id=document_id,
                page_number=0,
                image_path=image_path,
                ground_truth=gt,
                metadata={"dataset": self.dataset_id},
            )

    def get_ground_truth(self, document_id: str, page_number: int) -> str | None:
        if document_id not in self._ground_truth_cache:
            self._load_annotations(document_id)
        return self._ground_truth_cache.get(document_id)

    def _load_annotations(self, document_id: str) -> None:
        ann_path = self.root / "annotations" / f"{document_id}.json"
        if not ann_path.exists():
            return
        try:
            with open(ann_path) as f:
                data = json.load(f)
            # FUNSD format: list of form entries with text and bboxes
            texts = []
            if isinstance(data, dict) and "form" in data:
                for entry in data["form"]:
                    text = entry.get("text", "")
                    if text:
                        texts.append(text)
            self._ground_truth_cache[document_id] = "\n".join(texts)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load annotations for %s: %s", document_id, e)
