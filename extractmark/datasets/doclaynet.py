"""DocLayNet (D-06) dataset loader.

DocLayNet -- 80K+ pages of real enterprise documents with human-annotated
bounding boxes for 11 element types.
Expected directory structure:
    data/doclaynet/
        images/
            {document_id}/
                page_{n}.png
        annotations/
            {document_id}.json    # COCO-format annotations
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from extractmark.types import PageInput

logger = logging.getLogger(__name__)


class DocLayNetLoader:
    """Loader for DocLayNet dataset."""

    def __init__(self, dataset_id: str, path: str | Path):
        self.dataset_id = dataset_id
        self.root = Path(path)
        self._ground_truth_cache: dict[str, dict[int, str]] = {}

    def load(self) -> Iterator[PageInput]:
        images_dir = self.root / "images"
        if not images_dir.exists():
            logger.warning("DocLayNet images directory not found: %s", images_dir)
            return

        for doc_dir in sorted(images_dir.iterdir()):
            if not doc_dir.is_dir():
                continue
            document_id = doc_dir.name
            self._load_annotations(document_id)

            for image_path in sorted(doc_dir.glob("*.png")):
                page_num = self._parse_page_number(image_path.stem)
                gt = self.get_ground_truth(document_id, page_num)
                yield PageInput(
                    document_id=document_id,
                    page_number=page_num,
                    image_path=image_path,
                    ground_truth=gt,
                    metadata={"dataset": self.dataset_id},
                )

    def get_ground_truth(self, document_id: str, page_number: int) -> str | None:
        if document_id not in self._ground_truth_cache:
            self._load_annotations(document_id)
        return self._ground_truth_cache.get(document_id, {}).get(page_number)

    def _load_annotations(self, document_id: str) -> None:
        ann_path = self.root / "annotations" / f"{document_id}.json"
        if not ann_path.exists():
            return
        try:
            with open(ann_path) as f:
                data = json.load(f)
            pages: dict[int, str] = {}
            if isinstance(data, dict) and "annotations" in data:
                # COCO format: collect text from annotations per page
                for ann in data["annotations"]:
                    page_num = ann.get("page_number", 0)
                    text = ann.get("text", "")
                    if page_num not in pages:
                        pages[page_num] = text
                    else:
                        pages[page_num] += "\n" + text
            self._ground_truth_cache[document_id] = pages
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load annotations for %s: %s", document_id, e)

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
