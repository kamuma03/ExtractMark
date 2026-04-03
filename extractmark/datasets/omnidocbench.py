"""OmniDocBench (D-01) dataset loader.

OmniDocBench v1.5 -- overall text, tables, math, layout evaluation.
Actual directory structure:
    data/omnidocbench/
        images/
            {filename}.png or .jpg    # flat directory of page images
        OmniDocBench.json             # single annotation file (list of 1355 entries)

Each JSON entry has:
    - page_info.image_path: filename matching images/
    - layout_dets[]: list of elements with text, category_type, poly, order
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from extractmark.types import PageInput

logger = logging.getLogger(__name__)


class OmniDocBenchLoader:
    """Loader for OmniDocBench v1.5 dataset."""

    def __init__(self, dataset_id: str, path: str | Path):
        self.dataset_id = dataset_id
        self.root = Path(path)
        self._annotations: list[dict] | None = None
        self._gt_cache: dict[str, str] = {}

    def _load_annotations(self) -> list[dict]:
        if self._annotations is not None:
            return self._annotations

        ann_path = self.root / "OmniDocBench.json"
        if not ann_path.exists():
            logger.warning("OmniDocBench.json not found at %s", ann_path)
            self._annotations = []
            return self._annotations

        with open(ann_path) as f:
            self._annotations = json.load(f)

        # Build ground truth cache: image_path -> concatenated text
        for entry in self._annotations:
            image_path = entry.get("page_info", {}).get("image_path", "")
            if not image_path:
                continue

            # Concatenate all text elements in reading order
            layout_dets = entry.get("layout_dets", [])
            sorted_dets = sorted(layout_dets, key=lambda d: d.get("order") or 0)
            texts = []
            for det in sorted_dets:
                text = det.get("text", "")
                if text and not det.get("ignore", False):
                    texts.append(text)
            self._gt_cache[image_path] = "\n".join(texts)

        logger.info("Loaded %d OmniDocBench annotations", len(self._annotations))
        return self._annotations

    def load(self) -> Iterator[PageInput]:
        annotations = self._load_annotations()
        images_dir = self.root / "images"

        if not images_dir.exists():
            logger.warning("OmniDocBench images directory not found: %s", images_dir)
            return

        for i, entry in enumerate(annotations):
            page_info = entry.get("page_info", {})
            image_filename = page_info.get("image_path", "")
            if not image_filename:
                continue

            image_path = images_dir / image_filename
            if not image_path.exists():
                # Try alternate extensions
                for ext in [".png", ".jpg", ".jpeg"]:
                    alt = image_path.with_suffix(ext)
                    if alt.exists():
                        image_path = alt
                        break
                else:
                    logger.debug("Image not found: %s", image_path)
                    continue

            document_id = image_path.stem
            page_num = page_info.get("page_no") or i
            gt = self._gt_cache.get(image_filename)

            yield PageInput(
                document_id=document_id,
                page_number=page_num,
                image_path=image_path,
                ground_truth=gt,
                metadata={
                    "dataset": self.dataset_id,
                    "language": page_info.get("page_attribute", {}).get("language", ""),
                    "layout": page_info.get("page_attribute", {}).get("layout", ""),
                    "data_source": page_info.get("page_attribute", {}).get("data_source", ""),
                },
            )

    def get_ground_truth(self, document_id: str, page_number: int) -> str | None:
        self._load_annotations()
        # Search by document_id (stem of image filename)
        for image_path, gt in self._gt_cache.items():
            if Path(image_path).stem == document_id:
                return gt
        return None
