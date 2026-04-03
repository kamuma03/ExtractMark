"""LIB-15 -- Table Transformer (TATR) adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from PIL import Image

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class TableTransformerAdapter:
    """Adapter for Table Transformer (LIB-15) -- table detection + structure."""

    lib_id = "LIB-15"

    def __init__(self):
        self._detection_model = None
        self._structure_model = None
        self._processor = None

    def _load_models(self):
        """Lazy-load Table Transformer models."""
        if self._detection_model is not None:
            return

        from transformers import AutoModelForObjectDetection, AutoImageProcessor

        self._processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        self._detection_model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        self._structure_model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )

    def process_page(self, page: PageInput) -> PageOutput:
        """Detect and extract table structure from a page image."""
        import torch

        start = time.perf_counter()

        try:
            self._load_models()
            image = Image.open(page.image_path).convert("RGB")

            # Detect tables
            inputs = self._processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self._detection_model(**inputs)

            # Post-process detections
            target_sizes = torch.tensor([image.size[::-1]])
            results = self._processor.post_process_object_detection(
                outputs, threshold=0.7, target_sizes=target_sizes
            )[0]

            bboxes = []
            tables_md = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = box.tolist()
                bboxes.append({
                    "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                    "label": self._detection_model.config.id2label[label.item()],
                    "confidence": score.item(),
                })

            elapsed_ms = (time.perf_counter() - start) * 1000

            return PageOutput(
                document_id=page.document_id,
                page_number=page.page_number,
                raw_text=f"Detected {len(bboxes)} table region(s)",
                tables=tables_md,
                bboxes=bboxes if bboxes else None,
                inference_time_ms=elapsed_ms,
                metadata={"library": self.lib_id, "detections": len(bboxes)},
            )

        except ImportError:
            logger.error("transformers/timm not installed. Run: pip install transformers timm")
        except Exception as e:
            logger.error("Table Transformer failed on %s page %d: %s",
                         page.document_id, page.page_number, e)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return PageOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            raw_text="",
            inference_time_ms=elapsed_ms,
            metadata={"library": self.lib_id, "error": True},
        )

    def process_document(self, doc_path: Path) -> list[PageOutput]:
        """Process a PDF by converting pages to images."""
        outputs: list[PageOutput] = []
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(str(doc_path))
            for i, img in enumerate(images):
                tmp_path = Path(f"/tmp/tatr_page_{i}.png")
                img.save(str(tmp_path))
                page_input = PageInput(
                    document_id=doc_path.stem,
                    page_number=i,
                    image_path=tmp_path,
                )
                outputs.append(self.process_page(page_input))
        except ImportError:
            logger.error("pdf2image not installed. Run: pip install pdf2image")
        except Exception as e:
            logger.error("Table Transformer document processing failed: %s", e)
        return outputs
