"""LIB-11 -- Surya adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from PIL import Image

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class SuryaAdapter:
    """Adapter for Surya (LIB-11) -- standalone layout detection + OCR."""

    lib_id = "LIB-11"

    def process_page(self, page: PageInput) -> PageOutput:
        """Process a single page image using Surya OCR."""
        from surya.ocr import run_ocr
        from surya.model.detection.model import load_model as load_det_model
        from surya.model.detection.model import load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor

        start = time.perf_counter()

        try:
            image = Image.open(page.image_path)

            det_model = load_det_model()
            det_processor = load_det_processor()
            rec_model = load_rec_model()
            rec_processor = load_rec_processor()

            results = run_ocr(
                [image],
                [["en"]],
                det_model, det_processor,
                rec_model, rec_processor,
            )

            # Extract text lines and bboxes
            text_lines = []
            bboxes = []
            if results and results[0]:
                for line in results[0].text_lines:
                    text_lines.append(line.text)
                    if hasattr(line, "bbox"):
                        bboxes.append({
                            "x1": line.bbox[0], "y1": line.bbox[1],
                            "x2": line.bbox[2], "y2": line.bbox[3],
                            "text": line.text,
                        })

            elapsed_ms = (time.perf_counter() - start) * 1000

            return PageOutput(
                document_id=page.document_id,
                page_number=page.page_number,
                raw_text="\n".join(text_lines),
                bboxes=bboxes if bboxes else None,
                inference_time_ms=elapsed_ms,
                metadata={"library": self.lib_id},
            )

        except ImportError:
            logger.error("Surya not installed. Run: pip install surya-ocr")
        except Exception as e:
            logger.error("Surya failed on %s page %d: %s",
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
        """Process a PDF by converting pages to images first."""
        outputs: list[PageOutput] = []
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(str(doc_path))
            for i, img in enumerate(images):
                # Save temp image
                tmp_path = Path(f"/tmp/surya_page_{i}.png")
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
            logger.error("Surya document processing failed on %s: %s", doc_path, e)
        return outputs
