"""LIB-14 -- Tesseract (via pytesseract) adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from PIL import Image

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class TesseractAdapter:
    """Adapter for Tesseract (LIB-14) -- classic CPU OCR engine."""

    lib_id = "LIB-14"

    def process_page(self, page: PageInput) -> PageOutput:
        """Process a single page image using Tesseract OCR."""
        import pytesseract

        start = time.perf_counter()

        try:
            image = Image.open(page.image_path)
            text = pytesseract.image_to_string(image)

            # Get bounding box data
            bboxes = []
            try:
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                for i in range(len(data["text"])):
                    if data["text"][i].strip():
                        bboxes.append({
                            "x1": data["left"][i],
                            "y1": data["top"][i],
                            "x2": data["left"][i] + data["width"][i],
                            "y2": data["top"][i] + data["height"][i],
                            "text": data["text"][i],
                            "confidence": data["conf"][i],
                        })
            except Exception:
                pass

            elapsed_ms = (time.perf_counter() - start) * 1000

            return PageOutput(
                document_id=page.document_id,
                page_number=page.page_number,
                raw_text=text,
                bboxes=bboxes if bboxes else None,
                inference_time_ms=elapsed_ms,
                metadata={"library": self.lib_id},
            )

        except ImportError:
            logger.error("pytesseract not installed. Run: pip install pytesseract")
        except Exception as e:
            logger.error("Tesseract failed on %s page %d: %s",
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
                tmp_path = Path(f"/tmp/tesseract_page_{i}.png")
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
            logger.error("Tesseract document processing failed on %s: %s", doc_path, e)
        return outputs
