"""LIB-17 -- pdfminer.six adapter."""

from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class PdfminerAdapter:
    """Adapter for pdfminer.six (LIB-17) -- low-level PDF text + layout."""

    lib_id = "LIB-17"

    def process_page(self, page: PageInput) -> PageOutput:
        return PageOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            raw_text="",
            metadata={"note": "pdfminer works on PDF files. Use process_document()."},
        )

    def process_document(self, doc_path: Path) -> list[PageOutput]:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextBox, LTTextLine, LTChar

        document_id = doc_path.stem
        outputs: list[PageOutput] = []

        try:
            start = time.perf_counter()

            for page_num, page_layout in enumerate(extract_pages(str(doc_path))):
                page_start = time.perf_counter()
                text_parts: list[str] = []
                bboxes: list[dict] = []

                for element in page_layout:
                    if isinstance(element, (LTTextBox, LTTextLine)):
                        text = element.get_text().strip()
                        if text:
                            text_parts.append(text)
                            bbox = element.bbox  # (x0, y0, x1, y1)
                            bboxes.append({
                                "x1": bbox[0], "y1": bbox[1],
                                "x2": bbox[2], "y2": bbox[3],
                                "text": text,
                            })

                page_elapsed = (time.perf_counter() - page_start) * 1000

                outputs.append(PageOutput(
                    document_id=document_id,
                    page_number=page_num,
                    raw_text="\n".join(text_parts),
                    bboxes=bboxes if bboxes else None,
                    inference_time_ms=page_elapsed,
                    metadata={"library": self.lib_id},
                ))

        except ImportError:
            logger.error("pdfminer.six not installed. Run: pip install pdfminer.six")
        except Exception as e:
            logger.error("pdfminer failed on %s: %s", doc_path, e)

        return outputs
