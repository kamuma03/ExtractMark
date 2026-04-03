"""LIB-03 -- pypdfium2 adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class Pypdfium2Adapter:
    """Adapter for pypdfium2 (LIB-03) -- blazing speed raw text baseline."""

    lib_id = "LIB-03"

    def process_page(self, page: PageInput) -> PageOutput:
        return PageOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            raw_text="",
            metadata={"note": "pypdfium2 works on PDF files. Use process_document()."},
        )

    def process_document(self, doc_path: Path) -> list[PageOutput]:
        import pypdfium2 as pdfium

        outputs: list[PageOutput] = []
        document_id = doc_path.stem

        try:
            pdf = pdfium.PdfDocument(str(doc_path))
            for i in range(len(pdf)):
                start = time.perf_counter()
                page = pdf[i]
                textpage = page.get_textpage()
                text = textpage.get_text_range()
                textpage.close()
                page.close()
                elapsed_ms = (time.perf_counter() - start) * 1000

                outputs.append(PageOutput(
                    document_id=document_id,
                    page_number=i,
                    raw_text=text,
                    inference_time_ms=elapsed_ms,
                    metadata={"library": self.lib_id},
                ))
            pdf.close()
        except Exception as e:
            logger.error("pypdfium2 failed to open %s: %s", doc_path, e)

        return outputs
