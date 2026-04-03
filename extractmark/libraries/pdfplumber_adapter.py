"""LIB-02 -- pdfplumber adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class PdfplumberAdapter:
    """Adapter for pdfplumber (LIB-02) -- coordinate-level layout control."""

    lib_id = "LIB-02"

    def process_page(self, page: PageInput) -> PageOutput:
        return PageOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            raw_text="",
            metadata={"note": "pdfplumber works on PDF files. Use process_document()."},
        )

    def process_document(self, doc_path: Path) -> list[PageOutput]:
        import pdfplumber

        outputs: list[PageOutput] = []
        document_id = doc_path.stem

        try:
            with pdfplumber.open(str(doc_path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    start = time.perf_counter()
                    text = page.extract_text() or ""

                    # Extract tables
                    tables: list[str] = []
                    try:
                        for table in page.extract_tables():
                            if table:
                                md_rows = []
                                for row in table:
                                    md_rows.append(
                                        "| " + " | ".join(str(c or "") for c in row) + " |"
                                    )
                                tables.append("\n".join(md_rows))
                    except Exception as e:
                        logger.debug("Table extraction failed on page %d: %s", i, e)

                    elapsed_ms = (time.perf_counter() - start) * 1000

                    outputs.append(PageOutput(
                        document_id=document_id,
                        page_number=i,
                        raw_text=text,
                        tables=tables,
                        inference_time_ms=elapsed_ms,
                        metadata={"library": self.lib_id},
                    ))
        except Exception as e:
            logger.error("pdfplumber failed to open %s: %s", doc_path, e)

        return outputs
