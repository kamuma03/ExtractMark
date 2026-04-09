"""LIB-01 -- PyMuPDF / pymupdf4llm adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class PyMuPDFAdapter:
    """Adapter for PyMuPDF (LIB-01) -- fastest text extraction for digital PDFs."""

    lib_id = "LIB-01"

    def process_page(self, page: PageInput) -> PageOutput:
        """Process a page image. For PyMuPDF this is a no-op since it works on PDFs directly."""
        return PageOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            raw_text="",
            metadata={"note": "PyMuPDF works on PDF files, not images. Use process_document()."},
        )

    def process_document(self, doc_path: Path) -> list[PageOutput]:
        """Process a PDF document using PyMuPDF + pymupdf4llm."""
        import pymupdf

        outputs: list[PageOutput] = []
        try:
            doc = pymupdf.open(str(doc_path))
        except Exception as e:
            logger.error("PyMuPDF failed to open %s: %s", doc_path, e)
            return outputs

        document_id = doc_path.stem

        # Try pymupdf4llm for Markdown output
        try:
            import pymupdf4llm
            start = time.perf_counter()
            md_text = pymupdf4llm.to_markdown(str(doc_path))
            elapsed_ms = (time.perf_counter() - start) * 1000

            # pymupdf4llm returns full document as Markdown; split by page markers
            pages = md_text.split("\n---\n") if "\n---\n" in md_text else [md_text]
            for i, page_text in enumerate(pages):
                outputs.append(PageOutput(
                    document_id=document_id,
                    page_number=i,
                    raw_text=page_text,
                    inference_time_ms=elapsed_ms / len(pages),
                    metadata={"library": self.lib_id, "method": "pymupdf4llm"},
                ))
        except (ImportError, AttributeError):
            # Fallback to plain PyMuPDF text extraction
            for i, page in enumerate(doc):
                start = time.perf_counter()
                text = page.get_text("text")
                elapsed_ms = (time.perf_counter() - start) * 1000

                # Extract tables
                tables = []
                try:
                    for table in page.find_tables():
                        table_data = table.extract()
                        if table_data:
                            # Convert to pipe-delimited Markdown
                            md_rows = []
                            for row in table_data:
                                md_rows.append("| " + " | ".join(str(c or "") for c in row) + " |")
                            tables.append("\n".join(md_rows))
                except Exception:
                    pass

                outputs.append(PageOutput(
                    document_id=document_id,
                    page_number=i,
                    raw_text=text,
                    tables=tables,
                    inference_time_ms=elapsed_ms,
                    metadata={"library": self.lib_id, "method": "pymupdf"},
                ))

        doc.close()
        return outputs
