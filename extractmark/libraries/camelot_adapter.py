"""LIB-04 -- Camelot adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class CamelotAdapter:
    """Adapter for Camelot (LIB-04) -- best lattice table extraction."""

    lib_id = "LIB-04"

    def process_page(self, page: PageInput) -> PageOutput:
        return PageOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            raw_text="",
            metadata={"note": "Camelot works on PDF files. Use process_document()."},
        )

    def process_document(self, doc_path: Path) -> list[PageOutput]:
        import camelot

        outputs: list[PageOutput] = []
        document_id = doc_path.stem

        try:
            # Get page count first
            import pymupdf
            doc = pymupdf.open(str(doc_path))
            num_pages = len(doc)
            doc.close()
        except ImportError:
            logger.warning("pymupdf not available for page count; defaulting to page 1")
            num_pages = 1

        for page_num in range(1, num_pages + 1):
            start = time.perf_counter()
            tables_md: list[str] = []

            try:
                # Try lattice mode first (bordered tables)
                tables = camelot.read_pdf(
                    str(doc_path), pages=str(page_num), flavor="lattice"
                )
                if not tables:
                    # Fallback to stream mode (borderless tables)
                    tables = camelot.read_pdf(
                        str(doc_path), pages=str(page_num), flavor="stream"
                    )

                for table in tables:
                    df = table.df
                    md_rows = []
                    for _, row in df.iterrows():
                        md_rows.append("| " + " | ".join(str(c) for c in row) + " |")
                    if md_rows:
                        # Add separator after header
                        header = md_rows[0]
                        sep = "| " + " | ".join("---" for _ in row) + " |"
                        tables_md.append(header + "\n" + sep + "\n" + "\n".join(md_rows[1:]))

            except Exception as e:
                logger.debug("Camelot failed on page %d: %s", page_num, e)

            elapsed_ms = (time.perf_counter() - start) * 1000

            outputs.append(PageOutput(
                document_id=document_id,
                page_number=page_num - 1,
                raw_text="\n\n".join(tables_md),
                tables=tables_md,
                inference_time_ms=elapsed_ms,
                metadata={"library": self.lib_id},
            ))

        return outputs
