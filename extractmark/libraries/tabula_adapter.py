"""LIB-05 -- Tabula-py adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class TabulaAdapter:
    """Adapter for Tabula-py (LIB-05) -- table extraction via Java PDFBox."""

    lib_id = "LIB-05"

    def process_page(self, page: PageInput) -> PageOutput:
        return PageOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            raw_text="",
            metadata={"note": "Tabula works on PDF files. Use process_document()."},
        )

    def process_document(self, doc_path: Path) -> list[PageOutput]:
        import tabula

        outputs: list[PageOutput] = []
        document_id = doc_path.stem

        try:
            start = time.perf_counter()
            # Extract all tables from all pages
            dfs = tabula.read_pdf(
                str(doc_path),
                pages="all",
                multiple_tables=True,
                pandas_options={"header": None},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Group tables -- tabula doesn't track page numbers reliably,
            # so we create one output with all tables
            tables_md: list[str] = []
            for df in dfs:
                md_rows = []
                for _, row in df.iterrows():
                    md_rows.append("| " + " | ".join(str(c) for c in row) + " |")
                if md_rows:
                    tables_md.append("\n".join(md_rows))

            outputs.append(PageOutput(
                document_id=document_id,
                page_number=0,
                raw_text="\n\n".join(tables_md),
                tables=tables_md,
                inference_time_ms=elapsed_ms,
                metadata={"library": self.lib_id, "total_tables": len(dfs)},
            ))

        except Exception as e:
            logger.error("Tabula failed on %s: %s", doc_path, e)

        return outputs
