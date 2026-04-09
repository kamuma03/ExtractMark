"""LIB-08 -- Docling (IBM) adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class DoclingAdapter:
    """Adapter for Docling (LIB-08) -- layout-aware, multi-format extraction."""

    lib_id = "LIB-08"

    def process_page(self, page: PageInput) -> PageOutput:
        return PageOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            raw_text="",
            metadata={"note": "Docling works on document files. Use process_document()."},
        )

    def process_document(self, doc_path: Path) -> list[PageOutput]:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            AcceleratorOptions,
            AcceleratorDevice,
        )

        document_id = doc_path.stem
        outputs: list[PageOutput] = []

        try:
            start = time.perf_counter()
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.AUTO,
            )
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    ),
                }
            )
            result = converter.convert(str(doc_path))
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Docling returns a structured document; export as Markdown
            md_text = result.document.export_to_markdown()

            # Split by page if possible
            pages = md_text.split("\n---\n") if "\n---\n" in md_text else [md_text]

            # Extract tables from structured output
            tables_md: list[str] = []
            for table in result.document.tables:
                try:
                    table_md = table.export_to_markdown()
                    tables_md.append(table_md)
                except Exception:
                    pass

            for i, page_text in enumerate(pages):
                outputs.append(PageOutput(
                    document_id=document_id,
                    page_number=i,
                    raw_text=page_text,
                    tables=tables_md if i == 0 else [],
                    inference_time_ms=elapsed_ms / len(pages),
                    metadata={"library": self.lib_id},
                ))

        except Exception as e:
            logger.error("Docling failed on %s: %s", doc_path, e)

        return outputs
