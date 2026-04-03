"""LIB-10 -- marker adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class MarkerAdapter:
    """Adapter for marker (LIB-10) -- Surya OCR + layout detection, Markdown output."""

    lib_id = "LIB-10"

    def process_page(self, page: PageInput) -> PageOutput:
        return PageOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            raw_text="",
            metadata={"note": "marker works on document files. Use process_document()."},
        )

    def process_document(self, doc_path: Path) -> list[PageOutput]:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        document_id = doc_path.stem
        outputs: list[PageOutput] = []

        try:
            start = time.perf_counter()
            model_dict = create_model_dict()
            converter = PdfConverter(artifact_dict=model_dict)
            rendered = converter(str(doc_path))
            elapsed_ms = (time.perf_counter() - start) * 1000

            # rendered.markdown contains full document
            md_text = rendered.markdown

            # Split by page markers
            pages = md_text.split("\n---\n") if "\n---\n" in md_text else [md_text]

            for i, page_text in enumerate(pages):
                outputs.append(PageOutput(
                    document_id=document_id,
                    page_number=i,
                    raw_text=page_text,
                    inference_time_ms=elapsed_ms / len(pages),
                    metadata={"library": self.lib_id},
                ))

        except ImportError:
            logger.error("marker not installed. Run: pip install marker-pdf")
        except Exception as e:
            logger.error("marker failed on %s: %s", doc_path, e)

        return outputs
