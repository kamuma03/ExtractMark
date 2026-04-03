"""LIB-13 -- MarkItDown (Microsoft) adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class MarkItDownAdapter:
    """Adapter for MarkItDown (LIB-13) -- zero-dependency Markdown conversion."""

    lib_id = "LIB-13"

    def process_page(self, page: PageInput) -> PageOutput:
        return PageOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            raw_text="",
            metadata={"note": "MarkItDown works on document files. Use process_document()."},
        )

    def process_document(self, doc_path: Path) -> list[PageOutput]:
        from markitdown import MarkItDown

        document_id = doc_path.stem

        try:
            start = time.perf_counter()
            md = MarkItDown()
            result = md.convert(str(doc_path))
            elapsed_ms = (time.perf_counter() - start) * 1000

            text = result.text_content

            return [PageOutput(
                document_id=document_id,
                page_number=0,
                raw_text=text,
                inference_time_ms=elapsed_ms,
                metadata={"library": self.lib_id},
            )]

        except ImportError:
            logger.error("MarkItDown not installed. Run: pip install markitdown")
        except Exception as e:
            logger.error("MarkItDown failed on %s: %s", doc_path, e)

        return []
