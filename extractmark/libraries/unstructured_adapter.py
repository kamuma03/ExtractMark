"""LIB-12 -- Unstructured adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class UnstructuredAdapter:
    """Adapter for Unstructured (LIB-12) -- semantic element partitioning."""

    lib_id = "LIB-12"

    def process_page(self, page: PageInput) -> PageOutput:
        return PageOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            raw_text="",
            metadata={"note": "Unstructured works on document files. Use process_document()."},
        )

    def process_document(self, doc_path: Path) -> list[PageOutput]:
        from unstructured.partition.auto import partition

        document_id = doc_path.stem
        outputs: list[PageOutput] = []

        try:
            start = time.perf_counter()
            elements = partition(filename=str(doc_path))
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Group elements by page
            pages: dict[int, list] = {}
            for el in elements:
                page_num = getattr(el.metadata, "page_number", 1) or 1
                page_num -= 1  # 0-indexed
                if page_num not in pages:
                    pages[page_num] = []
                pages[page_num].append(el)

            for page_num in sorted(pages.keys()):
                text_parts = []
                tables_md = []

                for el in pages[page_num]:
                    el_type = type(el).__name__
                    if el_type == "Table":
                        # Convert table to Markdown
                        if hasattr(el, "metadata") and hasattr(el.metadata, "text_as_html"):
                            tables_md.append(el.metadata.text_as_html)
                        else:
                            tables_md.append(str(el))
                    elif el_type == "Title":
                        text_parts.append(f"# {el.text}")
                    elif el_type == "Header":
                        text_parts.append(f"## {el.text}")
                    else:
                        text_parts.append(el.text)

                full_text = "\n\n".join(text_parts)
                if tables_md:
                    full_text += "\n\n" + "\n\n".join(tables_md)

                outputs.append(PageOutput(
                    document_id=document_id,
                    page_number=page_num,
                    raw_text=full_text,
                    tables=tables_md,
                    inference_time_ms=elapsed_ms / max(len(pages), 1),
                    metadata={"library": self.lib_id, "elements": len(pages[page_num])},
                ))

        except ImportError:
            logger.error("Unstructured not installed. Run: pip install unstructured[all-docs]")
        except Exception as e:
            logger.error("Unstructured failed on %s: %s", doc_path, e)

        return outputs
