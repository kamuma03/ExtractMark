"""LIB-09 -- MinerU adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class MinerUAdapter:
    """Adapter for MinerU (LIB-09) -- hybrid rule+model PDF extraction."""

    lib_id = "LIB-09"

    def process_page(self, page: PageInput) -> PageOutput:
        return PageOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            raw_text="",
            metadata={"note": "MinerU works on PDF files. Use process_document()."},
        )

    def process_document(self, doc_path: Path) -> list[PageOutput]:
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        from magic_pdf.tools.common import do_parse

        document_id = doc_path.stem
        outputs: list[PageOutput] = []

        try:
            start = time.perf_counter()

            # Set up output paths
            output_dir = Path("results") / self.lib_id / document_id
            output_dir.mkdir(parents=True, exist_ok=True)
            image_dir = output_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            image_writer = FileBasedDataWriter(str(image_dir))

            # Read PDF bytes
            pdf_bytes = doc_path.read_bytes()

            # Run MinerU pipeline via the current API
            md_content = do_parse(
                pdf_bytes=pdf_bytes,
                model_list=[],
                image_writer=image_writer,
                is_debug=False,
                input_model_is_empty=True,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Split by page markers if present
            pages = md_content.split("\n---\n") if "\n---\n" in md_content else [md_content]

            for i, page_text in enumerate(pages):
                outputs.append(PageOutput(
                    document_id=document_id,
                    page_number=i,
                    raw_text=page_text,
                    inference_time_ms=elapsed_ms / len(pages),
                    metadata={"library": self.lib_id},
                ))

        except ImportError:
            logger.error("MinerU (magic-pdf) not installed. Run: pip install magic-pdf")
        except Exception as e:
            logger.error("MinerU failed on %s: %s", doc_path, e)

        return outputs
