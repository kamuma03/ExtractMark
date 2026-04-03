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
        from magic_pdf.pipe.UNIPipe import UNIPipe

        document_id = doc_path.stem
        outputs: list[PageOutput] = []

        try:
            start = time.perf_counter()

            # Read PDF bytes
            pdf_bytes = doc_path.read_bytes()

            # Set up output paths
            output_dir = Path("results") / self.lib_id / document_id
            output_dir.mkdir(parents=True, exist_ok=True)

            image_writer = FileBasedDataWriter(str(output_dir / "images"))
            reader = FileBasedDataReader("")

            # Run MinerU pipeline
            pipe = UNIPipe(pdf_bytes, {"_pdf_type": "", "model_list": []}, image_writer)
            pipe.pipe_classify()
            pipe.pipe_analyze()
            pipe.pipe_parse()

            md_content = pipe.pipe_mk_markdown(str(output_dir / "images"), drop_mode="none")
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
            logger.error("MinerU (magic-pdf) not installed. Run: pip install mineru")
        except Exception as e:
            logger.error("MinerU failed on %s: %s", doc_path, e)

        return outputs
