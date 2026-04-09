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
        from magic_pdf.tools.common import do_parse

        document_id = doc_path.stem
        outputs: list[PageOutput] = []

        try:
            start = time.perf_counter()

            # Set up output directory for MinerU
            output_dir = Path("results") / self.lib_id / document_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Read PDF bytes
            pdf_bytes = doc_path.read_bytes()

            # Run MinerU pipeline via do_parse (positional args API)
            do_parse(
                output_dir=str(output_dir),
                pdf_file_name=document_id,
                pdf_bytes_or_dataset=pdf_bytes,
                model_list=[],
                parse_method="auto",
                f_draw_span_bbox=False,
                f_draw_layout_bbox=False,
                f_dump_md=True,
                f_dump_middle_json=False,
                f_dump_model_json=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
                formula_enable=False,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Read the generated Markdown file
            md_path = output_dir / document_id / "auto" / f"{document_id}.md"
            if not md_path.exists():
                # Try alternative path structures
                for candidate in output_dir.rglob("*.md"):
                    md_path = candidate
                    break

            if md_path.exists():
                md_content = md_path.read_text(encoding="utf-8")
            else:
                md_content = ""
                logger.warning("MinerU produced no Markdown output for %s", doc_path)

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

        except (ImportError, ModuleNotFoundError) as e:
            logger.error("MinerU dependency missing: %s. Run: pip install magic-pdf[full]", e)
        except Exception as e:
            logger.error("MinerU failed on %s: %s", doc_path, e)

        return outputs
