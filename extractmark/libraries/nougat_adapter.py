"""LIB-16 -- Nougat (Meta) adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


class NougatAdapter:
    """Adapter for Nougat (LIB-16) -- scientific PDF parsing."""

    lib_id = "LIB-16"

    def process_page(self, page: PageInput) -> PageOutput:
        return PageOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            raw_text="",
            metadata={"note": "Nougat works on PDF files. Use process_document()."},
        )

    def process_document(self, doc_path: Path) -> list[PageOutput]:
        import subprocess
        import tempfile

        document_id = doc_path.stem
        outputs: list[PageOutput] = []

        try:
            start = time.perf_counter()

            # Nougat is typically run as a CLI tool
            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(
                    ["nougat", str(doc_path), "-o", tmpdir, "--markdown"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode == 0:
                    # Read output Markdown
                    mmd_files = list(Path(tmpdir).glob("*.mmd"))
                    if mmd_files:
                        text = mmd_files[0].read_text()
                    else:
                        text = result.stdout

                    elapsed_ms = (time.perf_counter() - start) * 1000

                    outputs.append(PageOutput(
                        document_id=document_id,
                        page_number=0,
                        raw_text=text,
                        inference_time_ms=elapsed_ms,
                        metadata={"library": self.lib_id},
                    ))
                else:
                    logger.error("Nougat CLI failed: %s", result.stderr)

        except FileNotFoundError:
            logger.error("Nougat CLI not found. Run: pip install nougat-ocr")
        except Exception as e:
            logger.error("Nougat failed on %s: %s", doc_path, e)

        return outputs
