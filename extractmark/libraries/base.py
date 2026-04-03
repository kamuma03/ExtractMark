"""Base protocol for library adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from extractmark.types import PageInput, PageOutput


@runtime_checkable
class LibraryAdapter(Protocol):
    """Interface for Python parsing libraries."""

    lib_id: str

    def process_page(self, page: PageInput) -> PageOutput:
        """Process a single page and return extraction output."""
        ...

    def process_document(self, doc_path: Path) -> list[PageOutput]:
        """Process a full document and return per-page outputs."""
        ...
