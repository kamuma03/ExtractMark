"""Base protocol for model adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from extractmark.types import PageInput, PageOutput


@runtime_checkable
class ModelAdapter(Protocol):
    """Interface for OCR frontend models."""

    model_id: str

    def process_page(self, page: PageInput) -> PageOutput:
        """Process a single page image and return extraction output."""
        ...

    def health_check(self) -> bool:
        """Verify the model endpoint is responsive."""
        ...
