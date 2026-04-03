"""Base protocol for dataset loaders."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from extractmark.types import PageInput


@runtime_checkable
class DatasetLoader(Protocol):
    """Interface for loading benchmark datasets."""

    dataset_id: str

    def load(self) -> Iterator[PageInput]:
        """Yield PageInput items for all pages in the dataset."""
        ...

    def get_ground_truth(self, document_id: str, page_number: int) -> str | None:
        """Return ground truth text for a specific page, if available."""
        ...
