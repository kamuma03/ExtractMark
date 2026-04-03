"""Dataset registry -- resolves dataset IDs to loader instances."""

from __future__ import annotations

from extractmark.config import DatasetConfig
from extractmark.datasets.base import DatasetLoader
from extractmark.datasets.doclaynet import DocLayNetLoader
from extractmark.datasets.docvqa import DocVQALoader
from extractmark.datasets.fintabnet import FinTabNetLoader
from extractmark.datasets.funsd import FUNSDLoader
from extractmark.datasets.olmocr_bench import OlmOCRBenchLoader
from extractmark.datasets.omnidocbench import OmniDocBenchLoader

_LOADER_MAP: dict[str, type] = {
    "omnidocbench": OmniDocBenchLoader,
    "fintabnet": FinTabNetLoader,
    "funsd": FUNSDLoader,
    "docvqa": DocVQALoader,
    "olmocr_bench": OlmOCRBenchLoader,
    "doclaynet": DocLayNetLoader,
}


def get_dataset(dataset_id: str, config: DatasetConfig) -> DatasetLoader:
    """Create a dataset loader for the given dataset ID."""
    loader_cls = _LOADER_MAP.get(config.loader)
    if loader_cls is None:
        raise ValueError(
            f"Unknown dataset loader: {config.loader}. "
            f"Available: {list(_LOADER_MAP.keys())}"
        )
    return loader_cls(dataset_id=dataset_id, path=config.path)
