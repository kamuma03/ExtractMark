"""Model registry -- resolves model IDs to adapter instances."""

from __future__ import annotations

from extractmark.config import ModelConfig
from extractmark.models.base import ModelAdapter
from extractmark.models.vllm_model import VLLMModelAdapter


def get_model(model_id: str, config: ModelConfig) -> ModelAdapter:
    """Create a model adapter for the given model ID.

    All models (M-01..M-13, L-01..L-03) use VLLMModelAdapter since they are
    all served via vLLM with an OpenAI-compatible API. The differences are
    captured in ModelConfig (prompt template, output format, bbox support).
    """
    return VLLMModelAdapter(model_id, config)
