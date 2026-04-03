"""L2 -- Semantic Similarity evaluator (SBERT cosine similarity)."""

from __future__ import annotations

import logging

from extractmark.types import EvalResult, PageOutput

logger = logging.getLogger(__name__)

_model_cache = {}


def _get_sbert_model(model_name: str = "all-MiniLM-L12-v2"):
    """Lazy-load SBERT model (cached after first call)."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class SemanticSimilarityEvaluator:
    """L2: SBERT cosine similarity between extracted and ground truth text."""

    metric_id = "L2"

    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        self.model_name = model_name

    def evaluate(self, output: PageOutput, ground_truth: str) -> list[EvalResult]:
        text = output.normalized_text or output.raw_text
        if not text or not ground_truth:
            return [
                EvalResult(metric_name="sbert_cosine", score=0.0, details={"error": "empty_input"})
            ]

        try:
            from sentence_transformers import util

            model = _get_sbert_model(self.model_name)
            embeddings = model.encode([text, ground_truth], convert_to_tensor=True)
            cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        except Exception as e:
            logger.warning("SBERT computation failed: %s", e)
            cosine_sim = 0.0

        return [
            EvalResult(
                metric_name="sbert_cosine",
                score=cosine_sim,
                details={
                    "document_id": output.document_id,
                    "page_number": output.page_number,
                    "model": self.model_name,
                },
            )
        ]
