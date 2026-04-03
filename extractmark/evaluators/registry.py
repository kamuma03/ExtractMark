"""Evaluator registry -- resolves evaluator IDs to instances."""

from __future__ import annotations

from extractmark.config import EvalDefaults
from extractmark.evaluators.base import Evaluator
from extractmark.evaluators.edit_distance import EditDistanceEvaluator
from extractmark.evaluators.llm_judge import LLMJudgeEvaluator
from extractmark.evaluators.semantic_similarity import SemanticSimilarityEvaluator
from extractmark.evaluators.unit_tests import UnitTestEvaluator


def get_evaluator(
    evaluator_id: str,
    eval_config: EvalDefaults | None = None,
) -> Evaluator:
    """Create an evaluator instance for the given evaluator ID."""
    if eval_config is None:
        eval_config = EvalDefaults()

    if evaluator_id == "L1":
        return EditDistanceEvaluator()
    elif evaluator_id == "L2":
        return SemanticSimilarityEvaluator(model_name=eval_config.sbert_model)
    elif evaluator_id == "L3":
        return UnitTestEvaluator()
    elif evaluator_id == "L4":
        return LLMJudgeEvaluator(
            temperature=eval_config.judge_temperature,
            seed=eval_config.judge_seed,
        )
    else:
        raise ValueError(
            f"Unknown evaluator: {evaluator_id}. Available: L1, L2, L3, L4"
        )
