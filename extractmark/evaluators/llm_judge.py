"""L4 -- LLM-as-Judge evaluator.

Uses a local LLM (Qwen2.5-7B / L-01) served via vLLM to score extraction
quality on dimensions: text completeness, table fidelity, reading order,
and figure/caption handling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import openai

from extractmark.types import EvalResult, PageOutput

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = Path(__file__).parent.parent.parent / "eval" / "prompts" / "judge_v1.txt"

_FALLBACK_PROMPT = """You are an expert document extraction evaluator. Compare the extracted text against the source document and score it on a 0-10 scale.

Evaluate across these dimensions:
1. **Text completeness** (0-10): Are all text elements captured?
2. **Table fidelity** (0-10): Are tables correctly structured with accurate cell content?
3. **Reading order** (0-10): Is the text in the correct reading sequence?
4. **Figure/caption handling** (0-10): Are figures detected and captions captured?

Provide your response as JSON:
{
  "text_completeness": <score>,
  "table_fidelity": <score>,
  "reading_order": <score>,
  "figure_caption": <score>,
  "overall": <score>,
  "reasoning": "<brief explanation>"
}

--- EXTRACTED TEXT ---
{extracted_text}

--- GROUND TRUTH ---
{ground_truth}
"""


class LLMJudgeEvaluator:
    """L4: LLM-as-judge for holistic extraction quality assessment."""

    metric_id = "L4"

    def __init__(
        self,
        judge_model: str = "Qwen/Qwen2.5-7B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.0,
        seed: int = 42,
        prompt_path: Path | None = None,
    ):
        self.judge_model = judge_model
        self.temperature = temperature
        self.seed = seed
        self._client = openai.OpenAI(base_url=base_url, api_key="not-needed")
        self._prompt_template = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: Path | None) -> str:
        path = prompt_path or _DEFAULT_PROMPT_PATH
        if path.exists():
            return path.read_text()
        return _FALLBACK_PROMPT

    def evaluate(self, output: PageOutput, ground_truth: str) -> list[EvalResult]:
        text = output.normalized_text or output.raw_text
        if not text or not ground_truth:
            return [
                EvalResult(
                    metric_name="llm_judge_overall",
                    score=0.0,
                    details={"error": "empty_input"},
                )
            ]

        prompt = self._prompt_template.format(
            extracted_text=text[:4000],  # Truncate to fit context
            ground_truth=ground_truth[:4000],
        )

        try:
            response = self._client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=512,
            )
            raw_response = response.choices[0].message.content or ""
            scores = self._parse_scores(raw_response)
        except Exception as e:
            logger.warning("LLM judge failed for %s page %d: %s",
                           output.document_id, output.page_number, e)
            scores = {"overall": 0.0, "error": str(e)}

        results = []
        overall = scores.get("overall", 0.0)
        results.append(
            EvalResult(
                metric_name="llm_judge_overall",
                score=float(overall),
                details={
                    "document_id": output.document_id,
                    "page_number": output.page_number,
                    "scores": scores,
                    "judge_model": self.judge_model,
                },
            )
        )

        # Add sub-dimension scores
        for dim in ["text_completeness", "table_fidelity", "reading_order", "figure_caption"]:
            if dim in scores:
                results.append(
                    EvalResult(
                        metric_name=f"llm_judge_{dim}",
                        score=float(scores[dim]),
                        details={
                            "document_id": output.document_id,
                            "page_number": output.page_number,
                        },
                    )
                )

        return results

    @staticmethod
    def _parse_scores(response: str) -> dict:
        """Parse JSON scores from LLM response."""
        # Try to extract JSON from response
        try:
            # Look for JSON block in response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract numeric score
        logger.warning("Could not parse JSON from LLM judge response, returning default")
        return {"overall": 0.0, "parse_error": True, "raw_response": response[:200]}
