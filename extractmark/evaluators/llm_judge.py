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
        self._configured_judge_model = judge_model
        self.temperature = temperature
        self.seed = seed
        self._client = openai.OpenAI(
            base_url=base_url, api_key="not-needed",
            timeout=120.0,
            max_retries=0,
        )
        self._prompt_template = self._load_prompt(prompt_path)
        self._served_model_name: str | None = None

    @property
    def judge_model(self) -> str:
        """Resolve the actual served model name from the vLLM server."""
        if self._served_model_name is not None:
            return self._served_model_name

        try:
            models_resp = self._client.models.list()
            available = [m.id for m in models_resp.data]
            if not available:
                self._served_model_name = self._configured_judge_model
            elif self._configured_judge_model in available:
                self._served_model_name = self._configured_judge_model
            else:
                self._served_model_name = available[0]
                logger.info(
                    "LLM judge: configured model '%s' not in served models %s, using '%s'",
                    self._configured_judge_model, available, self._served_model_name,
                )
        except Exception:
            self._served_model_name = self._configured_judge_model

        return self._served_model_name

    def _load_prompt(self, prompt_path: Path | None) -> str:
        path = prompt_path or _DEFAULT_PROMPT_PATH
        if path.exists():
            return path.read_text()
        return _FALLBACK_PROMPT

    def _call_judge(self, prompt: str, doc_id: str, page_num: int) -> str:
        """Send prompt to the judge LLM, retrying once on empty response."""
        messages = [
            {"role": "system", "content": "You are an expert document extraction evaluator. Always respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        # Use slightly above zero to avoid degenerate empty outputs from some models
        temp = max(self.temperature, 0.01)

        for attempt in range(3):
            kwargs = dict(
                model=self.judge_model,
                messages=list(messages),
                temperature=temp if attempt == 0 else 0.3,
                max_tokens=512,
            )
            if attempt == 0:
                kwargs["seed"] = self.seed

            # Try JSON mode (supported by vLLM >= 0.4); fall back gracefully
            if attempt == 0:
                try:
                    kwargs["response_format"] = {"type": "json_object"}
                    response = self._client.chat.completions.create(**kwargs)
                except Exception:
                    kwargs.pop("response_format", None)
                    response = self._client.chat.completions.create(**kwargs)
            else:
                response = self._client.chat.completions.create(**kwargs)

            choice = response.choices[0]
            raw_response = choice.message.content or ""

            logger.debug(
                "LLM judge response for %s page %d (attempt %d): "
                "finish_reason=%s, content_length=%d, usage=%s",
                doc_id, page_num, attempt + 1,
                choice.finish_reason, len(raw_response),
                getattr(response, "usage", None),
            )

            if raw_response.strip():
                return raw_response

            logger.warning(
                "LLM judge returned empty content for %s page %d "
                "(attempt %d, finish_reason=%s, prompt_length=%d chars)",
                doc_id, page_num, attempt + 1,
                choice.finish_reason, len(prompt),
            )

            # On retry, add an explicit nudge as a follow-up message
            if attempt == 0:
                messages.append({
                    "role": "user",
                    "content": "Please provide your evaluation now as a JSON object with keys: text_completeness, table_fidelity, reading_order, figure_caption, overall, reasoning.",
                })

        return raw_response

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
            raw_response = self._call_judge(prompt, output.document_id, output.page_number)
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
        import re

        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", response).strip()
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()

        # Try to extract JSON from cleaned response
        try:
            json_start = cleaned.find("{")
            json_end = cleaned.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(cleaned[json_start:json_end])
        except json.JSONDecodeError:
            pass

        # Try extracting individual scores via regex as fallback
        score_keys = ["text_completeness", "table_fidelity", "reading_order",
                      "figure_caption", "overall"]
        extracted = {}
        for key in score_keys:
            match = re.search(rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)', response)
            if match:
                extracted[key] = float(match.group(1))
        if extracted:
            extracted.setdefault("overall", 0.0)
            return extracted

        logger.warning("Could not parse JSON from LLM judge response, returning default. "
                       "Raw response (first 300 chars): %s", response[:300])
        return {"overall": 0.0, "parse_error": True, "raw_response": response[:200]}
