"""Generic vLLM model adapter -- handles all models served via OpenAI-compatible API."""

from __future__ import annotations

import base64
import logging
import re
import time
from pathlib import Path

import openai

from extractmark.config import ModelConfig
from extractmark.models.prompt_templates import get_template
from extractmark.serving.gpu_monitor import get_gpu_memory_mb
from extractmark.types import PageInput, PageOutput

logger = logging.getLogger(__name__)


def _encode_image(image_path: Path) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_tables_from_markdown(text: str) -> list[str]:
    """Extract pipe-delimited Markdown tables from text."""
    tables: list[str] = []
    current_table: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current_table.append(stripped)
        else:
            if current_table:
                tables.append("\n".join(current_table))
                current_table = []
    if current_table:
        tables.append("\n".join(current_table))
    return tables


def _extract_bboxes(text: str) -> list[dict]:
    """Extract bounding box coordinates from model output.

    Supports multiple formats:
    - Nemotron Parse: <x_0.05><y_0.25>text<x_0.85><y_0.39><class_Text>
    - Generic: <box>x1,y1,x2,y2</box> or <bbox>...</bbox>
    - Bracket: [x1, y1, x2, y2]
    """
    bboxes: list[dict] = []

    # Nemotron Parse format: <x_N><y_N>text<x_N><y_N><class_XXX>
    nemotron_pattern = re.compile(
        r"<x_([\d.]+)><y_([\d.]+)>(.*?)<x_([\d.]+)><y_([\d.]+)><class_(\w[\w-]*)>"
    )
    for match in nemotron_pattern.finditer(text):
        bboxes.append({
            "x1": float(match.group(1)),
            "y1": float(match.group(2)),
            "x2": float(match.group(4)),
            "y2": float(match.group(5)),
            "text": match.group(3).strip(),
            "class": match.group(6),
        })

    if bboxes:
        return bboxes

    # Pattern: <box>x1,y1,x2,y2</box> or <bbox>...</bbox>
    box_pattern = re.compile(r"<(?:box|bbox)>([\d.,\s]+)</(?:box|bbox)>")
    for match in box_pattern.finditer(text):
        coords = [float(c.strip()) for c in match.group(1).split(",")]
        if len(coords) == 4:
            bboxes.append({"x1": coords[0], "y1": coords[1], "x2": coords[2], "y2": coords[3]})

    # Pattern: [[x1, y1, x2, y2]]
    bracket_pattern = re.compile(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]")
    for match in bracket_pattern.finditer(text):
        bboxes.append({
            "x1": float(match.group(1)),
            "y1": float(match.group(2)),
            "x2": float(match.group(3)),
            "y2": float(match.group(4)),
        })

    return bboxes


class VLLMModelAdapter:
    """Adapter for any model served via vLLM OpenAI-compatible API.

    All 13 OCR models and 3 LLM backbone candidates are handled by this
    single class -- differences are captured in ModelConfig (YAML), not code.
    """

    def __init__(self, model_id: str, config: ModelConfig):
        self.model_id = model_id
        self.config = config
        self.base_url = f"http://localhost:{config.port}/v1"
        self._client = openai.OpenAI(base_url=self.base_url, api_key="not-needed")
        self._template_fn = get_template(config.prompt_template)

    def process_page(self, page: PageInput) -> PageOutput:
        """Process a single page image via vLLM chat completions."""
        image_b64 = _encode_image(page.image_path)
        messages = self._template_fn(image_b64)

        gen_params = {
            "temperature": self.config.generation_params.temperature,
            "max_tokens": self.config.generation_params.max_tokens,
            "seed": self.config.generation_params.seed,
        }
        if self.config.generation_params.top_k is not None:
            gen_params["extra_body"] = {"top_k": self.config.generation_params.top_k}

        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.config.hf_model_id,
                messages=messages,
                **gen_params,
            )
            raw_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Inference failed for %s on %s page %d: %s",
                         self.model_id, page.document_id, page.page_number, e)
            raw_text = ""
        elapsed_ms = (time.perf_counter() - start) * 1000

        gpu_mem = get_gpu_memory_mb()

        tables = _extract_tables_from_markdown(raw_text)
        bboxes = _extract_bboxes(raw_text) if self.config.supports_bbox else None

        return PageOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            raw_text=raw_text,
            tables=tables,
            bboxes=bboxes,
            inference_time_ms=elapsed_ms,
            gpu_memory_mb=gpu_mem,
        )

    def health_check(self) -> bool:
        """Check if the vLLM server is responding."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
