"""Prompt templates for different OCR models served via vLLM.

Each template returns the messages list for the OpenAI chat completions API.
Add new templates here when onboarding models with unique prompt formats.
"""

from __future__ import annotations


def _image_content(image_b64: str, media_type: str = "image/png") -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
    }


def generic_ocr(image_b64: str) -> list[dict]:
    """Generic OCR prompt -- works for most models."""
    return [
        {
            "role": "user",
            "content": [
                _image_content(image_b64),
                {
                    "type": "text",
                    "text": "Extract all text from this document image. "
                    "Output the content in Markdown format, preserving the original "
                    "structure including headings, paragraphs, tables, and lists.",
                },
            ],
        }
    ]


def nemotron_parse(image_b64: str) -> list[dict]:
    """Nemotron Parse v1.1 / v1.2 prompt format.

    Uses the model's specific task prompt tokens for structured output
    with bounding boxes, classes, and Markdown formatting.
    """
    return [
        {
            "role": "user",
            "content": [
                _image_content(image_b64),
                {
                    "type": "text",
                    "text": "</s><s><predict_bbox><predict_classes><output_markdown>",
                },
            ],
        }
    ]


def qwen_vl(image_b64: str) -> list[dict]:
    """Qwen2.5-VL prompt format."""
    return [
        {
            "role": "user",
            "content": [
                _image_content(image_b64),
                {
                    "type": "text",
                    "text": "Please extract all the text content from this document image. "
                    "Preserve the structure (headings, paragraphs, tables, lists) and "
                    "output in Markdown format.",
                },
            ],
        }
    ]


def glm_ocr(image_b64: str) -> list[dict]:
    """GLM-OCR prompt format."""
    return [
        {
            "role": "user",
            "content": [
                _image_content(image_b64),
                {
                    "type": "text",
                    "text": "OCR this document image. Extract all text preserving layout, "
                    "tables, and reading order. Output in Markdown.",
                },
            ],
        }
    ]


def got_ocr(image_b64: str) -> list[dict]:
    """GOT-OCR 2.0 prompt format."""
    return [
        {
            "role": "user",
            "content": [
                _image_content(image_b64),
                {
                    "type": "text",
                    "text": "OCR with format",
                },
            ],
        }
    ]


def olmocr(image_b64: str) -> list[dict]:
    """OlmOCR-2 prompt format."""
    return [
        {
            "role": "user",
            "content": [
                _image_content(image_b64),
                {
                    "type": "text",
                    "text": "Extract all text from this document page in Markdown format.",
                },
            ],
        }
    ]


def internvl(image_b64: str) -> list[dict]:
    """InternVL2.5 prompt format."""
    return [
        {
            "role": "user",
            "content": [
                _image_content(image_b64),
                {
                    "type": "text",
                    "text": "Please perform OCR on this document image. Extract all text "
                    "content preserving the layout structure. Output in Markdown format.",
                },
            ],
        }
    ]


def deepseek_ocr(image_b64: str) -> list[dict]:
    """DeepSeek-OCR prompt format with grounding mode."""
    return [
        {
            "role": "user",
            "content": [
                _image_content(image_b64),
                {
                    "type": "text",
                    "text": "<|grounding|>Extract all text from this document image. "
                    "Output in Markdown format with bounding box coordinates.",
                },
            ],
        }
    ]


def chandra(image_b64: str) -> list[dict]:
    """Chandra 2 (Datalab) prompt format."""
    return [
        {
            "role": "user",
            "content": [
                _image_content(image_b64),
                {
                    "type": "text",
                    "text": "Extract all text from this document image. Output in Markdown format.",
                },
            ],
        }
    ]


def extraction_llm(image_b64: str) -> list[dict]:
    """Prompt for LLM backbone candidates (L-01, L-02, L-03).

    Used in the pipeline extraction stage, not as an OCR frontend.
    """
    return [
        {
            "role": "user",
            "content": [
                _image_content(image_b64),
                {
                    "type": "text",
                    "text": "Extract structured information from this document. "
                    "Identify and extract all text, tables, key-value pairs, "
                    "and entities. Output in Markdown format.",
                },
            ],
        }
    ]


# Registry mapping template names to functions
TEMPLATES: dict[str, callable] = {
    "generic_ocr": generic_ocr,
    "nemotron_parse": nemotron_parse,
    "qwen_vl": qwen_vl,
    "glm_ocr": glm_ocr,
    "got_ocr": got_ocr,
    "olmocr": olmocr,
    "internvl": internvl,
    "deepseek_ocr": deepseek_ocr,
    "chandra": chandra,
    "extraction_llm": extraction_llm,
}


def get_template(name: str) -> callable:
    """Get a prompt template by name."""
    if name not in TEMPLATES:
        raise ValueError(
            f"Unknown prompt template: {name}. Available: {list(TEMPLATES.keys())}"
        )
    return TEMPLATES[name]
