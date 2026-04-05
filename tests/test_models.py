"""Tests for model adapters, prompt templates, and registry."""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from extractmark.config import ModelConfig
from extractmark.models.prompt_templates import (
    TEMPLATES, get_template,
    generic_ocr, nemotron_parse, qwen_vl, glm_ocr, got_ocr,
    olmocr, internvl, deepseek_ocr, chandra, extraction_llm,
    _image_content,
)
from extractmark.models.vllm_model import (
    _encode_image, _extract_tables_from_markdown, _extract_bboxes,
    VLLMModelAdapter,
)


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

class TestEncodeImage:
    def test_encode_image(self, tmp_path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"fake-png-data")
        b64 = _encode_image(img_path)
        assert base64.b64decode(b64) == b"fake-png-data"


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

class TestPromptTemplates:
    FAKE_B64 = "aW1hZ2VkYXRh"  # base64("imagedata")

    def test_all_templates_registered(self):
        expected = {
            "generic_ocr", "nemotron_parse", "qwen_vl", "glm_ocr",
            "got_ocr", "olmocr", "internvl", "deepseek_ocr", "chandra",
            "extraction_llm",
        }
        assert set(TEMPLATES.keys()) == expected

    def test_get_template_valid(self):
        fn = get_template("generic_ocr")
        assert callable(fn)

    def test_get_template_invalid(self):
        with pytest.raises(ValueError, match="Unknown prompt template"):
            get_template("nonexistent")

    def test_image_content_helper(self):
        result = _image_content("abc123")
        assert result["type"] == "image_url"
        assert "base64,abc123" in result["image_url"]["url"]

    @pytest.mark.parametrize("template_name,template_fn", list(TEMPLATES.items()))
    def test_template_returns_valid_messages(self, template_name, template_fn):
        """Every template should return a list of message dicts with role and content."""
        messages = template_fn(self.FAKE_B64)
        assert isinstance(messages, list)
        assert len(messages) >= 1
        for msg in messages:
            assert "role" in msg
            assert "content" in msg

    @pytest.mark.parametrize("template_name,template_fn", list(TEMPLATES.items()))
    def test_template_includes_image(self, template_name, template_fn):
        """Every template should include the base64 image."""
        messages = template_fn(self.FAKE_B64)
        msg_str = str(messages)
        assert self.FAKE_B64 in msg_str

    def test_nemotron_parse_has_special_tokens(self):
        messages = nemotron_parse(self.FAKE_B64)
        text_parts = str(messages)
        assert "<predict_bbox>" in text_parts
        assert "<predict_classes>" in text_parts
        assert "<output_markdown>" in text_parts

    def test_deepseek_has_grounding_token(self):
        messages = deepseek_ocr(self.FAKE_B64)
        text_parts = str(messages)
        assert "<|grounding|>" in text_parts


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

class TestExtractTables:
    def test_single_table(self):
        text = "Before\n| A | B |\n| 1 | 2 |\nAfter"
        tables = _extract_tables_from_markdown(text)
        assert len(tables) == 1
        assert "| A | B |" in tables[0]

    def test_multiple_tables(self):
        text = "| A |\n| 1 |\nText\n| B |\n| 2 |"
        tables = _extract_tables_from_markdown(text)
        assert len(tables) == 2

    def test_no_tables(self):
        assert _extract_tables_from_markdown("plain text") == []

    def test_table_at_end(self):
        text = "Text\n| X | Y |"
        tables = _extract_tables_from_markdown(text)
        assert len(tables) == 1


# ---------------------------------------------------------------------------
# Bbox extraction
# ---------------------------------------------------------------------------

class TestExtractBboxes:
    def test_nemotron_format(self):
        text = "<x_0.05><y_0.10>Hello<x_0.95><y_0.20><class_Text>"
        bboxes = _extract_bboxes(text)
        assert len(bboxes) == 1
        assert bboxes[0]["x1"] == 0.05
        assert bboxes[0]["y1"] == 0.10
        assert bboxes[0]["x2"] == 0.95
        assert bboxes[0]["y2"] == 0.20
        assert bboxes[0]["text"] == "Hello"
        assert bboxes[0]["class"] == "Text"

    def test_box_tag_format(self):
        text = "<box>10,20,100,200</box>"
        bboxes = _extract_bboxes(text)
        assert len(bboxes) == 1
        assert bboxes[0] == {"x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 200.0}

    def test_bbox_tag_format(self):
        text = "<bbox>5, 10, 50, 100</bbox>"
        bboxes = _extract_bboxes(text)
        assert len(bboxes) == 1
        assert bboxes[0]["x1"] == 5.0

    def test_bracket_format(self):
        text = "coords: [10.5, 20.3, 100.7, 200.1]"
        bboxes = _extract_bboxes(text)
        assert len(bboxes) == 1
        assert abs(bboxes[0]["x1"] - 10.5) < 0.01

    def test_no_bboxes(self):
        assert _extract_bboxes("no bounding boxes here") == []

    def test_multiple_nemotron_bboxes(self):
        text = (
            "<x_0.1><y_0.2>A<x_0.3><y_0.4><class_Title>"
            "<x_0.5><y_0.6>B<x_0.7><y_0.8><class_Text>"
        )
        bboxes = _extract_bboxes(text)
        assert len(bboxes) == 2
        assert bboxes[0]["class"] == "Title"
        assert bboxes[1]["class"] == "Text"


# ---------------------------------------------------------------------------
# VLLMModelAdapter
# ---------------------------------------------------------------------------

class TestVLLMModelAdapter:
    def test_instantiation(self):
        cfg = ModelConfig(name="Test", hf_model_id="org/model")
        adapter = VLLMModelAdapter("M-01", cfg)
        assert adapter.model_id == "M-01"
        assert "localhost:8000" in adapter.base_url

    def test_custom_port(self):
        cfg = ModelConfig(name="Test", hf_model_id="org/model", port=9000)
        adapter = VLLMModelAdapter("M-01", cfg)
        assert "9000" in adapter.base_url

    def test_health_check_failure(self):
        cfg = ModelConfig(name="Test", hf_model_id="org/model")
        adapter = VLLMModelAdapter("M-01", cfg)
        adapter._client = MagicMock()
        adapter._client.models.list.side_effect = Exception("connection refused")
        assert adapter.health_check() is False

    def test_health_check_success(self):
        cfg = ModelConfig(name="Test", hf_model_id="org/model")
        adapter = VLLMModelAdapter("M-01", cfg)
        adapter._client = MagicMock()
        adapter._client.models.list.return_value = MagicMock(data=[])
        assert adapter.health_check() is True

    def test_served_model_name_exact_match(self):
        cfg = ModelConfig(name="Test", hf_model_id="org/model")
        adapter = VLLMModelAdapter("M-01", cfg)
        adapter._client = MagicMock()
        model_obj = MagicMock()
        model_obj.id = "org/model"
        adapter._client.models.list.return_value = MagicMock(data=[model_obj])

        name = adapter._get_served_model_name()
        assert name == "org/model"

    def test_served_model_name_fallback(self):
        cfg = ModelConfig(name="Test", hf_model_id="org/model")
        adapter = VLLMModelAdapter("M-01", cfg)
        adapter._client = MagicMock()
        alt_model = MagicMock()
        alt_model.id = "local/different-name"
        adapter._client.models.list.return_value = MagicMock(data=[alt_model])

        name = adapter._get_served_model_name()
        assert name == "local/different-name"

    def test_served_model_name_cached(self):
        cfg = ModelConfig(name="Test", hf_model_id="org/model")
        adapter = VLLMModelAdapter("M-01", cfg)
        adapter._served_model_name = "cached-name"
        assert adapter._get_served_model_name() == "cached-name"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_get_model(self):
        from extractmark.models.registry import get_model
        cfg = ModelConfig(name="Test", hf_model_id="org/model")
        adapter = get_model("M-01", cfg)
        assert adapter.model_id == "M-01"
