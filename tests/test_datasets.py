"""Tests for dataset loaders with mock filesystem data."""

import json
from pathlib import Path

import pytest

from extractmark.types import PageInput


class TestOmniDocBenchLoader:
    def test_load_with_annotations(self, tmp_path):
        from extractmark.datasets.omnidocbench import OmniDocBenchLoader

        # Set up mock data
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        from PIL import Image
        img = Image.new("RGB", (100, 100), "white")
        img.save(images_dir / "page_001.png")

        annotations = [
            {
                "page_info": {
                    "image_path": "page_001.png",
                    "page_no": 1,
                    "page_attribute": {"language": "en", "layout": "single", "data_source": "test"},
                },
                "layout_dets": [
                    {"text": "First paragraph", "order": 1},
                    {"text": "Second paragraph", "order": 2},
                ],
            }
        ]
        (tmp_path / "OmniDocBench.json").write_text(json.dumps(annotations))

        loader = OmniDocBenchLoader("D-01", tmp_path)
        pages = list(loader.load())

        assert len(pages) == 1
        assert pages[0].document_id == "page_001"
        assert pages[0].page_number == 1
        assert pages[0].ground_truth == "First paragraph\nSecond paragraph"
        assert pages[0].metadata["language"] == "en"

    def test_load_with_sorting_order(self, tmp_path):
        from extractmark.datasets.omnidocbench import OmniDocBenchLoader

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        from PIL import Image
        img = Image.new("RGB", (100, 100), "white")
        img.save(images_dir / "page_001.png")

        # Elements out of order in JSON, should be sorted by 'order'
        annotations = [{
            "page_info": {"image_path": "page_001.png", "page_no": 1},
            "layout_dets": [
                {"text": "C", "order": 3},
                {"text": "A", "order": 1},
                {"text": "B", "order": 2},
            ],
        }]
        (tmp_path / "OmniDocBench.json").write_text(json.dumps(annotations))

        loader = OmniDocBenchLoader("D-01", tmp_path)
        pages = list(loader.load())
        assert pages[0].ground_truth == "A\nB\nC"

    def test_load_missing_images_dir(self, tmp_path):
        from extractmark.datasets.omnidocbench import OmniDocBenchLoader
        (tmp_path / "OmniDocBench.json").write_text("[]")
        loader = OmniDocBenchLoader("D-01", tmp_path)
        assert list(loader.load()) == []

    def test_load_missing_annotation_file(self, tmp_path):
        from extractmark.datasets.omnidocbench import OmniDocBenchLoader
        (tmp_path / "images").mkdir()
        loader = OmniDocBenchLoader("D-01", tmp_path)
        assert list(loader.load()) == []

    def test_ignored_elements_skipped(self, tmp_path):
        from extractmark.datasets.omnidocbench import OmniDocBenchLoader
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        from PIL import Image
        Image.new("RGB", (10, 10)).save(images_dir / "p.png")

        annotations = [{
            "page_info": {"image_path": "p.png"},
            "layout_dets": [
                {"text": "visible", "order": 1},
                {"text": "hidden", "order": 2, "ignore": True},
            ],
        }]
        (tmp_path / "OmniDocBench.json").write_text(json.dumps(annotations))

        loader = OmniDocBenchLoader("D-01", tmp_path)
        pages = list(loader.load())
        assert pages[0].ground_truth == "visible"


class TestFinTabNetLoader:
    def test_load_pages(self, tmp_path):
        from extractmark.datasets.fintabnet import FinTabNetLoader

        images = tmp_path / "images" / "doc1"
        images.mkdir(parents=True)
        from PIL import Image
        Image.new("RGB", (10, 10)).save(images / "page_0.png")
        Image.new("RGB", (10, 10)).save(images / "page_1.png")

        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        (ann_dir / "doc1.json").write_text(json.dumps({
            "tables": [
                {"page": 0, "html": "<table><tr><td>A</td></tr></table>"},
                {"page": 1, "html": "<table><tr><td>B</td></tr></table>"},
            ]
        }))

        loader = FinTabNetLoader("D-02", tmp_path)
        pages = list(loader.load())
        assert len(pages) == 2
        assert "A" in pages[0].ground_truth
        assert "B" in pages[1].ground_truth

    def test_parse_page_number(self):
        from extractmark.datasets.fintabnet import FinTabNetLoader
        assert FinTabNetLoader._parse_page_number("page_5") == 5
        assert FinTabNetLoader._parse_page_number("page_0") == 0
        assert FinTabNetLoader._parse_page_number("12") == 12
        assert FinTabNetLoader._parse_page_number("invalid") == 0

    def test_missing_images_dir(self, tmp_path):
        from extractmark.datasets.fintabnet import FinTabNetLoader
        loader = FinTabNetLoader("D-02", tmp_path)
        assert list(loader.load()) == []


class TestFUNSDLoader:
    def test_load_pages(self, tmp_path):
        from extractmark.datasets.funsd import FUNSDLoader

        images = tmp_path / "images"
        images.mkdir()
        from PIL import Image
        Image.new("RGB", (10, 10)).save(images / "form1.png")

        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        (ann_dir / "form1.json").write_text(json.dumps({
            "form": [
                {"text": "Name:", "words": []},
                {"text": "John Doe", "words": []},
            ]
        }))

        loader = FUNSDLoader("D-03", tmp_path)
        pages = list(loader.load())
        assert len(pages) == 1
        assert pages[0].page_number == 0
        assert "Name:" in pages[0].ground_truth
        assert "John Doe" in pages[0].ground_truth

    def test_missing_images_dir(self, tmp_path):
        from extractmark.datasets.funsd import FUNSDLoader
        loader = FUNSDLoader("D-03", tmp_path)
        assert list(loader.load()) == []


class TestDatasetRegistry:
    def test_get_known_dataset(self):
        from extractmark.datasets.registry import get_dataset
        from extractmark.config import DatasetConfig

        cfg = DatasetConfig(name="ODB", loader="omnidocbench", path="/tmp/odb")
        loader = get_dataset("D-01", cfg)
        assert loader.dataset_id == "D-01"

    def test_get_unknown_dataset(self):
        from extractmark.datasets.registry import get_dataset
        from extractmark.config import DatasetConfig

        cfg = DatasetConfig(name="Fake", loader="nonexistent", path="/tmp")
        with pytest.raises(ValueError, match="Unknown dataset loader"):
            get_dataset("D-99", cfg)

    def test_all_loaders_registered(self):
        from extractmark.datasets.registry import _LOADER_MAP
        expected = {"omnidocbench", "fintabnet", "funsd", "docvqa", "olmocr_bench", "doclaynet"}
        assert set(_LOADER_MAP.keys()) == expected
