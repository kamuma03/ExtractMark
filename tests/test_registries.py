"""Tests for all registry modules (evaluators, libraries, datasets, models)."""

import pytest

from extractmark.config import EvalDefaults, LibraryConfig, DatasetConfig, ModelConfig


class TestEvaluatorRegistry:
    def test_get_l1(self):
        from extractmark.evaluators.registry import get_evaluator
        ev = get_evaluator("L1")
        assert ev.metric_id == "L1"

    def test_get_l2(self):
        from extractmark.evaluators.registry import get_evaluator
        ev = get_evaluator("L2")
        assert ev.metric_id == "L2"

    def test_get_l3(self):
        from extractmark.evaluators.registry import get_evaluator
        ev = get_evaluator("L3")
        assert ev.metric_id == "L3"

    def test_get_l4(self):
        from extractmark.evaluators.registry import get_evaluator
        ev = get_evaluator("L4")
        assert ev.metric_id == "L4"

    def test_l4_uses_config(self):
        from extractmark.evaluators.registry import get_evaluator
        cfg = EvalDefaults(judge_model="custom-model", judge_temperature=0.5, judge_seed=123)
        ev = get_evaluator("L4", cfg)
        assert ev._configured_judge_model == "custom-model"
        assert ev.temperature == 0.5
        assert ev.seed == 123

    def test_unknown_evaluator(self):
        from extractmark.evaluators.registry import get_evaluator
        with pytest.raises(ValueError, match="Unknown evaluator"):
            get_evaluator("L99")


class TestLibraryRegistry:
    def test_all_libraries_instantiate(self):
        from extractmark.libraries.registry import get_library, _LIB_MAP
        cfg = LibraryConfig(name="test")
        for lib_id in _LIB_MAP:
            adapter = get_library(lib_id, cfg)
            assert hasattr(adapter, "process_page")
            assert hasattr(adapter, "process_document")

    def test_unknown_library(self):
        from extractmark.libraries.registry import get_library
        cfg = LibraryConfig(name="test")
        with pytest.raises(ValueError, match="Library adapter not implemented"):
            get_library("LIB-99", cfg)

    def test_library_count(self):
        from extractmark.libraries.registry import _LIB_MAP
        assert len(_LIB_MAP) == 17


class TestDatasetRegistry:
    def test_all_loaders_registered(self):
        from extractmark.datasets.registry import _LOADER_MAP
        expected = {"omnidocbench", "fintabnet", "funsd", "docvqa", "olmocr_bench", "doclaynet"}
        assert set(_LOADER_MAP.keys()) == expected

    def test_unknown_loader(self):
        from extractmark.datasets.registry import get_dataset
        cfg = DatasetConfig(name="Fake", loader="nonexistent", path="/tmp")
        with pytest.raises(ValueError, match="Unknown dataset loader"):
            get_dataset("D-99", cfg)


class TestModelRegistry:
    def test_get_model(self):
        from extractmark.models.registry import get_model
        cfg = ModelConfig(name="Test", hf_model_id="org/model")
        adapter = get_model("M-01", cfg)
        assert adapter.model_id == "M-01"
