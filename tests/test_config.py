"""Tests for extractmark.config -- YAML configuration loading & validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from extractmark.config import (
    ExtractMarkConfig, GenerationParams, ModelConfig, LibraryConfig,
    DatasetConfig, RunConfig, EvalDefaults, PipelineDefaults,
    load_config, _load_yaml,
)


class TestPydanticModels:
    """Test that all config models have correct defaults and validation."""

    def test_generation_params_defaults(self):
        gp = GenerationParams()
        assert gp.temperature == 0.0
        assert gp.seed == 42
        assert gp.max_tokens == 4096
        assert gp.top_k is None
        assert gp.repetition_penalty is None

    def test_model_config_defaults(self):
        mc = ModelConfig(name="test", hf_model_id="org/model")
        assert mc.dtype == "bfloat16"
        assert mc.supports_bbox is False
        assert mc.prompt_template == "generic_ocr"
        assert mc.vllm_args == []
        assert mc.port == 8000

    def test_model_config_custom(self):
        mc = ModelConfig(
            name="Nemotron", hf_model_id="nvidia/NP",
            dtype="float16", supports_bbox=True,
            prompt_template="nemotron_parse",
            vllm_args=["--dtype", "float16", "--max-model-len", "8192"],
        )
        assert mc.supports_bbox is True
        assert len(mc.vllm_args) == 4

    def test_library_config_defaults(self):
        lc = LibraryConfig(name="PyMuPDF")
        assert lc.tier == 1
        assert lc.formats == []
        assert lc.gpu_required is False

    def test_dataset_config(self):
        dc = DatasetConfig(name="OmniDocBench", loader="omnidocbench", path="data/odb/")
        assert dc.has_ground_truth is True
        assert dc.has_unit_tests is False
        assert dc.priority == "MEDIUM"

    def test_eval_defaults(self):
        ed = EvalDefaults()
        assert ed.sbert_model == "all-MiniLM-L12-v2"
        assert ed.judge_model == "L-01"
        assert ed.judge_temperature == 0.0
        assert ed.judge_seed == 42

    def test_run_config_defaults(self):
        rc = RunConfig()
        assert rc.models == []
        assert rc.libraries == []
        assert rc.evaluators == ["L1", "L2"]
        assert rc.max_pages is None
        assert rc.warmup_pages == 3

    def test_extractmark_config_empty(self):
        cfg = ExtractMarkConfig()
        assert cfg.models == {}
        assert cfg.libraries == {}
        assert cfg.datasets == {}


class TestLoadYaml:
    def test_load_valid_yaml(self, tmp_path):
        p = tmp_path / "test.yaml"
        p.write_text("key: value\nnum: 42\n")
        data = _load_yaml(p)
        assert data == {"key": "value", "num": 42}

    def test_load_empty_yaml(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        assert _load_yaml(p) == {}


class TestLoadConfig:
    def _write_configs(self, tmp_path):
        """Write minimal config files for testing."""
        configs = tmp_path / "configs"
        configs.mkdir()
        runs = configs / "runs"
        runs.mkdir()

        # models.yaml
        (configs / "models.yaml").write_text(yaml.dump({"models": {
            "M-01": {"name": "TestModel", "hf_model_id": "org/model"},
        }}))

        # libraries.yaml
        (configs / "libraries.yaml").write_text(yaml.dump({"libraries": {
            "LIB-01": {"name": "PyMuPDF"},
        }}))

        # datasets.yaml
        (configs / "datasets.yaml").write_text(yaml.dump({"datasets": {
            "D-01": {"name": "ODB", "loader": "omnidocbench", "path": "data/odb/"},
        }}))

        # defaults.yaml
        (configs / "defaults.yaml").write_text(yaml.dump({
            "pipeline": {"warmup_pages": 5},
            "evaluation": {"sbert_model": "all-MiniLM-L6-v2"},
        }))

        # run config
        run_path = runs / "test.yaml"
        run_path.write_text(yaml.dump({"run": {
            "name": "test_run",
            "models": ["M-01"],
            "datasets": ["D-01"],
            "evaluators": ["L1"],
        }}))

        return run_path

    def test_load_full_config(self, tmp_path):
        run_path = self._write_configs(tmp_path)
        cfg = load_config(run_path)

        assert "M-01" in cfg.models
        assert cfg.models["M-01"].hf_model_id == "org/model"
        assert "LIB-01" in cfg.libraries
        assert "D-01" in cfg.datasets
        assert cfg.run.name == "test_run"
        assert cfg.run.models == ["M-01"]
        assert cfg.pipeline.warmup_pages == 5
        assert cfg.evaluation.sbert_model == "all-MiniLM-L6-v2"

    def test_load_missing_optional_files(self, tmp_path):
        """Config should still load if optional YAML files are missing."""
        configs = tmp_path / "configs"
        configs.mkdir()
        runs = configs / "runs"
        runs.mkdir()
        run_path = runs / "minimal.yaml"
        run_path.write_text(yaml.dump({"run": {"name": "minimal"}}))

        cfg = load_config(run_path)
        assert cfg.run.name == "minimal"
        assert cfg.models == {}
        assert cfg.libraries == {}

    def test_load_real_configs(self):
        """Test loading the actual project config files."""
        run_path = Path("configs/runs/quick_smoke.yaml")
        if not run_path.exists():
            pytest.skip("Project config files not present")
        cfg = load_config(run_path)
        assert cfg.run.name
        assert len(cfg.models) > 0 or len(cfg.run.models) > 0
