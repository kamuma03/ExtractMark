"""Pre-flight readiness tests for every model (M-01..M-13) and library (LIB-01..LIB-17).

These tests catch configuration, import, and dependency issues *before* a
multi-hour benchmark run.  They do NOT require a running vLLM server or GPU.

Categories:
  1. Config completeness -- every model/library referenced in configs has valid fields
  2. Prompt template coverage -- every model maps to a registered prompt template
  3. Library dependency imports -- every library's backing package can be imported
  4. Library adapter contracts -- every adapter has process_page + process_document
  5. GPU memory computation -- formula produces sane values for every model
  6. Runner integration -- TRUST_REMOTE_CODE and MODEL_EXTRA_DEPS are consistent
  7. LLM Judge readiness -- judge model is defined and its config is valid
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from extractmark.config import load_config, ModelConfig, LibraryConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_config():
    """Load the full benchmark config (the real YAML files)."""
    run_path = Path("configs/runs/full_benchmark.yaml")
    if not run_path.exists():
        pytest.skip("Project config files not present")
    return load_config(run_path)


@pytest.fixture(scope="module")
def all_model_ids(full_config):
    """All model IDs referenced in the full benchmark run config."""
    return full_config.run.models


@pytest.fixture(scope="module")
def all_library_ids(full_config):
    """All library IDs referenced in the full benchmark run config."""
    return full_config.run.libraries


# ---------------------------------------------------------------------------
# 1. Config completeness -- models
# ---------------------------------------------------------------------------

ALL_RUN_MODELS = [
    "M-01", "M-02", "M-04", "M-06",
    "M-07", "M-08", "M-09", "M-10", "M-11", "M-12", "M-13",
    # M-03 (GOT-OCR2) excluded: incompatible with current vLLM version
]

ALL_RUN_LIBRARIES = [
    "LIB-01", "LIB-02", "LIB-08", "LIB-09", "LIB-10", "LIB-12", "LIB-13",
]

# All library adapters registered in the codebase (not just the run config)
ALL_REGISTERED_LIBRARIES = [
    "LIB-01", "LIB-02", "LIB-03", "LIB-04", "LIB-05", "LIB-06", "LIB-07",
    "LIB-08", "LIB-09", "LIB-10", "LIB-11", "LIB-12", "LIB-13",
    "LIB-14", "LIB-15", "LIB-16", "LIB-17",
]


class TestModelConfigCompleteness:
    """Every model referenced in the run config must have a valid definition."""

    @pytest.mark.parametrize("model_id", ALL_RUN_MODELS)
    def test_model_exists_in_config(self, full_config, model_id):
        assert model_id in full_config.models, (
            f"{model_id} is in run config but missing from models.yaml"
        )

    @pytest.mark.parametrize("model_id", ALL_RUN_MODELS)
    def test_model_has_hf_model_id(self, full_config, model_id):
        cfg = full_config.models[model_id]
        assert cfg.hf_model_id, f"{model_id} has empty hf_model_id"
        assert "/" in cfg.hf_model_id, (
            f"{model_id} hf_model_id '{cfg.hf_model_id}' should be org/model format"
        )

    @pytest.mark.parametrize("model_id", ALL_RUN_MODELS)
    def test_model_has_model_size(self, full_config, model_id):
        cfg = full_config.models[model_id]
        assert cfg.model_size_gb is not None, f"{model_id} has no model_size_gb"
        assert cfg.model_size_gb > 0, f"{model_id} model_size_gb must be positive"

    @pytest.mark.parametrize("model_id", ALL_RUN_MODELS)
    def test_model_has_valid_dtype(self, full_config, model_id):
        cfg = full_config.models[model_id]
        assert cfg.dtype in ("bfloat16", "float16", "float32", "auto"), (
            f"{model_id} has unexpected dtype '{cfg.dtype}'"
        )

    @pytest.mark.parametrize("model_id", ALL_RUN_MODELS)
    def test_model_vllm_args_include_dtype(self, full_config, model_id):
        cfg = full_config.models[model_id]
        assert "--dtype" in cfg.vllm_args, (
            f"{model_id} vllm_args should include --dtype"
        )

    @pytest.mark.parametrize("model_id", ALL_RUN_MODELS)
    def test_model_generation_params(self, full_config, model_id):
        cfg = full_config.models[model_id]
        gp = cfg.generation_params
        assert gp.max_tokens > 0, f"{model_id} max_tokens must be positive"
        assert 0 <= gp.temperature <= 2.0, f"{model_id} temperature out of range"


# ---------------------------------------------------------------------------
# 2. Prompt template coverage
# ---------------------------------------------------------------------------

class TestPromptTemplateCoverage:
    """Every model must reference a registered prompt template."""

    @pytest.mark.parametrize("model_id", ALL_RUN_MODELS)
    def test_model_prompt_template_registered(self, full_config, model_id):
        from extractmark.models.prompt_templates import TEMPLATES
        cfg = full_config.models[model_id]
        assert cfg.prompt_template in TEMPLATES, (
            f"{model_id} references prompt template '{cfg.prompt_template}' "
            f"which is not registered. Available: {list(TEMPLATES.keys())}"
        )

    @pytest.mark.parametrize("model_id", ALL_RUN_MODELS)
    def test_prompt_template_produces_valid_messages(self, full_config, model_id):
        from extractmark.models.prompt_templates import get_template
        cfg = full_config.models[model_id]
        template_fn = get_template(cfg.prompt_template)
        messages = template_fn("fakebase64image")
        assert isinstance(messages, list)
        assert len(messages) >= 1
        for msg in messages:
            assert "role" in msg
            assert "content" in msg


# ---------------------------------------------------------------------------
# 3. Library dependency imports
# ---------------------------------------------------------------------------

# Map library IDs to the top-level import that must succeed
_LIB_IMPORTS = {
    "LIB-01": [("pymupdf", "fitz")],
    "LIB-02": [("pdfplumber", "pdfplumber")],
    "LIB-03": [("pypdfium2", "pypdfium2")],
    "LIB-04": [("camelot", "camelot")],
    "LIB-05": [("tabula", "tabula")],
    "LIB-06": [("docx", "docx")],
    "LIB-07": [("pptx", "pptx")],
    "LIB-08": [("docling.document_converter", "docling.document_converter")],
    "LIB-09": [("magic_pdf.tools.common", "magic_pdf.tools.common")],
    "LIB-10": [("marker.converters.pdf", "marker.converters.pdf")],
    "LIB-11": [("surya", "surya")],
    "LIB-12": [("unstructured.partition.auto", "unstructured.partition.auto")],
    "LIB-13": [("markitdown", "markitdown")],
}


class TestLibraryDependencyImports:
    """Every library used in the benchmark must have its backing packages importable."""

    @pytest.mark.parametrize("lib_id", ALL_RUN_LIBRARIES)
    def test_library_packages_importable(self, lib_id):
        if lib_id not in _LIB_IMPORTS:
            pytest.skip(f"No import check defined for {lib_id}")
        for pkg_name, import_path in _LIB_IMPORTS[lib_id]:
            try:
                importlib.import_module(import_path)
            except ImportError as e:
                pytest.fail(
                    f"{lib_id} requires '{pkg_name}' but import failed: {e}. "
                    f"Install it with: pip install {pkg_name}"
                )


# ---------------------------------------------------------------------------
# 4. Library adapter contract
# ---------------------------------------------------------------------------

class TestLibraryAdapterContract:
    """Every registered library adapter must implement the expected interface."""

    @pytest.mark.parametrize("lib_id", ALL_REGISTERED_LIBRARIES)
    def test_adapter_has_process_page(self, lib_id):
        from extractmark.libraries.registry import get_library
        cfg = LibraryConfig(name="test")
        adapter = get_library(lib_id, cfg)
        assert hasattr(adapter, "process_page"), (
            f"{lib_id} adapter missing process_page method"
        )
        assert callable(adapter.process_page)

    @pytest.mark.parametrize("lib_id", ALL_REGISTERED_LIBRARIES)
    def test_adapter_has_process_document(self, lib_id):
        from extractmark.libraries.registry import get_library
        cfg = LibraryConfig(name="test")
        adapter = get_library(lib_id, cfg)
        assert hasattr(adapter, "process_document"), (
            f"{lib_id} adapter missing process_document method"
        )
        assert callable(adapter.process_document)

    @pytest.mark.parametrize("lib_id", ALL_REGISTERED_LIBRARIES)
    def test_adapter_has_lib_id(self, lib_id):
        from extractmark.libraries.registry import get_library
        cfg = LibraryConfig(name="test")
        adapter = get_library(lib_id, cfg)
        assert adapter.lib_id == lib_id, (
            f"Adapter lib_id mismatch: expected {lib_id}, got {adapter.lib_id}"
        )


# ---------------------------------------------------------------------------
# 5. GPU memory computation
# ---------------------------------------------------------------------------

class TestGPUMemoryComputation:
    """GPU memory allocation must give every model enough headroom."""

    @pytest.mark.parametrize("model_id", ALL_RUN_MODELS)
    def test_gpu_mem_util_is_default(self, full_config, model_id):
        from run_benchmark import _compute_gpu_memory_utilization, DEFAULT_GPU_MEM_UTIL
        cfg = full_config.models[model_id]
        util = _compute_gpu_memory_utilization(cfg)
        expected = cfg.gpu_memory_utilization if cfg.gpu_memory_utilization is not None else DEFAULT_GPU_MEM_UTIL
        assert util == expected, (
            f"{model_id} gpu_mem_util {util} != expected {expected}"
        )

    @pytest.mark.parametrize("model_id", ALL_RUN_MODELS)
    def test_gpu_mem_reserves_enough_for_model(self, full_config, model_id):
        from run_benchmark import _compute_gpu_memory_utilization, SYSTEM_MEMORY_GB
        cfg = full_config.models[model_id]
        util = _compute_gpu_memory_utilization(cfg)
        reserved_gb = util * SYSTEM_MEMORY_GB
        assert reserved_gb >= cfg.model_size_gb, (
            f"{model_id} reserves {reserved_gb:.1f}GB but model needs {cfg.model_size_gb}GB"
        )

    def test_default_when_no_override(self):
        from run_benchmark import _compute_gpu_memory_utilization, DEFAULT_GPU_MEM_UTIL
        cfg = ModelConfig(name="Test", hf_model_id="org/test", model_size_gb=14)
        util = _compute_gpu_memory_utilization(cfg)
        assert util == DEFAULT_GPU_MEM_UTIL

    def test_override_respected(self):
        from run_benchmark import _compute_gpu_memory_utilization
        cfg = ModelConfig(name="Test", hf_model_id="org/test", model_size_gb=14,
                          gpu_memory_utilization=0.50)
        util = _compute_gpu_memory_utilization(cfg)
        assert util == 0.50

    def test_no_size_uses_default(self):
        from run_benchmark import _compute_gpu_memory_utilization, DEFAULT_GPU_MEM_UTIL
        cfg = ModelConfig(name="Unknown", hf_model_id="org/unknown", model_size_gb=None)
        util = _compute_gpu_memory_utilization(cfg)
        assert util == DEFAULT_GPU_MEM_UTIL


# ---------------------------------------------------------------------------
# 6. Runner integration
# ---------------------------------------------------------------------------

class TestRunnerIntegration:
    """TRUST_REMOTE_CODE_MODELS and MODEL_EXTRA_DEPS must cover all models that need them."""

    def test_trust_remote_code_models_exist_in_config(self, full_config):
        from run_benchmark import TRUST_REMOTE_CODE_MODELS
        for mid in TRUST_REMOTE_CODE_MODELS:
            assert mid in full_config.models, (
                f"TRUST_REMOTE_CODE_MODELS references {mid} which is not in models.yaml"
            )

    def test_extra_deps_models_exist_in_config(self, full_config):
        from run_benchmark import MODEL_EXTRA_DEPS
        for mid in MODEL_EXTRA_DEPS:
            assert mid in full_config.models, (
                f"MODEL_EXTRA_DEPS references {mid} which is not in models.yaml"
            )

    # pip package name -> Python import name (only where they differ)
    _PIP_TO_IMPORT = {
        "open_clip_torch": "open_clip",
    }

    def test_extra_deps_packages_importable(self):
        """Every package listed in MODEL_EXTRA_DEPS should be importable."""
        from run_benchmark import MODEL_EXTRA_DEPS
        for model_id, deps in MODEL_EXTRA_DEPS.items():
            for dep in deps:
                import_name = self._PIP_TO_IMPORT.get(dep, dep)
                try:
                    importlib.import_module(import_name)
                except ImportError as e:
                    pytest.fail(
                        f"{model_id} requires '{dep}' (import as '{import_name}', "
                        f"via MODEL_EXTRA_DEPS) but import failed: {e}"
                    )

    def test_all_run_models_have_adapter(self, full_config):
        """Every model in the run config should be creatable via the registry."""
        from extractmark.models.registry import get_model
        for model_id in full_config.run.models:
            cfg = full_config.models[model_id]
            adapter = get_model(model_id, cfg)
            assert adapter.model_id == model_id

    def test_all_run_libraries_have_adapter(self, full_config):
        """Every library in the run config should be creatable via the registry."""
        from extractmark.libraries.registry import get_library
        for lib_id in full_config.run.libraries:
            cfg = full_config.libraries[lib_id]
            adapter = get_library(lib_id, cfg)
            assert adapter.lib_id == lib_id


# ---------------------------------------------------------------------------
# 7. LLM Judge readiness
# ---------------------------------------------------------------------------

class TestLLMJudgeReadiness:
    """The LLM judge model must be properly configured."""

    def test_judge_model_defined_in_config(self, full_config):
        judge_id = full_config.evaluation.judge_model
        assert judge_id in full_config.models, (
            f"Judge model '{judge_id}' referenced in defaults.yaml "
            f"but not defined in models.yaml"
        )

    def test_judge_model_has_valid_hf_id(self, full_config):
        judge_id = full_config.evaluation.judge_model
        judge_cfg = full_config.models[judge_id]
        assert judge_cfg.hf_model_id, f"Judge model {judge_id} has empty hf_model_id"
        assert "/" in judge_cfg.hf_model_id

    def test_judge_model_has_model_size(self, full_config):
        judge_id = full_config.evaluation.judge_model
        judge_cfg = full_config.models[judge_id]
        assert judge_cfg.model_size_gb is not None, (
            f"Judge model {judge_id} has no model_size_gb"
        )

    def test_judge_gpu_memory_reasonable(self, full_config):
        from run_benchmark import _compute_gpu_memory_utilization, SYSTEM_MEMORY_GB, DEFAULT_GPU_MEM_UTIL
        judge_id = full_config.evaluation.judge_model
        judge_cfg = full_config.models[judge_id]
        util = _compute_gpu_memory_utilization(judge_cfg)
        assert util >= DEFAULT_GPU_MEM_UTIL, (
            f"Judge gpu_mem_util {util} is below default {DEFAULT_GPU_MEM_UTIL}"
        )
        reserved_gb = util * SYSTEM_MEMORY_GB
        assert reserved_gb >= judge_cfg.model_size_gb, (
            f"Judge reserves {reserved_gb:.1f}GB but model needs {judge_cfg.model_size_gb}GB"
        )

    def test_judge_evaluator_instantiates(self):
        from extractmark.evaluators.llm_judge import LLMJudgeEvaluator
        ev = LLMJudgeEvaluator(judge_model="Qwen/Qwen2.5-7B-Instruct")
        assert ev.metric_id == "L4"

    def test_l4_in_full_benchmark_evaluators(self, full_config):
        assert "L4" in full_config.run.evaluators, (
            "L4 (LLM Judge) should be in full_benchmark evaluators"
        )


# ---------------------------------------------------------------------------
# 8. Config cross-references
# ---------------------------------------------------------------------------

class TestConfigCrossReferences:
    """Run config references must all resolve to defined components."""

    def test_all_run_models_defined(self, full_config):
        for mid in full_config.run.models:
            assert mid in full_config.models, (
                f"Run config references model {mid} not in models.yaml"
            )

    def test_all_run_libraries_defined(self, full_config):
        for lid in full_config.run.libraries:
            assert lid in full_config.libraries, (
                f"Run config references library {lid} not in libraries.yaml"
            )

    def test_all_run_datasets_defined(self, full_config):
        for did in full_config.run.datasets:
            assert did in full_config.datasets, (
                f"Run config references dataset {did} not in datasets.yaml"
            )

    def test_all_run_evaluators_valid(self, full_config):
        from extractmark.evaluators.registry import get_evaluator
        for eid in full_config.run.evaluators:
            ev = get_evaluator(eid, full_config.evaluation)
            assert ev.metric_id == eid


# ---------------------------------------------------------------------------
# 9. Library config completeness
# ---------------------------------------------------------------------------

class TestLibraryConfigCompleteness:
    """Every library in the run config must have a valid definition."""

    @pytest.mark.parametrize("lib_id", ALL_RUN_LIBRARIES)
    def test_library_exists_in_config(self, full_config, lib_id):
        assert lib_id in full_config.libraries, (
            f"{lib_id} is in run config but missing from libraries.yaml"
        )

    @pytest.mark.parametrize("lib_id", ALL_RUN_LIBRARIES)
    def test_library_has_name(self, full_config, lib_id):
        cfg = full_config.libraries[lib_id]
        assert cfg.name, f"{lib_id} has empty name"

    @pytest.mark.parametrize("lib_id", ALL_RUN_LIBRARIES)
    def test_library_has_pip_packages(self, full_config, lib_id):
        cfg = full_config.libraries[lib_id]
        assert len(cfg.pip_packages) > 0, (
            f"{lib_id} has no pip_packages listed -- "
            f"this makes it hard to diagnose install issues"
        )


# ---------------------------------------------------------------------------
# 10. Model adapter instantiation
# ---------------------------------------------------------------------------

class TestModelAdapterInstantiation:
    """Every model adapter must instantiate cleanly (without a server)."""

    @pytest.mark.parametrize("model_id", ALL_RUN_MODELS)
    def test_model_adapter_creates(self, full_config, model_id):
        from extractmark.models.registry import get_model
        cfg = full_config.models[model_id]
        adapter = get_model(model_id, cfg)
        assert adapter.model_id == model_id
        assert hasattr(adapter, "process_page")
        assert hasattr(adapter, "health_check")
