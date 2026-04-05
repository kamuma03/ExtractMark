"""Configuration loading and validation for ExtractMark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class GenerationParams(BaseModel):
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 4096
    top_k: int | None = None
    repetition_penalty: float | None = None


class ModelConfig(BaseModel):
    name: str
    hf_model_id: str
    dtype: str = "bfloat16"
    supports_bbox: bool = False
    prompt_template: str = "generic_ocr"
    output_format: str = "markdown"
    vllm_args: list[str] = []
    generation_params: GenerationParams = GenerationParams()
    port: int = 8000
    model_size_gb: float | None = None  # Estimated weight size for memory management


class LibraryConfig(BaseModel):
    name: str
    tier: int = 1
    formats: list[str] = []
    gpu_required: bool = False
    pip_packages: list[str] = []


class DatasetConfig(BaseModel):
    name: str
    loader: str
    path: str
    has_ground_truth: bool = True
    has_unit_tests: bool = False
    priority: str = "MEDIUM"


class PipelineDefaults(BaseModel):
    warmup_pages: int = 3
    save_raw_output: bool = True
    normalize_before_eval: bool = True


class EvalDefaults(BaseModel):
    sbert_model: str = "all-MiniLM-L12-v2"
    judge_model: str = "L-01"
    judge_temperature: float = 0.0
    judge_seed: int = 42


class RunConfig(BaseModel):
    name: str = "default_run"
    models: list[str] = []
    libraries: list[str] = []
    datasets: list[str] = []
    evaluators: list[str] = ["L1", "L2"]
    max_pages: int | None = None
    warmup_pages: int = 3


class ExtractMarkConfig(BaseModel):
    """Top-level configuration combining all sources."""

    models: dict[str, ModelConfig] = {}
    libraries: dict[str, LibraryConfig] = {}
    datasets: dict[str, DatasetConfig] = {}
    pipeline: PipelineDefaults = PipelineDefaults()
    evaluation: EvalDefaults = EvalDefaults()
    run: RunConfig = RunConfig()


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(
    run_config_path: Path,
    configs_dir: Path | None = None,
) -> ExtractMarkConfig:
    """Load and merge configuration from YAML files.

    Args:
        run_config_path: Path to the run config YAML (e.g. configs/runs/quick_smoke.yaml).
        configs_dir: Path to the configs directory. Defaults to run_config_path's grandparent.
    """
    if configs_dir is None:
        configs_dir = run_config_path.parent.parent
        if configs_dir.name != "configs":
            configs_dir = run_config_path.parent

    run_data = _load_yaml(run_config_path)

    # Load component definitions
    models_data = {}
    models_path = configs_dir / "models.yaml"
    if models_path.exists():
        raw = _load_yaml(models_path)
        models_data = raw.get("models", raw)

    libraries_data = {}
    libraries_path = configs_dir / "libraries.yaml"
    if libraries_path.exists():
        raw = _load_yaml(libraries_path)
        libraries_data = raw.get("libraries", raw)

    datasets_data = {}
    datasets_path = configs_dir / "datasets.yaml"
    if datasets_path.exists():
        raw = _load_yaml(datasets_path)
        datasets_data = raw.get("datasets", raw)

    # Load defaults
    defaults_data: dict[str, Any] = {}
    defaults_path = configs_dir / "defaults.yaml"
    if defaults_path.exists():
        defaults_data = _load_yaml(defaults_path)

    # Build merged config
    config_dict: dict[str, Any] = {
        "models": {k: ModelConfig(**v) for k, v in models_data.items()},
        "libraries": {k: LibraryConfig(**v) for k, v in libraries_data.items()},
        "datasets": {k: DatasetConfig(**v) for k, v in datasets_data.items()},
        "pipeline": defaults_data.get("pipeline", {}),
        "evaluation": defaults_data.get("evaluation", {}),
        "run": run_data.get("run", run_data),
    }

    return ExtractMarkConfig(**config_dict)
