# ExtractMark

Self-hosted document extraction model evaluation benchmark for NVIDIA DGX Spark.

## Overview

ExtractMark benchmarks self-hostable document extraction models and Python parsing libraries on a personal NVIDIA DGX Spark workstation (ARM64/Linux). All inference runs on-premise via [vLLM](https://github.com/vllm-project/vllm) -- no external APIs.

The benchmark evaluates text accuracy, table fidelity, layout understanding, bounding-box precision, and reading-order correctness using a **multi-layer metric framework**:

- **Edit distance** (CER/WER) via jiwer
- **Semantic similarity** via Sentence-BERT
- **LLM-as-judge** via self-hosted vLLM
- **Binary unit tests** (OlmOCR-Bench)

## Models Under Evaluation

13 OCR/VLM models deployed via vLLM OpenAI-compatible API, including Qwen2.5-VL, InternVL3, Gemma3, SmolVLM2, and more.

## Libraries Under Evaluation

17 Python parsing libraries across three tiers:

| Tier | Libraries |
|------|-----------|
| Rule-based | PyMuPDF, pdfplumber, pypdfium2, Camelot, Tabula, python-docx, python-pptx |
| AI-augmented | Docling, MinerU, Marker, Surya, Unstructured, MarkItDown |
| Specialized | Tesseract, Table Transformer, Nougat, pdfminer |

## Datasets

All public benchmark datasets:

- **OmniDocBench v1.5** -- Multi-format document understanding
- **OlmOCR-Bench** -- OCR accuracy unit tests
- **DocLayNet** -- Document layout analysis
- **FinTabNet** -- Financial table extraction
- **FUNSD** -- Form understanding
- **DocVQA** -- Document visual question answering

## Installation

```bash
# Clone and install
git clone https://github.com/<your-username>/ExtractMark.git
cd ExtractMark
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# Download datasets
python scripts/download_datasets.py

# Run the full benchmark
extractmark run configs/runs/full_benchmark.yaml

# Run a quick smoke test
extractmark run configs/runs/quick_smoke.yaml
```

## Project Structure

```
extractmark/          # Main Python package
  cli.py              # Typer CLI entry point
  pipeline.py         # Benchmarking pipeline
  config.py           # YAML configuration loading
  datasets/           # Dataset loaders (6 datasets)
  evaluators/         # Evaluation metrics (4 evaluators)
  libraries/          # Parsing library adapters (17 libraries)
  models/             # VLM/LLM model interfaces
  serving/            # vLLM server management
  reporting/          # Result summarization
configs/              # YAML configuration files
scripts/              # Utility and runner scripts
eval/                 # Evaluation resources (judge prompts)
setup/                # Installation scripts
```

## Requirements

- Python >= 3.10
- NVIDIA GPU with CUDA support (tested on DGX Spark)
- vLLM for model serving

See [Requirements.md](Requirements.md) for full project requirements and evaluation methodology.

## License

Personal research project. All models evaluated under their respective licenses for personal/research use. All datasets are publicly available.
