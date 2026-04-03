#!/bin/bash
# ExtractMark Full Benchmark Runner
# Usage: bash scripts/run_benchmark.sh [--config configs/runs/quick_smoke.yaml]
#
# This script:
# 1. Starts vLLM for each model sequentially
# 2. Runs the benchmark pipeline with live progress
# 3. Stops the server between models
# 4. Generates a final report
#
# Requires: docker (for vLLM serving) or native vllm in .venv

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/runs/full_benchmark.yaml}"
VENV=".venv"
PORT=8000

echo "============================================"
echo "  ExtractMark Benchmark Runner"
echo "============================================"
echo "Config: ${CONFIG}"
echo "Time:   $(date)"
echo ""

source "${VENV}/bin/activate"
python -m extractmark.cli run --config "${CONFIG}"
