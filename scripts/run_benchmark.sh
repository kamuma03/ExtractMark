#!/bin/bash
# ExtractMark Full Benchmark Runner
# Usage: bash scripts/run_benchmark.sh [--config configs/runs/quick_smoke.yaml] [extra args...]
#
# This script delegates to scripts/run_benchmark.py which:
# 1. Starts vLLM for each model sequentially (managed mode)
# 2. Runs the benchmark pipeline with live progress
# 3. Stops the server between models to free GPU memory
# 4. Runs library adapters (no vLLM needed)
# 5. Generates a final report
#
# All extra arguments are forwarded (e.g. --docker, --no-serve, -m M-01).
# Requires: docker (for vLLM serving) or native vllm in .venv

set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv"

echo "============================================"
echo "  ExtractMark Benchmark Runner"
echo "============================================"
echo "Time:   $(date)"
echo ""

source "${VENV}/bin/activate"
python scripts/run_benchmark.py "$@"
