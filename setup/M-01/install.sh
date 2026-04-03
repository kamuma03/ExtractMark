#!/bin/bash
# Setup script for M-01 -- Nemotron Parse v1.1
# Run from project root: bash setup/M-01/install.sh

set -euo pipefail

MODEL_ID="M-01"
MODEL_HF="nvidia/nemotron-parse-v1.1"
VENV_DIR=".venvs/${MODEL_ID}"

echo "=== Setting up ${MODEL_ID} (Nemotron Parse v1.1) ==="

# Create isolated venv
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating venv at ${VENV_DIR}..."
    uv venv "${VENV_DIR}"
fi

# Activate and install
source "${VENV_DIR}/bin/activate"

echo "Installing dependencies..."
uv pip install vllm openai albumentations torch

# Pre-download model
echo "Downloading model weights (this may take a while)..."
huggingface-cli download "${MODEL_HF}"

echo ""
echo "=== ${MODEL_ID} setup complete ==="
echo "To serve: source ${VENV_DIR}/bin/activate && vllm serve ${MODEL_HF} --dtype bfloat16"
echo "To test:  extractmark serve ${MODEL_ID}"
