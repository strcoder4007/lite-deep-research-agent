#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Ensuring mlx-lm is up to date ==="
echo "The model requires mlx-lm >= 0.31 (qwen3_5 architecture)."
pip install --upgrade mlx-lm

echo ""
echo "=== Setup complete ==="
echo "Start the LLM server:  bash scripts/serve.sh"
echo "Run the agent:         python -m research_agent"
