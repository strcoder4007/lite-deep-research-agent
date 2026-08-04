#!/usr/bin/env bash
set -euo pipefail

MODEL="${LLM_MODEL:-Jackrong/MLX-Qwen3.5-9B-DeepSeek-V4-Flash-4bit}"
PORT="${LLM_PORT:-8080}"

echo "Starting mlx_lm.server for ${MODEL} on port ${PORT}..."
echo "(first run downloads the model, ~5 GB)"

exec mlx_lm.server \
  --model "${MODEL}" \
  --port "${PORT}"
