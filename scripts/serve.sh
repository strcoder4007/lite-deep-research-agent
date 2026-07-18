#!/usr/bin/env bash
set -euo pipefail

MODEL="${LLM_MODEL:-prism-ml/Ternary-Bonsai-27B-mlx-2bit}"
PORT="${LLM_PORT:-8080}"

echo "Starting mlx_lm.server for ${MODEL} on port ${PORT}..."
echo "(first run downloads the model, ~8.5 GB)"

# Thinking is disabled on purpose: with thinking enabled the OpenAI /v1 response
# leaves `content` empty and LangChain's ChatOpenAI drops the `reasoning` field.
exec mlx_lm.server \
  --model "${MODEL}" \
  --port "${PORT}" \
  --chat-template-args '{"enable_thinking":false}'
