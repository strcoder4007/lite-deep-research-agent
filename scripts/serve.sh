#!/usr/bin/env bash
set -euo pipefail

LLM_PORT="${LLM_PORT:-8000}"
EMBED_PORT="${EMBED_PORT:-8001}"
LLM_MODEL="${LLM_MODEL:-cyankiwi/Qwen3.5-4B-AWQ-4bit}"
EMBED_MODEL="${EMBED_MODEL:-nomic-ai/nomic-embed-text-v1.5}"

cleanup() {
    echo ""
    echo "Shutting down vLLM servers..."
    kill $LLM_PID 2>/dev/null || true
    kill $EMBED_PID 2>/dev/null || true
    wait $LLM_PID 2>/dev/null || true
    wait $EMBED_PID 2>/dev/null || true
    echo "Servers stopped."
}
trap cleanup EXIT INT TERM

echo "Starting LLM server on port $LLM_PORT..."
vllm serve "$LLM_MODEL" \
    --port "$LLM_PORT" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.75 \
    --enable-auto-tool-choice &
LLM_PID=$!

echo "Starting embedding server on port $EMBED_PORT..."
vllm serve "$EMBED_MODEL" \
    --port "$EMBED_PORT" \
    --task embedding \
    --gpu-memory-utilization 0.10 &
EMBED_PID=$!

echo ""
echo "Waiting for servers to be ready..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:$LLM_PORT/v1/models" > /dev/null 2>&1 && \
       curl -s "http://localhost:$EMBED_PORT/v1/models" > /dev/null 2>&1; then
        echo "Both servers ready."
        echo ""
        echo "LLM:       http://localhost:$LLM_PORT/v1"
        echo "Embedding: http://localhost:$EMBED_PORT/v1"
        echo ""
        echo "Press Ctrl+C to stop servers."
        wait
        exit 0
    fi
    sleep 2
done

echo "ERROR: Servers did not become ready within 2 minutes."
exit 1