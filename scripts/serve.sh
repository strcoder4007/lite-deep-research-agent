#!/usr/bin/env bash
set -euo pipefail

echo "Starting Ollama..."
ollama serve &
OLLAMA_PID=$!

cleanup() {
    echo ""
    echo "Shutting down Ollama..."
    kill $OLLAMA_PID 2>/dev/null || true
    wait $OLLAMA_PID 2>/dev/null || true
    echo "Ollama stopped."
}
trap cleanup EXIT INT TERM

echo "Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:11434/api/tags" > /dev/null 2>&1; then
        echo "Ollama is ready on http://localhost:11434"
        echo "Press Ctrl+C to stop."
        wait
        exit 0
    fi
    sleep 2
done

echo "ERROR: Ollama did not become ready within 1 minute."
exit 1