#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Pulling Ollama models ==="
echo "These download ~2.5GB (LLM) + ~137MB (embeddings)."
echo "Make sure Ollama is running (ollama serve)."
echo "Continue? [y/N]"
read -r response
if [[ "$response" != "y" && "$response" != "Y" ]]; then
    echo "Skipping model pull. Run manually:"
    echo "  ollama pull qwen3.5:4b"
    echo "  ollama pull nomic-embed-text"
    exit 0
fi

ollama pull qwen3.5:4b
ollama pull nomic-embed-text

echo ""
echo "=== Setup complete ==="
echo "Run the agent: python -m research_agent"