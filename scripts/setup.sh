#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Installing vLLM ==="
pip install vllm

echo ""
echo "=== Downloading models ==="
echo "This downloads ~8GB for the LLM and ~300MB for embeddings."
echo "Continue? [y/N]"
read -r response
if [[ "$response" != "y" && "$response" != "Y" ]]; then
    echo "Skipping model download. Run manually:"
    echo "  huggingface-cli download cyankiwi/Qwen3.5-4B-AWQ-4bit"
    echo "  huggingface-cli download nomic-ai/nomic-embed-text-v1.5"
    exit 0
fi

huggingface-cli download cyankiwi/Qwen3.5-4B-AWQ-4bit
huggingface-cli download nomic-ai/nomic-embed-text-v1.5

echo ""
echo "=== Setup complete ==="
echo "Start vLLM servers: bash scripts/serve.sh"
echo "Then run the agent:  python -m research_agent"