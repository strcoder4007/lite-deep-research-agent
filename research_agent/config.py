import os
from pathlib import Path

# Model configuration
# LLM is served by any OpenAI-compatible HTTP server (e.g. mlx_lm.server).
LLM_MODEL = os.getenv("LLM_MODEL", "Jackrong/MLX-Qwen3.5-9B-DeepSeek-V4-Flash-4bit")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
# Native context length (262K) comes from the model's config.json;
# mlx_lm.server has no --max-context flag. This constant is used for
# context-usage tracking in the agent loop.
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "262144"))

# Embeddings: in-process HuggingFace sentence-transformers (no external server needed).
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Execution/limits
# The server splits reasoning traces into a separate `reasoning` field, so
# `content` carries the direct answer and these budgets stay modest.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.25"))
# Dedicated LLM timeout in seconds (separate from HTTP fetch timeout).
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))

# Search/fetch settings
SEARCH_RESULTS_PER_QUERY = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "8"))

# Memory/vector store
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", "advanced_memory"))
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "5"))
MEMORY_SIMILARITY_THRESHOLD = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.35"))
MEMORY_MIN_CHARS = int(os.getenv("MEMORY_MIN_CHARS", "200"))

# Text splitting
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Timeouts and caps
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "12"))
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "5000"))

# Agent loop (custom tool-calling loop in agent.py)
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "12"))
# Nudge appended once when the model returns empty/degenerate output.
AGENT_NUDGE = "Output only the JSON tool call or your final answer."
