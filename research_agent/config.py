import os
from pathlib import Path

# Model configuration
# LLM is served by any OpenAI-compatible HTTP server (e.g. mlx_lm.server).
LLM_MODEL = os.getenv("LLM_MODEL", "prism-ml/Ternary-Bonsai-27B-mlx-2bit")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
# Native context length (262K) comes from the model's config.json; mlx_lm.server
# has no --max-context flag. This constant is documentation only.
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "262144"))

# Embeddings: in-process HuggingFace sentence-transformers (no external server needed).
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "huggingface")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Execution/limits
# Thinking is disabled at the server (--chat-template-args '{"enable_thinking":false}'),
# so responses are direct answers and these budgets stay modest.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_SYNTHESIS_MAX_TOKENS = int(os.getenv("LLM_SYNTHESIS_MAX_TOKENS", "4096"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.25"))
SYNTH_TEMPERATURE = float(os.getenv("SYNTH_TEMPERATURE", "0.25"))
# Dedicated LLM timeout in seconds (separate from HTTP fetch timeout).
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))
CONCURRENCY = 1

# Search/fetch settings
SEARCH_RESULTS_PER_QUERY = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "8"))
FETCH_LIMIT = int(os.getenv("FETCH_LIMIT", "15"))
SEARCH_RERANK_TOP_N = int(os.getenv("SEARCH_RERANK_TOP_N", "10"))
SEARCH_RERANK_USE_HOST_DEDUP = os.getenv("SEARCH_RERANK_USE_HOST_DEDUP", "1") == "1"
SEARCH_SINCE_DAYS = int(os.getenv("SEARCH_SINCE_DAYS", "0"))
SEARCH_DATE_FROM = (os.getenv("SEARCH_DATE_FROM", "") or "").strip() or None
SEARCH_DATE_TO = (os.getenv("SEARCH_DATE_TO", "") or "").strip() or None
SEARCH_RECENCY_BOOST = float(os.getenv("SEARCH_RECENCY_BOOST", "0.05"))
SEARCH_TIME_LIMIT = (os.getenv("SEARCH_TIME_LIMIT", "") or "").strip() or None

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
FETCH_CONCURRENCY = int(os.getenv("FETCH_CONCURRENCY", "5"))

# Analysis
ANALYSIS_SNIPPET_CHARS = int(os.getenv("ANALYSIS_SNIPPET_CHARS", "4000"))
ANALYSIS_TEMPERATURE = float(os.getenv("ANALYSIS_TEMPERATURE", "0.1"))

# Conditional loop defaults
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "2"))
MIN_FETCHED_FOR_STOP = int(os.getenv("MIN_FETCHED_FOR_STOP", "3"))
MIN_FACTS_FOR_STOP = int(os.getenv("MIN_FACTS_FOR_STOP", "5"))
