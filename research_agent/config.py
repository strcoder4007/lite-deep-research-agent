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
EMBED_CACHE_MAX = int(os.getenv("EMBED_CACHE_MAX", "5000"))

# Execution/limits
# The server splits reasoning traces into a separate `reasoning` field, so
# `content` carries the direct answer and these budgets stay modest.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.25"))
# Dedicated LLM timeout in seconds (separate from HTTP fetch timeout).
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))

# Search/fetch settings
SEARCH_RESULTS_PER_QUERY = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "5"))
FETCH_CONCURRENCY = int(os.getenv("FETCH_CONCURRENCY", "4"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
# TTL (seconds) for the in-memory web_search cache; follow-up turns often
# repeat the same query. 0 disables.
SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "1800"))
# Max chars of extracted page text kept per fetched URL.  Tool results are
# truncated further (MESSAGE_PAGE_CHARS) when appended to the message
# history; this bounds what tools like `remember` can receive.
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "5000"))

# URL fetch cache (TTL in seconds; 0 disables).  Caches both successes
# and failures (negative cache) so repeated dead URLs don't waste time.
FETCH_CACHE_TTL = int(os.getenv("FETCH_CACHE_TTL", "3600"))
FETCH_CACHE_MAX = int(os.getenv("FETCH_CACHE_MAX", "500"))

# Adaptive auto-fetch waves: after the first wave, check whether the
# fetched content looks thin (total chars / successful sources).  If
# thin, fetch the next wave up to AUTO_FETCH_MAX_TOTAL.
AUTO_FETCH_WAVE_THRESHOLD = float(os.getenv("AUTO_FETCH_WAVE_THRESHOLD", "2000"))

# Streaming final answer: when 1, stream the final answer live
# instead of buffering it.  Suppressed for structured output
# (JSON / code fence).  Set to 0 to disable.
STREAM_FINAL = os.getenv("STREAM_FINAL", "1") == "1"

# Long-term memory (Chroma vector store persisted under MEMORY_DIR).
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", "advanced_memory"))
# Text chunks are split with this size/overlap before embedding.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
# Texts shorter than this are not written to memory.
MEMORY_MIN_CHARS = int(os.getenv("MEMORY_MIN_CHARS", "200"))
# Recall returns at most this many chunks, filtered by cosine
# similarity threshold (also used for near-duplicate detection on write).
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "5"))
MEMORY_SIMILARITY_THRESHOLD = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.35"))

# Per-run token budget guard: stop auto-fetching/compress once
# accumulated prompt tokens exceed this fraction of LLM_NUM_CTX.
TOKEN_BUDGET_GUARD = float(os.getenv("TOKEN_BUDGET_GUARD", "0.8"))
# Agent loop (custom tool-calling loop in agent.py)
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "6"))
# After this many rounds whose search/fetch tools actually returned content,
# stop letting the model open another research round and force it to write the
# final answer. Weak models otherwise keep chaining search/fetch rounds until
# MAX_AGENT_STEPS and the run dies as "stopped after N steps without a final
# answer". 0 disables (runs the full step budget).
RESEARCH_ROUND_LIMIT = int(os.getenv("RESEARCH_ROUND_LIMIT", "2"))
# After web_search results arrive, fetch this many top URLs automatically
# (in parallel, no extra model call). 0 disables auto-fetch.
AUTO_FETCH_TOP_N = int(os.getenv("AUTO_FETCH_TOP_N", "3"))
AUTO_FETCH_MAX_TOTAL = int(os.getenv("AUTO_FETCH_MAX_TOTAL", "6"))
# Tool results are truncated to this many chars of page text when appended to
# the message history (full text still goes to tools like `remember`).
# Keeps per-step prompt size — the dominant token cost — bounded.
MESSAGE_PAGE_CHARS = int(os.getenv("MESSAGE_PAGE_CHARS", "5000"))
# Domains to skip during auto-fetch (trafilatura can't extract content
# from these or they block automated access).
AUTO_FETCH_SKIP_DOMAINS: list[str] = [
    "google.com",
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "x.com",
    "twitter.com",
    "instagram.com",
    "tiktok.com",
    "linkedin.com",
    "reddit.com",
    "stackoverflow.com",
    "stackexchange.com",
]
# Reputable sources the agent prioritizes during auto-fetch.
# Domains here are fetched before non-listed domains when
# multiple search results are available.
PREFERRED_DOMAINS: list[str] = [
    "arxiv.org",
    "openreview.net",
    "aclanthology.org",
    "paperswithcode.com",
    "distill.pub",
    "nature.com",
    "science.org",
    "cell.com",
    "pnas.org",
    "journals.plos.org",
    "frontiersin.org",
    "mdpi.com",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "cdc.gov",
    "nih.gov",
    "who.int",
    "fda.gov",
    "ema.europa.eu",
    "unicef.org",
    "ourworldindata.org",
    "worldbank.org",
    "imf.org",
    "oecd.org",
    "data.worldbank.org",
    "data.gov",
    "data.gov.uk",
    "census.gov",
    "fred.stlouisfed.org",
    "tradingeconomics.com",
    "nasa.gov",
    "science.nasa.gov",
    "esa.int",
    "noaa.gov",
    "usgs.gov",
    "earthobservatory.nasa.gov",
    "ipcc.ch",
    "climate.gov",
    "un.org",
    "unesco.org",
    "fao.org",
    "iea.org",
    "energy.gov",
    "eia.gov",
    "bls.gov",
    "bea.gov",
    "sec.gov",
    "federalreserve.gov",
    "cbo.gov",
    "gao.gov",
    "docs.python.org",
    "developer.mozilla.org",
    "go.dev",
    "doc.rust-lang.org",
    "kubernetes.io",
    "docker.com",
    "pytorch.org",
    "tensorflow.org",
    "huggingface.co",
    "onnx.ai",
    "vllm.ai",
    "llamaindex.ai",
    "langchain.com",
    "numpy.org",
    "scipy.org",
    "pandas.pydata.org",
    "scikit-learn.org",
    "opencv.org",
    "docs.nvidia.com",
    "developer.nvidia.com",
    "openai.com",
    "ai.googleblog.com",
    "deepmind.google",
    "engineering.fb.com",
    "netflixtechblog.com",
    "aws.amazon.com",
    "cloud.google.com",
    "azure.microsoft.com",
    "blog.cloudflare.com",
    "blog.jetbrains.com",
    "martinfowler.com",
    "queue.acm.org",
    "cacm.acm.org",
    "research.google",
    "research.ibm.com",
    "research.microsoft.com",
    "eff.org",
    "owasp.org",
    "cisa.gov",
    "krebsonsecurity.com",
    "schneier.com",
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "economist.com",
    "wikipedia.org",
    "wikibooks.org",
    "wikiversity.org",
    "archive.org",
    "news.ycombinator.com",
    "lobste.rs",
    "dev.to",
    "hashnode.com",
    "medium.com",
    "substack.com",
    "discuss.python.org",
    "forum.djangoproject.com",
    "discourse.julialang.org",
    "community.openai.com",
    "community.cloudflare.com",
    "forums.developer.nvidia.com",
    "community.intel.com",
    "forum.arduino.cc",
    "forum.raspberrypi.com",
    "bbs.archlinux.org",
    "forum.manjaro.org",
    "forum.level1techs.com",
]
# Nudge appended once when the model returns empty/degenerate output.
AGENT_NUDGE = "Output only the JSON tool call or your final answer."

# Directive appended when RESEARCH_ROUND_LIMIT is reached: end the loop by
# writing the answer instead of opening yet another research round.
AGENT_FINALIZE = (
    "Research is complete. You now have enough material to answer. "
    "Write the final answer as plain text. Do NOT call any tools."
)

# Digest old tool results: after this many steps, compress older
# tool-result messages into one-line digests to save prompt tokens.
# Lower values reduce TTFT for subsequent steps by keeping the
# prompt smaller. 1 = digest after the first step.
MESSAGE_DIGEST_AFTER = int(os.getenv("MESSAGE_DIGEST_AFTER", "1"))
# Max chars kept per digested tool result line.
MESSAGE_DIGEST_MAX_CHARS = int(os.getenv("MESSAGE_DIGEST_MAX_CHARS", "300"))

# Adaptive fetch: minimum successful fetches before considering the
# coverage sufficient (avoids fetching more when we already have enough).
AUTO_FETCH_MIN_SUCCESSFUL = int(os.getenv("AUTO_FETCH_MIN_SUCCESSFUL", "3"))
