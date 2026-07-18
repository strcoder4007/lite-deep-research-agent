# Lite Deep Research Agent

Local deep research agent built on LangGraph. It plans search queries from a
user question, searches the web, fetches and extracts page content, iterates
when coverage is thin, and synthesizes a grounded, sourced report.

The LLM runs locally via an **OpenAI-compatible server** (`mlx_lm.server`,
Apple Silicon). Embeddings run **in-process** via HuggingFace
sentence-transformers. No cloud APIs, no Ollama.

## Prerequisites

- **Apple Silicon Mac** (M1/M2/M3/M4/M5) with ~16 GB unified memory
- **Python 3.10+**
- **`mlx-lm >= 0.31`** (older releases don't support the `qwen3_5` architecture)

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install / upgrade mlx-lm (the model needs a recent version)
pip install --upgrade mlx-lm

# 3. Start the LLM server (downloads the model on first run, ~8.5 GB)
#    IMPORTANT: keep thinking disabled — see note below.
mlx_lm.server \
  --model "prism-ml/Ternary-Bonsai-27B-mlx-2bit" \
  --port 8080 \
  --chat-template-args '{"enable_thinking":false}'

# 4. In another terminal, run the agent
python -m research_agent
```

> **Why `enable_thinking:false`?** This is a reasoning model. When thinking is
> enabled, the OpenAI `/v1` response returns the reasoning trace in a
> non-standard `reasoning` field and leaves `content` empty — and LangChain's
> `ChatOpenAI` client drops that field, so fact extraction gets nothing.
> Disabling thinking makes the model answer directly (fast, `content` always
> populated). The flag is only honored at server launch, not per-request.

## Run

```bash
# Interactive REPL (type a query)
python -m research_agent

# Node-by-node timing (default)
python -m research_agent --verbose

# Custom iteration limit
python -m research_agent --iterations 3
```

Reports are saved to `reports/report_<hash>.txt`.

## Architecture

A single sequential LangGraph `StateGraph` pipeline (monolithic; `fetch`
parallelizes its own I/O internally):

```
plan → search → fetch → analyze → should_continue ──"synthesize"──▶ memory → synthesize → END
                          ▲                          │
                          └────────"search"──────────┘  (loop back, capped at MAX_ITERATIONS)
```

- **plan** — LLM decomposes the query into 4–5 search queries + key aspects + gaps.
- **search** — DuckDuckGo (`ddgs`) per query, dedup by URL/host, embedding rerank with a recency bonus.
- **fetch** — parallel `trafilatura` fetch (semaphore-limited), writes each page to Chroma memory.
- **analyze** — per-page LLM fact extraction; each fact carries a `source_url`. JSON is parsed defensively (tolerates code fences / schema drift).
- **should_continue** — heuristic loop: continue if too few pages/facts or gaps remain; otherwise proceed. Hard cap at `MAX_ITERATIONS`.
- **memory** — Chroma query with the original query (+ gaps/aspects) for broad context.
- **synthesize** — structured report (Executive Summary → Findings → Analysis → Conclusion → Notes) with inline source markers.

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph (`StateGraph`, sequential pipeline) |
| LLM | `prism-ml/Ternary-Bonsai-27B-mlx-2bit` via `mlx_lm.server` (OpenAI-compatible) |
| LLM client | `langchain_openai.ChatOpenAI` pointed at the local server |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` in-process via `langchain_huggingface` (384d) |
| Vector DB | Chroma (persisted to `./advanced_memory/`) |
| Search | DuckDuckGo (`ddgs`) |
| Fetch | `trafilatura` (clean markdown) |
| Tracing | LangSmith (optional) |

## Configuration

Edit `.env` directly. Key settings:

```bash
# LLM (OpenAI-compatible server)
LLM_MODEL=prism-ml/Ternary-Bonsai-27B-mlx-2bit
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=not-needed
LLM_NUM_CTX=262144            # native context; documentation only (no server flag)
LLM_TIMEOUT=180               # seconds; LLM request timeout

# Embeddings (in-process)
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Search / fetch
SEARCH_RESULTS_PER_QUERY=8
FETCH_LIMIT=15
MAX_PAGE_CHARS=10000
FETCH_CONCURRENCY=5

# Iteration
MAX_ITERATIONS=5
MIN_FACTS_FOR_STOP=5

# Memory
MEMORY_DIR=advanced_memory
CHUNK_SIZE=1000
MEMORY_SIMILARITY_THRESHOLD=0.35
```

### LangSmith Tracing (optional)

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-key>
LANGCHAIN_PROJECT=lite-deep-research
```

## Memory Budget (Apple Silicon, 4-bit KV cache)

| Component | 64K ctx | 256K ctx |
|---|---|---|
| LLM weights (27B ternary 2-bit, MLX) | 7.57 GB | 7.57 GB |
| KV cache (4-bit, 16 full-attn layers) | 1.07 GB | 4.30 GB |
| Linear-attention state (48 layers, fixed) | 0.08 GB | 0.08 GB |
| Embeddings (in-process, MiniLM) | 0.24 GB | 0.24 GB |
| Runtime overhead | 1.30 GB | 1.30 GB |
| **Total** | **~10.3 GB** | **~13.5 GB** |

> The model's native context is 262K (256K). Even the full window fits ~13.5 GB
> on a 16 GB Mac with the 4-bit KV cache, so there's no need to cap context.

> **Performance:** this 2-bit 27B runs at roughly ~18–26 tok/s on M-series, so
> multi-page research runs can take a while — that's the model, not the pipeline.

## Project Structure

```
lite-deep-research-agent/
├── HLD.md                  # High-Level Design document
├── handoff.md              # Project handoff reference
├── README.md               # This file
├── requirements.txt
├── .env                    # Configuration (edit directly)
│
├── research_agent/
│   ├── __init__.py
│   ├── __main__.py         # Entry point (loads .env, runs CLI)
│   ├── config.py           # All env-read constants
│   ├── tools.py            # LLM/embedder/vectorstore factories, DDG search,
│   │                       #   trafilatura fetch, JSON-extraction helper
│   ├── state.py            # ResearchState, helpers
│   ├── memory.py           # Chroma add/query with recency boosting
│   ├── nodes.py            # Orchestrator nodes (plan/search/fetch/analyze/...)
│   ├── graph.py            # create_research_graph() — compiles the pipeline
│   ├── agent.py            # AdvancedResearchAgent — synchronous stream + logging
│   └── cli.py              # Interactive REPL, saves report to reports/
│
└── scripts/
    ├── setup.sh            # pip install helper
    └── serve.sh            # Launch the LLM server
```
