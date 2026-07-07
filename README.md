# Lite Deep Research Agent

Local deep research agent built on LangGraph. The orchestrator decomposes a user query into sub-topics, spawns parallel sub-agents to research each one, iterates when coverage is insufficient, and synthesizes a grounded, sourced report. Everything runs locally via Ollama.

## Quick Start

```bash
# 1. Start Ollama (if not already running)
ollama serve

# 2. One-time setup (pull models)
bash scripts/setup.sh

# 3. Run the agent
python -m research_agent
```

## Architecture

```
Orchestrator sends sub-tasks to parallel sub-agents, iterates until sufficient coverage

  user query
      │
  ┌───▼────┐    ┌───────────────┐    ┌───────────┐    ┌────────────────┐
  │  plan  │───▶│ research_round│───▶│ aggregate │───▶│should_continue │
  │(decomp)│    │  (N parallel  │    │(dedup +   │    │ (LLM assesses  │
  │        │    │   sub-agents) │    │ merge)    │    │  coverage)     │
  └────────┘    └───────┬───────┘    └───────────┘    └───┬──────┬─────┘
                        │                         "continue"│      │"synthesize"
              ┌─────────┼─────────┐                ┌───────┘      ▼
              ▼         ▼         ▼                │         ┌──────────┐
         sub_agent  sub_agent  sub_agent           │         │  memory  │
         search       search     search            │         └────┬─────┘
           │            │          │               │              ▼
         fetch        fetch      fetch             │         ┌──────────┐
           │            │          │               │         │synthesize│
        analyze      analyze    analyze            │         └────┬─────┘
           │            │          │               │              ▼
       per-topic    per-topic  per-topic           └── replan    END
        memory       memory     memory
```

- **plan** — LLM decomposes query into 3–5 sub-tasks with dynamic page budgets (2–8 pages each)
- **research_round** — `Send` fan-out spawns N parallel sub-agents. Each runs search → trafilatura fetch → LLM fact extraction → per-topic Chroma query. Outputs merge via `operator.add` reducers.
- **aggregate** — Deduplicates facts (cosine similarity 0.85) and sources
- **should_continue** — LLM assesses coverage. If insufficient, generates focused follow-up sub-tasks. Safety cap at `MAX_ITERATIONS` (default 5).
- **memory** — Global Chroma query with original query
- **synthesize** — Builds structured report (Executive Summary → Findings → Analysis → Conclusion → Notes) with inline source markers

## Run

```bash
# Interactive REPL (pick from examples or type a custom query)
python -m research_agent

# With streaming node-by-node progress (default)
python -m research_agent --verbose

# Custom iteration limit
python -m research_agent --iterations 3
```

Reports are saved to `reports/report_<hash>.txt`.

### LangSmith Tracing

Set in `.env`:
```bash
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=<your-key>
LANGSMITH_PROJECT=lite-deep-research
```

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph (`StateGraph`, `Send` fan-out, subgraph composition) |
| LLM | `qwen3.5:4b` via Ollama (256K context) |
| Embeddings | `nomic-embed-text` via Ollama (768d) |
| Vector DB | Chroma (persisted to `./advanced_memory/`) |
| Search | DuckDuckGo with `s.jina.ai` auto-fallback |
| Fetch | `trafilatura` (multi-extractor, clean markdown) |
| Structured output | Pydantic via `with_structured_output()` |
| Tracing | LangSmith (optional) |

## Project Structure

```
lite-deep-research-agent/
├── HLD.md                  # High-Level Design document
├── handoff.md              # Project handoff reference
├── README.md               # This file
├── requirements.txt
├── .env                    # Configuration (copy from .env.example or edit directly)
│
├── research_agent/
│   ├── __init__.py
│   ├── __main__.py         # Entry point
│   ├── config.py           # All env-read constants
│   ├── llm.py              # ChatOllama + OllamaEmbeddings factories, ResearchTools
│   ├── state.py            # ResearchState, SubTask, reducers, helpers
│   ├── search.py           # search_web() — multi-backend with DDG→Jina fallback
│   ├── fetch.py            # trafilatura async wrapper, parallel fetch_pages()
│   ├── memory.py           # Chroma add/query with recency boosting
│   ├── sub_agent.py        # Sub-agent subgraph + sub_search/fetch/analyze/memory nodes
│   ├── nodes.py            # Orchestrator nodes: plan, aggregate, should_continue, memory, synthesize
│   ├── graph.py            # create_research_graph() — composes parent + subgraph
│   ├── agent.py            # AdvancedResearchAgent — async generator with streaming
│   └── cli.py              # Async REPL with streaming output
│
└── scripts/
    ├── setup.sh            # One-time: pip install + ollama pull
    └── serve.sh            # Launch Ollama
```

## Configuration

Edit `.env` directly. Key settings:

```bash
# Ollama models
LLM_MODEL=qwen3.5:4b           # LLM model (256K context)
EMBED_MODEL=nomic-embed-text   # Embedding model (768d)
LLM_NUM_CTX=262144             # Context window size

# Search
SEARCH_BACKEND=duckduckgo      # duckduckgo | jina | brave
SEARCH_FALLBACK=1               # DDG → Jina auto-fallback

# Sub-agent behavior
MAX_SUB_TASKS=5                 # max sub-tasks per plan
MAX_ITERATIONS=5                # max research rounds
MAX_PAGE_CHARS=10000            # chars per fetched page

# Memory
MEMORY_DIR=advanced_memory
CHUNK_SIZE=1000
MEMORY_SIMILARITY_THRESHOLD=0.35
```

## Memory Strategy

Chroma persists to `./advanced_memory/` across runs. Each sub-agent:
- **Writes** fetched content with metadata `{url, title, query, topic, focus, timestamp}`
- **Reads** per-topic memory before analyzing (sub-task query + focus)
- Recency boost: `1 / (1 + age_hours / 24)` applied to similarity scores

The central memory node queries with the original user query for broad context before synthesis.

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Parallelism | LangGraph `Send` fan-out | Graph-visible, async-native, automatic reducer merging |
| Fetch | trafilatura | Clean markdown output, zero Chromium overhead |
| Search | DDG + Jina fallback | Free, resilient to DDG API changes |
| Decomposition | LLM structured output | Typed `SubTask` objects, no YAML parsing |
| Iteration | LLM-driven coverage assessment | Generates targeted follow-up sub-tasks, not naive string concat |
| Context | 256K via `num_ctx` | Ollama supports large contexts on Mac with Metal/MLX |