# High-Level Design — lite-deep-research-agent

> **Status note:** This document originally described a parallel sub-agent (`Send` fan-out) orchestrator. The **implemented** system is a **monolithic, sequential LangGraph pipeline** (`plan → search → fetch → analyze → should_continue → memory → synthesize`). The sub-agent architecture is a *future* target (see §12 Roadmap) and is **not** in the current code. This document has been updated to describe what is actually implemented, plus the recent optimization pass.

## 1. System Overview

**lite-deep-research-agent** is a local deep-research agent built on LangGraph. It plans search queries from a user question, searches the web, fetches and extracts content, iterates when coverage is thin, and synthesizes a grounded, sourced report. Everything runs locally via Ollama (LLM + embeddings).

### Design Goals

| Goal | How (current) |
|---|---|
| **Lightweight** | Small Ollama model (4B-class), local embeddings, zero cloud dependencies |
| **Local-first** | Ollama serves both LLM and embeddings; Chroma for memory; no external APIs |
| **Grounded output** | Facts extracted with `source_url` via structured output; citations flow into the report |
| **Iterative** | `should_continue` loops back to search when coverage is insufficient (safety-capped) |
| **Single process** | Sequential node pipeline; I/O (fetch) parallelized within the fetch node |

---

## 2. Architecture Overview (implemented)

```
┌──────────────────────────────────────────────────────────────┐
│                  MONOLITHIC RESEARCH GRAPH                     │
│                                                                │
│   ┌──────┐   ┌────────┐   ┌────────┐   ┌─────────┐            │
│   │ plan │──▶│ search │──▶│ fetch  │──▶│ analyze │            │
│   └──────┘   └────────┘   └────────┘   └─────────┘            │
│                                            │                  │
│                                     ┌──────▼─────────┐        │
│                                     │ should_continue │        │
│                                     └──┬─────────┬───┘        │
│                                  "search"│       │"synthesize" │
│                                        ▼       ▼             │
│                                (loop back)  ┌────────┐        │
│                                            │ memory │        │
│                                            └───┬────┘        │
│                                                ▼             │
│                                            │synthesize│      │
│                                            └───┬────┘        │
│                                                ▼             │
│                                               END            │
└──────────────────────────────────────────────────────────────┘
```

The graph is a single `StateGraph(ResearchState)` compiled in `graph.py`. All nodes are synchronous and run in order; `fetch_node` parallelizes its own I/O internally with `asyncio`.

---

## 3. State Design

```python
from typing import Any, Dict, List, Optional, TypedDict

class ResearchState(TypedDict, total=False):
    query: str                              # original user question
    research_plan: Dict[str, Any]           # parsed plan (queries/aspects/gaps)
    search_queries: List[str]               # queries this round (accumulate across loops)
    search_results: List[Dict[str, Any]]    # reranked results
    fetched_content: List[Dict[str, Any]]   # fetched pages (url/title/text/metadata)
    extracted_facts: List[str]              # "claim (source: url)" strings
    relevant_memory: List[Dict[str, Any]]   # Chroma retrieval for synthesis
    final_answer: str                       # synthesized report
    sources: List[str]                      # cited source URLs
    iteration: int                          # current round
    max_iterations: int                     # safety cap
    plan_gaps: List[str]                    # gaps driving continuation
    next_step: str                          # "search" | "synthesize"
    errors: List[str]
    messages: List[str]
```

### State semantics

Unlike the original sub-agent design, there are **no** `Annotated[..., operator.add]` reducers. Each node returns the keys it owns and the pipeline carries a single growing list of facts via plain reassignment in `agent.py` (`latest_state = {**latest_state, **data}`). `extracted_facts` therefore persists across iterations because `should_continue` and later nodes don't overwrite it.

---

## 4. Component Design

### 4.1 LLM: Ollama ChatOllama

| Property | Value |
|---|---|
| Client | `langchain_ollama.ChatOllama` |
| Model | `.env`: `qwen3.5:4b`; `config.py` default: `qwen3:8b-q4_K_M` |
| Context window | 262144 (256K) recommended via `num_ctx` |
| Embeddings | `.env`: `nomic-embed-text`; `config.py` default: `mxbai-embed-large` |
| Structured output | `with_structured_output()` used in `analyze_node` |

`create_llm()` in `tools.py` sets `num_thread=4`. For analysis, `analyze_node` binds a lower temperature (`ANALYSIS_TEMPERATURE`, default 0.1) before invoking.

### 4.2 Search: DuckDuckGo (`ddgs`)

`run_ddg_search()` in `tools.py` queries DuckDuckGo via the `ddgs` library (text or news, depending on `SEARCH_TIME_LIMIT`). Results are normalized to `{url, title, snippet, published_at}`.

> The original design specified `s.jina.ai` and `brave` fallbacks. These are **not** implemented in the current code — DuckDuckGo is the only backend.

### 4.3 Fetch: trafilatura (parallel)

`fetch_url()` in `tools.py` uses `trafilatura.fetch_url()` then `trafilatura.extract()` with `output_format="markdown"`, `favor_precision=True`, `include_comments=False`, `include_tables=False`. Output is truncated to `MAX_PAGE_CHARS` (default 10000).

`fetch_node` in `nodes.py` runs all fetches concurrently:

```python
sem = asyncio.Semaphore(config.FETCH_CONCURRENCY)  # default 5
async def _fetch_one(item):
    async with sem:
        resp = await asyncio.to_thread(fetch_url, item["url"], config.REQUEST_TIMEOUT)
    ...
results = asyncio.run(_fetch_all(to_visit))
```

Memory writes happen **after** the gather, in the main thread, to avoid concurrent Chroma writes.

### 4.4 Memory: Chroma

Persisted to `./advanced_memory/`, collection `research_memory`. `memory.py` provides `add_to_memory()` (splits text, writes chunks, persists) and `query_memory()` (similarity search with recency boost `1/(1+age_hours/24)`).

`fetch_node` writes each page; `memory_node` reads once before synthesis.

### 4.5 Analysis: structured fact extraction

`analyze_node` calls the LLM with `with_structured_output(AnalyzeOutput)`:

```python
class FactItem(BaseModel):
    claim: str
    source_url: str

class AnalyzeOutput(BaseModel):
    facts: List[FactItem]
```

Each page is analyzed with up to `ANALYSIS_SNIPPET_CHARS` (default 4000) of its text. Facts are stored as `"{claim} (source: {url})"` strings so citations survive into synthesis. A plain-text fallback runs if structured parsing fails.

---

## 5. Graph Design (`graph.py` / `nodes.py`)

| Node | Input | Action | Output keys |
|---|---|---|---|
| `plan` | `query` | LLM → YAML → parse into queries/aspects/gaps | `research_plan`, `search_queries`, `plan_gaps` |
| `search` | `search_queries` | DDG search + dedup + embedding rerank | `search_results` |
| `fetch` | `search_results` | parallel trafilatura fetch + memory write | `fetched_content` |
| `analyze` | `fetched_content` | per-page structured fact extraction | `extracted_facts` |
| `should_continue` | `fetched_content`, `extracted_facts`, `plan_gaps`, `iteration` | heuristic loop decision | `next_step`, `search_queries`, `iteration` |
| `memory` | `query`, `plan_gaps` | Chroma query for context | `relevant_memory` |
| `synthesize` | `extracted_facts`, `relevant_memory` | LLM report with inline citations | `final_answer`, `sources` |

### Loop control

`should_continue` returns `next_step="search"` when `iteration < max_iterations` AND (`fetched < MIN_FETCHED_FOR_STOP` OR `facts < MIN_FACTS_FOR_STOP` OR `plan_gaps`). Otherwise `next_step="synthesize"`. On continuation, new queries are appended as `f"{query} {gap}"`.

---

## 6. Data Flow (Request Lifecycle)

```
User query
  → plan: 4–5 search queries + aspects + gaps
  → search: DDG → dedup → embedding rerank → top-N
  → fetch: parallel trafilatura fetch (≤ FETCH_LIMIT pages) → write to Chroma
  → analyze: per-page structured facts (claim + source_url)
  → should_continue: enough facts?
        no  → append gap-based queries, iteration++, loop to search
        yes → memory
  → memory: Chroma query with original query + gaps
  → synthesize: report from facts + memory, with inline source markers
  → END
```

---

## 7. Ollama Integration

### Setup

```bash
pip install -r requirements.txt
ollama serve
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
```

### Model Config (256K context)

```dockerfile
# Modelfile.qwen3.5-4b-256k
FROM qwen3.5:4b
PARAMETER num_ctx 262144
PARAMETER temperature 0.25
PARAMETER num_thread 4
```

`ollama create qwen3.5-4b-256k -f Modelfile.qwen3.5-4b-256k`

### Memory Budget (Mac Unified Memory, indicative)

| Component | Estimate |
|---|---|
| LLM weights (4B, Q4_K_M) | ~3.5GB |
| KV cache (256K context, fp16) | ~2.5GB |
| Embeddings | ~0.5GB |
| System + overhead | ~2GB |
| **Total** | **~8.5GB** |

### CLI Flags

```bash
python -m research_agent            # interactive REPL
python -m research_agent --verbose   # node-by-node timing (default)
python -m research_agent --iterations 3
```

---

## 8. Configuration (`config.py`)

All values read from env vars with sensible defaults. Key settings:

```bash
# Models
LLM_MODEL=qwen3.5:4b            # or qwen3:8b-q4_K_M
EMBED_MODEL=nomic-embed-text    # or mxbai-embed-large
LLM_TEMPERATURE=0.25
LLM_MAX_TOKENS=2048
LLM_NUM_CTX=262144

# Search
SEARCH_RESULTS_PER_QUERY=8
SEARCH_RERANK_TOP_N=10
SEARCH_RERANK_USE_HOST_DEDUP=1
SEARCH_SINCE_DAYS=0
SEARCH_RECENCY_BOOST=0.05
FETCH_LIMIT=15

# Fetch
MAX_PAGE_CHARS=10000
REQUEST_TIMEOUT=12
FETCH_CONCURRENCY=5

# Analysis (added in optimization pass)
ANALYSIS_SNIPPET_CHARS=4000
ANALYSIS_TEMPERATURE=0.1

# Iteration / loop
MAX_ITERATIONS=2                 # .env overrides to 5
MIN_FETCHED_FOR_STOP=3
MIN_FACTS_FOR_STOP=5

# Memory
MEMORY_DIR=advanced_memory
MEMORY_TOP_K=5
MEMORY_SIMILARITY_THRESHOLD=0.35
MEMORY_MIN_CHARS=200
CHUNK_SIZE=1000
CHUNK_OVERLAP=100

# Tracing (optional)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=lite-deep-research
```

---

## 9. File Map

```
lite-deep-research-agent/
├── HLD.md                  # This document
├── handoff.md              # Project handoff reference
├── README.md               # User-facing readme
├── requirements.txt
├── .env                    # Configuration (edit directly)
│
├── research_agent/
│   ├── __init__.py
│   ├── __main__.py         # Entry point (loads .env, runs CLI)
│   ├── config.py           # All env-read constants
│   ├── tools.py            # ResearchTools, LLM/embedder/vectorstore factories,
│   │                       #   DDG search, trafilatura fetch_url, helpers
│   ├── state.py            # ResearchState, helpers (append_message/append_error)
│   ├── memory.py           # Chroma add/query with recency boosting
│   ├── nodes.py            # All orchestrator nodes (plan/search/fetch/analyze/...)
│   ├── graph.py            # create_research_graph() — compiles the pipeline
│   ├── agent.py            # AdvancedResearchAgent — synchronous stream + logging
│   └── cli.py              # Interactive REPL, saves report to reports/
│
└── scripts/
    ├── setup.sh            # pip install + ollama pull
    └── serve.sh            # Launch Ollama
```

> Note: there is **no** `llm.py` / `search.py` / `fetch.py` / `sub_agent.py` — those were from the earlier sub-agent design. The current code consolidates everything into `tools.py` + `nodes.py`.

---

## 10. Streaming & Observability

`AdvancedResearchAgent.research()` streams graph events synchronously and logs per-node timing + summary (results/fetched/facts/memory/sources/errors counts). When `LANGCHAIN_TRACING_V2=true`, nodes and LLM calls are traced via LangSmith.

---

## 11. Error Handling

| Failure | Strategy |
|---|---|
| DDG search fails | Returns empty results; `should_continue` may loop or synthesize with what's available |
| trafilatura fetch fails / empty | Page skipped (returns `None`); remaining pages continue |
| Structured analysis parse failure | Falls back to plain-text fact extraction for that page |
| Chroma persistence error | Logged; run continues with in-memory operation |
| Max iterations reached | Forces synthesis with available facts |
| All pages yield 0 facts | `should_continue` loops with gap-based queries; eventually synthesizes with gaps noted |

---

## 12. Roadmap

| Priority | Feature | Status |
|---|---|---|
| Done | Parallel trafilatura fetch | ✅ |
| Done | Structured fact extraction with source URLs | ✅ |
| Done | Larger analysis context + lower analysis temperature | ✅ |
| Done | Fix recency date-filter bug (`site:news`) | ✅ |
| P1 | Orchestrator + parallel sub-agent `Send` fan-out architecture | ⬜ not started |
| P1 | Cross-encoder reranking for search results | ⬜ |
| P1 | LLM-driven coverage assessment in `should_continue` (replace heuristic) | ⬜ |
| P1 | Pydantic structured output for `plan_node` (replace YAML) | ⬜ |
| P2 | Cross-source fact deduplication (cosine similarity) | ⬜ |
| P2 | Human-in-the-loop checkpoints | ⬜ |
| P2 | Gradio web UI | ⬜ |
| P2 | Markdown report export with references | ⬜ |
| P3 | Playwright JS fallback for trafilatura misses | ⬜ |
| P3 | `s.jina.ai` / `brave` search fallback | ⬜ |
| P3 | Source URL accessibility verification (HEAD checks) | ⬜ |
| P3 | Embedding fallback to CPU | ⬜ |
