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

---

## 13. Open Items (detailed)

This section expands on the not-yet-implemented improvements ("remaining improvements") identified while optimizing the pipeline. They are ordered roughly by impact on research quality. None require adding new graph nodes — most are within existing nodes, plus a few architectural items.

### 13.1 Pydantic structured output for `plan_node` (replace YAML)
- **Where:** `plan_node` in `nodes.py` (currently parses YAML via `yaml.safe_load` with `_parse_plan`).
- **Why:** YAML parsing is fragile — small model deviations (code fences, indentation) break the plan and force a fallback to the raw query. The handoff explicitly calls for no YAML parsing.
- **What:** Use `tools.llm.with_structured_output(PlanOutput)` returning `search_queries`, `key_aspects`, `gaps_to_address`. Adds robustness and removes the `NO_THINK_FLAG`/`_with_no_think` hack.

### 13.2 LLM-driven coverage assessment in `should_continue` (replace heuristic)
- **Where:** `should_continue_node` in `nodes.py` (currently counts `fetched < 3`, `facts < 5`, or `plan_gaps`).
- **Why:** Pure thresholds don't measure *coverage* — redundant facts can satisfy the count while real gaps remain, or vice versa. Next-round queries are naive concatenations (`f"{query} {gap}"`).
- **What:** Add an LLM call (structured output) that reads the extracted facts + gaps and decides `coverage_sufficient` plus *refined* follow-up queries targeting the actual gap (not string concat). Keep the `MAX_ITERATIONS` hard cap.

### 13.3 Cross-source fact deduplication
- **Where:** after `analyze_node` (or in `should_continue`), before synthesis.
- **Why:** Facts accumulate across pages with no merging; near-duplicate facts from different sources inflate the report and waste context.
- **What:** Embed each fact and drop near-duplicates above a cosine threshold (e.g. 0.85), mirroring the original `aggregate_node` design. The structured `source_url` per fact makes keeping a canonical source trivial.

### 13.4 Memory self-retrieval fix
- **Where:** graph edges / ordering in `graph.py` + `memory_node` in `nodes.py`.
- **Why:** `fetch_node` writes pages to Chroma *before* `memory_node` reads, so the memory node largely re-retrieves the just-fetched content rather than genuinely prior knowledge.
- **What:** Query memory *before* fetch (so it reflects prior runs), or tag writes with the current run id and exclude them from the memory read. Makes memory a real cross-run knowledge base.

### 13.5 Better analysis context & cross-source reconciliation
- **Where:** `analyze_node` in `nodes.py`.
- **Why:** Each page is still analyzed in isolation with only `ANALYSIS_SNIPPET_CHARS` (4000) of context. Contradictions/redundancy across sources aren't reconciled, and large pages are truncated.
- **What:** Pass already-extracted facts from prior pages as context so later pages can corroborate/conflict; consider feeding more of each page now that 256K context is available. Optionally raise `ANALYSIS_SNIPPET_CHARS`.

### 13.6 `s.jina.ai` / `brave` search fallback
- **Where:** `run_ddg_search` in `tools.py` (+ `SEARCH_BACKEND`/`SEARCH_FALLBACK` config).
- **Why:** Current code uses DuckDuckGo only. When DDG returns < 3 results or errors, the run degrades instead of recovering.
- **What:** Implement the auto-fallback chain (DDG → s.jina.ai → partial) described in the original design, with `SEARCH_BACKEND` and `SEARCH_FALLBACK` wired through `config.py`.

### 13.7 Cross-encoder reranking for search results
- **Where:** `search_node` rerank step (`_rerank_results`) in `nodes.py`.
- **Why:** Single-stage embedding rerank is decent but a cross-encoder second pass improves precision, especially for ambiguous queries.
- **What:** Add a `sentence-transformers.CrossEncoder` pass after embedding rerank (two-stage ranking). Adds a model dependency; gate behind a config flag.

### 13.8 Source citation verification
- **Where:** post-synthesis (new helper, no new node needed) + `synthesize_node`.
- **Why:** Facts now carry `source_url`, but cited URLs are never validated; a dead/hallucinated source can slip into the report.
- **What:** After synthesis, HEAD-check each cited URL; flag broken ones and note uncertainty in the report's Notes section.

### 13.9 Markdown report export with references
- **Where:** `cli.py` / `agent.py` output handling.
- **Why:** Reports are currently saved as flat `.txt` with a sources list.
- **What:** Emit `.md` with YAML frontmatter, footnote-style citations, and a References section, using the `source_url` already attached to each fact.

### 13.10 Playwright JS fallback for trafilatura misses
- **Where:** `fetch_url` in `tools.py`.
- **Why:** trafilatura can't extract JS-heavy pages; those silently return no text.
- **What:** On empty trafilatura extraction, fall back to a headless Playwright render. Optional dependency; gate behind a config flag.

### 13.11 Human-in-the-loop checkpoints
- **Where:** CLI flow (`cli.py`) around `plan` and after `fetch`.
- **Why:** No opportunity to steer decomposition or review sources before synthesis.
- **What:** Pause for approve/reject/revise on the plan and on the fetched source list.

### 13.12 Gradio web UI
- **Why:** No browser interface; current usage is a terminal REPL.
- **What:** Add a Gradio app with streaming research progress and report display.

### 13.13 Embedding fallback to CPU
- **Where:** `create_embedder` in `tools.py`.
- **Why:** Under tight VRAM during bulk fetches, embedding on GPU can OOM.
- **What:** Detect VRAM pressure and fall back to CPU embeddings.

### 13.14 Orchestrator + parallel sub-agent `Send` fan-out (biggest architectural item)
- **Where:** new `sub_agent.py` subgraph + `graph.py` composition (not started in code).
- **Why:** The original design goal — decompose into sub-topics and research them concurrently for breadth and speed.
- **What:** Introduce `SubTask`, `sub_search/sub_fetch/sub_analyze/sub_memory` nodes, `operator.add` reducers, and a `research_round` subgraph fanning out via `Send`. This is the largest change and is intentionally deferred; all items above are achievable on the current monolithic pipeline first.
