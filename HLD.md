# High-Level Design — lite-deep-research-agent

> **Status note:** This document originally described a parallel sub-agent (`Send` fan-out) orchestrator. The **implemented** system is a **monolithic, sequential LangGraph pipeline** (`plan → search → fetch → analyze → should_continue → memory → synthesize`). The sub-agent architecture is a *future* target (see §12 Roadmap) and is **not** in the current code. This document has been updated to describe what is actually implemented, plus the recent optimization pass.

## 1. System Overview

**lite-deep-research-agent** is a local deep-research agent built on LangGraph. It plans search queries from a user question, searches the web, fetches and extracts content, iterates when coverage is thin, and synthesizes a grounded, sourced report. The LLM is served locally by an OpenAI-compatible server (`mlx_lm.server`); embeddings run locally in-process via HuggingFace.

### Design Goals

| Goal | How (current) |
|---|---|
| **Lightweight** | Small local MLX model (ternary 2-bit), local in-process embeddings, zero cloud dependencies |
| **Local-first** | `mlx_lm.server` serves the LLM; HuggingFace embeddings run locally; Chroma for memory; no external APIs |
| **Grounded output** | Facts extracted with `source_url` via structured output; citations flow into the report |
| **Iterative** | `should_continue` loops back to search when coverage is insufficient (safety-capped) |
| **Single process** | Sequential node pipeline; I/O (fetch) parallelized within the fetch node |

---

## 2. Architecture Overview (implemented)

```mermaid
flowchart TD
    query([User query]) --> plan[plan]
    plan --> search[search]
    search --> fetch[fetch]
    fetch --> analyze[analyze]
    analyze --> should{should_continue}
    should -->|"search (loop back)"| search
    should -->|"synthesize"| memory[memory]
    memory --> synthesize[synthesize]
    synthesize --> end([END])

    subgraph loop[Iteration loop]
        search --> fetch --> analyze --> should
    end
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

### 4.1 LLM: OpenAI-compatible server (mlx_lm.server)

| Property | Value |
|---|---|
| Client | `langchain_openai.ChatOpenAI` (OpenAI-compatible) |
| Server | `mlx_lm.server` serving a local HF model (no Ollama) |
| Model | `.env`: `prism-ml/Ternary-Bonsai-27B-mlx-2bit` |
| Base URL | `.env`: `LLM_BASE_URL` (default `http://localhost:8080/v1`) |
| Context window | `LLM_NUM_CTX` (default 262144); native 262K from model `config.json` |
| Embeddings | Local in-process `langchain_huggingface.HuggingFaceEmbeddings` (`.env`: `EMBED_MODEL`, default `sentence-transformers/all-MiniLM-L6-v2`) |
| Structured output | `with_structured_output()` used in `analyze_node` |

`create_llm()` in `tools.py` instantiates `ChatOpenAI` against `LLM_BASE_URL` with `max_retries=2`. For analysis, `analyze_node` binds a lower temperature (`ANALYSIS_TEMPERATURE`, default 0.1) before invoking.

> **Model note:** the chosen LLM is a 2-bit ternary MLX model optimized for Apple Silicon unified memory. It is a *chat* model only — it has no embedding endpoint — so embeddings run in-process via `HuggingFaceEmbeddings` rather than through the server.

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

## 7. LLM Server Integration (mlx_lm.server)

### Setup

```bash
pip install -r requirements.txt
mlx_lm.server \
  --model "prism-ml/Ternary-Bonsai-27B-mlx-2bit" \
  --port 8080 \
  --chat-template-args '{"enable_thinking":false}'
```

The server speaks the OpenAI `/v1` protocol at `http://localhost:8080/v1`. `config.py` points `ChatOpenAI` at `LLM_BASE_URL`. Embeddings run in-process (no server) via `langchain_huggingface.HuggingFaceEmbeddings`.

> **Thinking is disabled.** This is a reasoning model, but LangChain's `ChatOpenAI` does not surface the MLX `reasoning` field — when thinking is on, the response comes back with empty `content` and the reasoning is dropped, so structured fact extraction gets nothing. Launching with `--chat-template-args '{"enable_thinking":false}'` makes the model answer directly (fast, `content` always populated). This flag is only honored at server launch, not per-request.

> **Context window:** the 262K context is native to this model (read from its `config.json`), so there is no `--max-context` flag — `mlx_lm.server` uses the model's own context length. `LLM_NUM_CTX` in `config.py` is documentation only and is not sent to the server.

> **Requires `mlx-lm >= 0.31`** (the `qwen3_5` architecture in this model is unsupported by older releases; upgrade with `pip install --upgrade mlx-lm`).

### Memory Budget (Mac Unified Memory, 4-bit KV cache)

| Component | 64K ctx | 256K ctx |
|---|---|---|
| LLM weights (27B ternary 2-bit, MLX) | 7.57GB | 7.57GB |
| KV cache (4-bit, 16 full-attn layers) | 1.07GB | 4.30GB |
| Linear-attention state (48 layers, fixed) | 0.08GB | 0.08GB |
| Embeddings (in-process, MiniLM) | 0.24GB | 0.24GB |
| Runtime overhead | 1.30GB | 1.30GB |
| **Total** | **~10.3GB** | **~13.5GB** |

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
LLM_MODEL=prism-ml/Ternary-Bonsai-27B-mlx-2bit
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=not-needed
LLM_TEMPERATURE=0.25
LLM_MAX_TOKENS=2048
LLM_NUM_CTX=262144

# Embeddings (local in-process)
EMBED_BACKEND=huggingface
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

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
    ├── setup.sh            # pip install + mlx_lm setup
    └── serve.sh            # Launch mlx_lm.server
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

### 13.15 JSONL traffic log (from `minion`)
- **Where:** new `logs/research.log` writer + `agent.py` research loop.
- **Why:** `agent.py` only prints per-node wall-clock timing; runs are not replayable/debuggable. `minion.py:1204-1214,3038` writes an append-only JSONL of every request/response chunk.
- **What:** Add a JSONL writer that records each node's input/output summary, LLM calls, and errors, so runs can be replayed/inspected offline. Low effort, high debug value.

### 13.16 Recovery from stalled / degenerate model output (from `minion`)
- **Where:** `analyze_node` (empty `extracted_facts`) and `synthesize_node` (empty report).
- **Why:** The 2-bit ternary LLM can emit only reasoning tokens, repetition loops, or empty output. `minion.py:3273-3360,3498-3620` cuts the stream and appends a "[Runtime note: ...]" nudge.
- **What:** Detect empty/low-yield LLM output per node and retry once with a nudge prompt (e.g. "Output only the structured facts") instead of silently proceeding to synthesis.

### 13.17 Retry-with-backoff + connection-failure resilience (from `minion`)
- **Where:** network + LLM call sites in `tools.py` / `nodes.py`.
- **Why:** LDR's search/fetch/LLM calls have no retry layer; a transient error just logs and continues thin. `minion.py:2447-2470,2839-2870` retries connection errors with backoff.
- **What:** Add a small bounded-retry wrapper `with_retry(fn, max_attempts, backoff)` around DDG, the OpenAI-compatible LLM, and Chroma calls.

### 13.18 Fact dedup/cap helper (from `minion`)
- **Where:** after `analyze_node`, before `synthesize_node` (overlaps with §13.3).
- **Why:** `analyze_node` truncates per page but never dedupes facts across pages. `minion.py:2680-2718` dedupes identical consecutive lines (runs ≥3) and caps result size to bound context.
- **What:** Port the dedup/cap helper and apply it to `extracted_facts` before synthesis. Same mechanism as §13.3 cross-source dedup, presented under a shared helper.

### 13.19 Normalized token/time telemetry footer (from `minion`)
- **Where:** end of `research()` in `agent.py`.
- **Why:** Current observability (§10) is basic wall-clock only. `minion.py:1940-1999,3622-3661` normalizes token usage and prints a compact footer (tokens, tok/s, TTFT, ctx util).
- **What:** Record per-node token + latency cost into a normalized summary struct and print it at the end of `research()`.

### 13.20 Backend-agnostic usage normalization layer (from `minion`)
- **Where:** `tools.py` as a thin `normalize_usage(...)`.
- **Why:** `minion.py:1940-1999` never assumes one usage format. Although the server is fixed now, a second backend (roadmap) would reuse this.
- **What:** Build `normalize_usage(...)` now so a future LLM backend swap is trivial.

### 13.21 Compact-counter abbreviation helpers (from `minion`)
- **Where:** `util._abbr(n)` helper used in the CLI footer.
- **Why:** Trivial but useful for a tidy footer. `minion.py:2093-2122` renders counts/char totals compactly (facts=1.5K, chars=78K).
- **What:** Port `util._abbr(n)` and use it in the §13.19 telemetry footer.

### 13.22 Auto context-compression (from `minion`, partial)
- **Where:** right before `synthesize_node`.
- **Why:** LDR has no compression; `extracted_facts` just grow. `minion.py:2880-3035` folds old history when context is full.
- **What:** Lighter version — when `extracted_facts` exceeds `FACT_COMPRESS_THRESHOLD`, summarize/compress them so the synthesis context stays bounded (medium value given the fixed 256K ctx).

