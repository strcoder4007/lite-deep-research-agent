# High-Level Design — lite-deep-research-agent

> **Status note (current):** The system is now a **general-purpose local agent** with a custom lightweight tool-calling loop (`agent.py: run()` + `llm.py` + a decorated tool registry in `research_agent/tools/`). **No LangGraph.** The old LangGraph research pipeline (`nodes.py` / `graph.py` / `state.py`) has been **removed** from the repo. See §14 for the agent loop design.
>
> **Status note (historical):** This document originally described a parallel sub-agent (`Send` fan-out) orchestrator, then a **monolithic, sequential LangGraph pipeline** (`plan → search → fetch → analyze → should_continue → memory → synthesize`). Sections 1–13 below describe that removed pipeline, kept here as design history.

## 1. System Overview

**lite-deep-research-agent** is a local deep-research agent built on a custom lightweight tool-calling loop (`agent.py: run()` + `llm.py` + a decorated tool registry in `research_agent/tools/`). **No LangGraph.** It plans search queries from a user question, searches the web, fetches and extracts content, iterates when coverage is thin, and synthesizes a grounded, sourced answer. The LLM is served locally by an OpenAI-compatible server (`mlx_lm.server`); embeddings run locally in-process via HuggingFace.

### Design Goals

| Goal | How (current) |
|---|---|
| **Lightweight** | Small local MLX model (9B 4-bit), local in-process embeddings, zero cloud dependencies |
| **Local-first** | `mlx_lm.server` serves the LLM; HuggingFace embeddings run locally; Chroma for memory; no external APIs |
| **Grounded output** | Facts extracted with `source_url` via structured output; citations flow into the answer |
| **Iterative** | The agent loops back to search when coverage is insufficient (safety-capped at `MAX_AGENT_STEPS`) |
| **Parallel I/O** | Multiple tool calls per turn execute concurrently via `ThreadPoolExecutor`; auto-fetch runs in parallel |

---

## 2. Architecture Overview (current)

```mermaid
flowchart TD
    query([User query]) --> run[agent.run]
    run --> plan[Plan searches]
    plan --> search[web_search / web_search_batch]
    search --> auto_fetch[Auto-fetch top URLs]
    auto_fetch --> llm[llm.stream_chat]
    llm --> tool_exec{Tool calls?}
    tool_exec -->|Yes| exec[Execute in parallel]
    exec --> search
    tool_exec -->|No| answer[Final answer]
    answer --> end([END])
```

The current system is a custom tool-calling loop in `agent.py: run()`.
The model plans all independent searches upfront and emits them as a
single JSON array; multiple tool calls per turn execute in parallel via
`ThreadPoolExecutor`. After `web_search` results arrive, the top URLs
are auto-fetched in parallel. The model streams its response via
`llm.stream_chat()`.

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
| Model | `.env`: `Jackrong/MLX-Qwen3.5-9B-DeepSeek-V4-Flash-4bit` |
| Base URL | `.env`: `LLM_BASE_URL` (default `http://localhost:8080/v1`) |
| Context window | native 262K from model `config.json` |
| Embeddings | Local in-process `langchain_huggingface.HuggingFaceEmbeddings` (`.env`: `EMBED_MODEL`, default `sentence-transformers/all-MiniLM-L6-v2`) |
| Streaming | `llm.stream_chat()` uses `ChatOpenAI.stream()` for real-time token output |
| Caching | `CachingEmbeddings` wrapper avoids redundant embedding computation |

`create_llm()` in `tools/base.py` instantiates `ChatOpenAI` against `LLM_BASE_URL` with `max_retries=2`. Embeddings are cached in-process via `CachingEmbeddings`.

### 4.2 Search: DuckDuckGo (`ddgs`)

`run_ddg_search()` in `tools/base.py` queries DuckDuckGo via the `ddgs` library (text or news, depending on `SEARCH_TIME_LIMIT`). Results are normalized to `{url, title, snippet, published_at}`. A thread-local DDGS session singleton (`_get_ddg_session()`) is reused across calls to avoid creating new sessions per search.

`web_search_batch()` in `tools/web_search.py` runs multiple queries in parallel via `ThreadPoolExecutor`.

> The original design specified `s.jina.ai` and `brave` fallbacks. These are **not** implemented in the current code — DuckDuckGo is the only backend.

### 4.3 Fetch: trafilatura (parallel)

`fetch_url()` in `tools/base.py` uses `trafilatura.fetch_url()` then `trafilatura.extract()` with `output_format="markdown"`, `favor_precision=True`, `include_comments=False`, `include_tables=False`. Output is truncated to `MAX_PAGE_CHARS` (default 5000). The download timeout is set via the trafilatura config object (`DEFAULT` / `DOWNLOAD_TIMEOUT`), compatible with trafilatura ≥ 2.1 which removed the `timeout` kwarg.

Multiple URLs are fetched concurrently via `ThreadPoolExecutor` in `fetch_pages()` (`tools/fetch_pages.py`). After `web_search` results arrive, the top `AUTO_FETCH_TOP_N` URLs per search are auto-fetched in parallel.

### 4.4 Memory: Chroma + ConversationMemory

Persisted to `./advanced_memory/`, collection `research_memory`. `memory.py` provides:
- `ConversationMemory` — in-memory store of previous query/answer turns for the REPL
- `add_to_memory()` — splits text, writes chunks, persists to Chroma
- `query_memory()` — similarity search with recency boost `1/(1+age_hours/24)`

`add_to_memory()` writes each page; `recall_memory` reads from Chroma. Embedding computations are cached via `CachingEmbeddings` to avoid redundant model loads.

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

After extraction, `_dedupe_facts()` performs **cross-source fact deduplication**: each fact's claim (without the source suffix) is embedded, and facts with cosine similarity ≥ `FACT_DEDUP_THRESHOLD` (default 0.9) to an already-kept fact are dropped. Falls back to the raw list if embedding fails; can be disabled via `FACT_DEDUP_ENABLED=0`.

### 4.6 Fetch timeout configuration (trafilatura ≥ 2.1)

`trafilatura.fetch_url()` no longer accepts a `timeout` keyword (removed in trafilatura 2.1 — passing it raises `TypeError`, which previously was swallowed and made **every** fetch fail silently). The download timeout is now set via the trafilatura config object (`DEFAULT` / `DOWNLOAD_TIMEOUT`) in `fetch_url()`.

---

## 5. Graph Design (`graph.py` / `nodes.py`)

| Node | Input | Action | Output keys |
|---|---|---|---|
| `plan` | `query` | LLM → YAML → parse into queries/aspects/gaps | `research_plan`, `search_queries`, `plan_gaps` |
| `search` | `search_queries` | DDG search + dedup + embedding rerank | `search_results` |
| `fetch` | `search_results` | parallel trafilatura fetch + memory write | `fetched_content` |
| `analyze` | `fetched_content` | per-page structured fact extraction + cross-source dedup | `extracted_facts` |
| `should_continue` | `fetched_content`, `extracted_facts`, `research_plan`, `iteration` | LLM coverage assessment (heuristic fallback) | `next_step`, `search_queries`, `iteration` |
| `memory` | `query`, `plan_gaps`, `fetched_content` | Chroma query for prior-run context (current-run pages excluded) | `relevant_memory` |
| `synthesize` | `extracted_facts`, `relevant_memory` | LLM report with inline citations | `final_answer`, `sources` |

### Loop control

`should_continue` first asks the LLM (`_assess_coverage`) to judge whether the extracted facts cover the user query and the plan's `KEY_ASPECTS`. The model returns `{"sufficient": bool, "missing": [...], "new_queries": [...]}`; when coverage is insufficient, the targeted `new_queries` are appended (deduped) and `next_step="search"`. The `MAX_ITERATIONS` hard cap always wins. If the assessment call fails or returns unparseable output (or `COVERAGE_ASSESSMENT_ENABLED=0`), the old heuristic applies: continue when `fetched < MIN_FETCHED_FOR_STOP` OR `facts < MIN_FACTS_FOR_STOP` OR `plan_gaps`, appending naive `f"{query} {gap}"` queries.

---

## 6. Data Flow (Request Lifecycle)

```
User query
  → plan: 4–5 search queries + aspects + gaps
  → search: DDG → dedup → embedding rerank → top-N
  → fetch: parallel trafilatura fetch (≤ FETCH_LIMIT pages) → write to Chroma
  → analyze: per-page structured facts (claim + source_url) → cross-source dedup
  → should_continue: LLM coverage assessment (heuristic fallback)
        no  → append targeted follow-up queries, iteration++, loop to search
        yes → memory
  → memory: Chroma query with original query + gaps (current-run pages excluded)
  → synthesize: report from facts + memory, with inline source markers
  → END
```

---

## 7. LLM Server Integration (mlx_lm.server)

### Setup

```bash
pip install -r requirements.txt
mlx_lm.server \
  --model "Jackrong/MLX-Qwen3.5-9B-DeepSeek-V4-Flash-4bit" \
  --port 8080
```

The server speaks the OpenAI `/v1` protocol at `http://localhost:8080/v1`. `config.py` points `ChatOpenAI` at `LLM_BASE_URL`. Embeddings run in-process (no server) via `langchain_huggingface.HuggingFaceEmbeddings`.

> **Lazy model load.** The server binds the port immediately and loads the weights on the first request — silence after "Download complete" means *ready*, not stuck. The first generation request pays the model-load cost.

> **No `--chat-template-args`.** Unlike the old 27B, this model's chat template has no `enable_thinking` switch, so the flag is a no-op and has been dropped from the launch command.

> **Context window:** the 262K context is native to this model (read from its `config.json`), so there is no `--max-context` flag — `mlx_lm.server` uses the model's own context length.

> **Requires `mlx-lm >= 0.31`** (the `qwen3_5` architecture in this model is unsupported by older releases; upgrade with `pip install --upgrade mlx-lm`).

### Memory Budget (Mac Unified Memory, 4-bit KV cache, approximate)

| Component | 64K ctx | 256K ctx |
|---|---|---|
| LLM weights (9B 4-bit, MLX) | ~5.3GB | ~5.3GB |
| KV cache (4-bit, 8 full-attn layers) | ~0.5GB | ~2.0GB |
| Linear-attention state (24 layers, fixed) | ~0.05GB | ~0.05GB |
| Embeddings (in-process, MiniLM) | 0.24GB | 0.24GB |
| Runtime overhead | 1.30GB | 1.30GB |
| **Total** | **~7.4GB** | **~8.9GB** |

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
LLM_MODEL=Jackrong/MLX-Qwen3.5-9B-DeepSeek-V4-Flash-4bit
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=not-needed
LLM_TEMPERATURE=0.25
LLM_MAX_TOKENS=2048
LLM_TIMEOUT=180

# Embeddings (local in-process)
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Search / fetch
SEARCH_RESULTS_PER_QUERY=8
FETCH_CONCURRENCY=4            # max parallel fetches/searches
REQUEST_TIMEOUT=12
MAX_PAGE_CHARS=5000

# Auto-fetch (automatic URL fetching from search results)
AUTO_FETCH_TOP_N=6             # top URLs to auto-fetch per search (0=disable)
AUTO_FETCH_MAX_TOTAL=14        # max total auto-fetched URLs per turn
AUTO_FETCH_SKIP_DOMAINS=google.com,youtube.com,facebook.com,x.com,twitter.com,instagram.com,tiktok.com,linkedin.com,reddit.com

# Embedding cache
EMBED_CACHE_MAX=5000           # max entries in embedding cache

# Agent loop
MAX_AGENT_STEPS=12               # max tool-call turns per run

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
│   ├── __init__.py         # re-exports run()
│   ├── __main__.py         # Entry point (loads .env, runs CLI)
│   ├── config.py           # All env-read constants
│   ├── agent.py            # run() — custom tool-calling loop
│   ├── llm.py              # system prompt + tool-call parsing
│   ├── memory.py           # Chroma add/query, Scratchpad
│   ├── logutil.py          # colors, previews, per-step tool-call logging
│   ├── cli.py              # Interactive REPL (no report saving)
│   └── tools/
│       ├── __init__.py     # @tool registry, auto-discovery, catalog, init_tools
│       ├── base.py         # factories + helpers (was tools.py)
│       ├── web_search.py   # web_search tool (DuckDuckGo)
│       ├── fetch_page.py   # fetch_page tool (trafilatura)
│       ├── memory_tools.py # recall_memory / remember tools
│       └── finalize.py     # final_answer tool + sentinel
│
└── scripts/
    ├── setup.sh            # pip install + mlx_lm setup
    └── serve.sh            # Launch mlx_lm.server
```

> Note: the removed LangGraph pipeline (`nodes.py` / `graph.py` / `state.py`) is no longer in the repo; §1–§13 describe it as design history.

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
| P1 | Shrink per-step prompt growth — truncate/digest tool results (§13.24) | ⬜ |
| P1 | Two-model split: fast decider + strong synthesizer (§13.26) | ⬜ |
| P1 | Adaptive fetch depth — stop fetching when coverage suffices (§13.28) | ⬜ |
| P2 | Prompt-prefix stability + KV-cache reuse across steps (§13.25) | ⬜ |
| P2 | Streaming final answer, tool-call-suppressed (§13.27) | ⬜ |
| P2 | Cross-turn search/fetch caching (§13.29) | ⬜ |
| P1 | Orchestrator + parallel sub-agent `Send` fan-out architecture | ⬜ not started |
| P1 | Cross-encoder reranking for search results | ⬜ |
| P1 | Pydantic structured output for `plan_node` (replace YAML) | ⬜ |
| P2 | Human-in-the-loop checkpoints | ⬜ |
| P2 | Gradio web UI | ⬜ |
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

### 13.5 Better analysis context & cross-source reconciliation
- **Where:** `analyze_node` in `nodes.py`.
- **Why:** Each page is still analyzed in isolation with only `ANALYSIS_SNIPPET_CHARS` (4000) of context. Contradictions/redundancy across sources aren't reconciled, and large pages are truncated.
- **What:** Pass already-extracted facts from prior pages as context so later pages can corroborate/conflict; consider feeding more of each page now that 256K context is available. Optionally raise `ANALYSIS_SNIPPET_CHARS`.
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

### 13.15 JSONL traffic log (from `minion`) — ⏳ PARTIAL
- **Status:** plain-text per-run logs (`logs/run_<hash>.log`) now capture every
  printed line (steps, tool results, errors, answer) via `logutil.set_log_file`.
  Still missing: structured, machine-replayable JSONL of request/response pairs.

### 13.16 Retry-with-backoff + connection-failure resilience (from `minion`)
- **Where:** network + LLM call sites in `tools.py` / `nodes.py`.
- **Why:** LDR's search/fetch/LLM calls have no retry layer; a transient error just logs and continues thin. `minion.py:2447-2470,2839-2870` retries connection errors with backoff.
- **What:** Add a small bounded-retry wrapper `with_retry(fn, max_attempts, backoff)` around DDG, the OpenAI-compatible LLM, and Chroma calls.

### 13.18 Fact dedup/cap helper (from `minion`)
- **Where:** after `analyze_node`, before `synthesize_node` (overlaps with §13.3).
- **Why:** `analyze_node` truncates per page but never dedupes facts across pages. `minion.py:2680-2718` dedupes identical consecutive lines (runs ≥3) and caps result size to bound context.
- **What:** Port the dedup/cap helper and apply it to `extracted_facts` before synthesis. Same mechanism as §13.3 cross-source dedup, presented under a shared helper.

### 13.19 Shrink per-step prompt growth (biggest measured cost)
- **Where:** `agent.py` message assembly for tool results.
- **Why:** prompt tokens dominate spend (typical run: ~10K prompt vs ~1.4K
  completion) and every step re-sends the full history. Raw `fetch_page`
  results (`MAX_PAGE_CHARS=5000` each × up to 14 pages) are appended verbatim.
- **What:** truncate fetched pages harder for the *message* copy (e.g. 1.5–2K
  chars, full text still available to `remember`), drop `snippet` fields from
  search results once the page was fetched, and collapse older tool results to
  one-line digests after N steps.

### 13.20 Prompt-prefix stability for MLX/KV-cache reuse
- **Where:** `agent.py` message list + `llm.build_system_prompt`.
- **Why:** `mlx_lm.server` can reuse KV for shared prefixes, but the system
  prompt embeds the current date and the loop appends per-step messages —
  fine — while any mid-conversation mutation of early messages would bust the
  cache. Worth verifying the prefix is strictly append-only and, if the server
  exposes a prompt cache, that we actually hit it (measure TTFT across steps).
- **What:** keep system prompt byte-stable within a run; append-only messages;
  add a TTFT metric to `status_line` to confirm cache hits.

### 13.21 Two-model split: fast decider, strong synthesizer
- **Where:** `config.py` (`LLM_MODEL` vs `LLM_DECIDER_MODEL`) + `agent.py`.
- **Why:** tool-call turns need only pick a tool + args — a tiny 1–3B model
  does that at several× the tok/s of the 9B. The 9B is only needed for the
  final synthesis.
- **What:** route tool-call turns to a small model (second `mlx_lm.server`
  instance or same server, second model), reserve the main model for the final
  answer. Optional: speculative pattern — small model drafts, big model only
  answers.

### 13.22 Streaming final answer (perceived latency / TTFT) — ✅ IMPLEMENTED
- **Where:** `llm.py` (`stream_chat`) + `agent.py` (uses `stream_chat` for all LLM calls).
- **Implementation:** `llm.stream_chat()` uses `llm.stream(messages)` to yield
  tokens via SSE, printing each chunk to stdout in real time. The agent loop
  uses `stream_chat()` for all turns so the final answer streams as it's
  generated. A first-token buffer suppresses streaming when the output starts
  with `{`/`[`/code fence (tool call JSON).

### 13.23 Batched parallel tool calls + auto-fetch — ✅ IMPLEMENTED
- **Where:** `agent.py` step loop + `llm.py` system prompt.
- **Implementation:** the model plans all independent searches upfront and emits
  them as one JSON array; calls run in parallel via `ThreadPoolExecutor`.
  After any `web_search`, the top `AUTO_FETCH_TOP_N` URLs per search are
  fetched automatically in parallel (no fetch-planning LLM call) and appended
  as tool results. Fast path: 1 LLM call (batched searches) → searches +
  auto-fetch → 1 LLM call (final answer).

### 13.24 Adaptive fetch depth (stop early when coverage is enough)
- **Where:** `agent.py` auto-fetch block (`AUTO_FETCH_TOP_N`/`_MAX_TOTAL`).
- **Why:** auto-fetch always pulls up to 14 pages; for easy questions the first
  2–3 pages suffice, and every extra page costs fetch time + prompt tokens on
  all subsequent steps.
- **What:** cheap heuristic gate — fetch in waves (e.g. 3 at a time), stop when
  total fetched chars / distinct sources cross a threshold, or when the query
  needs no fresh sources at all (small talk, follow-ups answerable from
  history).

### 13.25 Cross-turn caching of search/fetch results
- **Where:** `tools/web_search.py` / `tools/fetch_page.py` (TTL LRU).
- **Why:** conversational follow-ups often re-search/re-fetch the same or
  similar queries within a session; each repeat costs full network + model
  turns. Chroma memory already stores page text — reuse it as a fetch cache.
- **What:** TTL-bounded in-memory cache keyed by normalized query/URL (e.g.
  30 min), optionally backed by a `recall_memory` similarity probe before
  fetching a URL that was recently fetched.


---

## 14. Agent Loop (current implementation)

The LangGraph pipeline (§1–§13) is deprecated. The current system is a
general-purpose local agent with a custom lightweight tool-calling loop —
no agent framework.

```
user query
   │
   ▼
┌──────────────────────── agent.run() ─────────────────────────┐
│                                                              │
│   messages = [system(build_system_prompt(TOOL_CATALOG)),     │
│               user(query)]                                   │
│                    │                                         │
│                    ▼                                         │
│              ┌──────────┐   plain text                       │
│      ┌─────► │ llm.chat │ ──────────────►  answer            │
│      │       └────┬─────┘                                    │
│      │            │  {"tool": "<name>", "args": {...}}       │
│      │            ▼                                          │
│      │     TOOL_REGISTRY[name](**args)                       │
│      │       web_search · fetch_page · recall_memory ·       │
│      │       remember · final_answer (sets sentinel → stop)  │
│      │            │                                          │
│      └────────────┘  tool result appended to messages        │
│                                                              │
│   capped at MAX_AGENT_STEPS (default 12); empty output gets  │
│   one AGENT_NUDGE retry; bad args fed back as tool result    │
└──────────────────────────────────────────────────────────────┘
                    │
                    ▼
   DuckDuckGo (ddgs) · trafilatura · Chroma (./advanced_memory/)
   LLM: mlx_lm.server OpenAI /v1 at LLM_BASE_URL
```

### 14.1 Layout

| File | Role |
|---|---|
| `research_agent/tools/__init__.py` | `@tool` decorator, `TOOL_REGISTRY`, submodule auto-discovery, `build_catalog()`, `init_tools()`/`get_shared()` |
| `research_agent/tools/base.py` | Original factories + helpers (was `tools.py`): `build_tools`, `run_ddg_search`, `fetch_url`, `_extract_json_object`, `count_tokens`, … |
| `research_agent/tools/web_search.py` | `web_search(query, max_results=8)` → `[{url, title, snippet}]` |
| `research_agent/tools/fetch_page.py` | `fetch_page(url)` → `{url, title, text}` or error dict |
| `research_agent/tools/memory_tools.py` | `recall_memory(query, top_k=5)`, `remember(text, title="")` |
| `research_agent/tools/finalize.py` | `final_answer(answer)` — sets a sentinel the loop reads to terminate |
| `research_agent/llm.py` | `build_system_prompt(catalog)`, `chat(messages, llm)` → `(text, toolcall_or_none)` |
| `research_agent/agent.py` | `run(query)` — the step loop |

Adding a tool = create `tools/<name>.py` with a `@tool`-decorated function.
Submodules are auto-imported on package import; no other wiring.

### 14.2 Tool-call format

Robust for a weak 2-bit local model. The system prompt instructs: to use a
tool, reply with ONLY a JSON block or a JSON array of blocks; plain text
is treated as the final answer. Parsing reuses
`_extract_json_object` (tolerates code fences / surrounding prose); a
parsed dict with a string `tool` key counts as a tool call. Multiple
tool calls in a single turn are executed in parallel via `ThreadPoolExecutor`.

### 14.3 Loop algorithm (`agent.run`)

1. `messages = [system(catalog), user(query)]`.
2. For `step` in `1..MAX_AGENT_STEPS` (default 12):
   - `text, calls, info = llm.stream_chat(messages, llm)`.
   - Empty output → append the nudge (`config.AGENT_NUDGE`) and retry once
     (the §13.16 pattern); still empty → stop with an error.
   - No tool calls → `text` is the final answer; stop.
   - Tool calls → execute ALL calls in parallel via `ThreadPoolExecutor`.
     `TypeError` (bad args) and other exceptions are caught and fed back to
     the model as the tool result. After all results arrive, append assistant
     message and tool-result messages. `final_answer` (or its sentinel) ends
     the loop.
   - After `web_search` results, auto-fetch the top `AUTO_FETCH_TOP_N` URLs
     per search in parallel (skipping known-bad domains like google.com,
     youtube.com, social media).
3. Per-step logging via `logutil.tool_step`: `step N | tool=<name> | preview`.
4. Returns `{query, answer, steps, errors}`.

Shared `ResearchTools` (llm/embedder/vectorstore/text_splitter) are built
once via `build_tools()` (singleton) and passed to tool modules through
`tools.init_tools()` at loop start.
