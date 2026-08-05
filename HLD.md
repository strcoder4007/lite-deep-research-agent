# High-Level Design — tiny-deep-researcher

> **Status note (current):** The system is now a **general-purpose local agent** with a custom lightweight tool-calling loop (`agent.py: run()` + `llm.py` + a decorated tool registry in `research_agent/tools/`). **No LangGraph.** The old LangGraph research pipeline (`nodes.py` / `graph.py` / `state.py`) has been **removed** from the repo. See §14 for the agent loop design.
>
> **Status note (historical):** This document originally described a parallel sub-agent (`Send` fan-out) orchestrator, then a **monolithic, sequential LangGraph pipeline** (`plan → search → fetch → analyze → should_continue → memory → synthesize`). Sections 1–13 below describe that removed pipeline, kept here as design history.

## 1. System Overview

**tiny-deep-researcher** is a local deep-research agent built on a custom lightweight tool-calling loop (`agent.py: run()` + `llm.py` + a decorated tool registry in `research_agent/tools/`). **No LangGraph.** It plans search queries from a user question, searches the web, fetches and extracts content, iterates when coverage is thin, and synthesizes a grounded, sourced answer. The LLM is served locally by an OpenAI-compatible server (`mlx_lm.server`); embeddings run locally in-process via HuggingFace.

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
SEARCH_RESULTS_PER_QUERY=5
FETCH_CONCURRENCY=4            # max parallel fetches/searches
REQUEST_TIMEOUT=15
MAX_PAGE_CHARS=5000

# Auto-fetch (automatic URL fetching from search results)
AUTO_FETCH_TOP_N=3              # top URLs to auto-fetch per search (0=disable)
AUTO_FETCH_MAX_TOTAL=6           # max total auto-fetched URLs per turn
AUTO_FETCH_SKIP_DOMAINS=google.com,youtube.com,facebook.com,x.com,twitter.com,instagram.com,tiktok.com,linkedin.com,reddit.com

# Embedding cache
EMBED_CACHE_MAX=5000           # max entries in embedding cache

# Agent loop
MAX_AGENT_STEPS=6                # max tool-call turns per run

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
LANGCHAIN_PROJECT=tiny-deep-researcher
```

---

## 9. File Map

```
tiny-deep-researcher/
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

Current-loop items (details in §13.15+; legacy-pipeline items §13.1–§13.14 are design history):

| Priority | Feature | Status |
|---|---|---|
| Done | Token/time telemetry + per-call status line | ✅ |
| Done | Empty-output nudge retry | ✅ |
| Done | Batched parallel tool calls + auto-fetch top hits | ✅ |
| Done | Auto-fetch hardening (skip domains, quality gate, hard timeout) | ✅ |
| Done | Message-history page truncation (`MESSAGE_PAGE_CHARS`) | ✅ |
| Done | Search-result TTL cache | ✅ |
| Done | S1: digest old tool results in message history | ✅ |
| Done | S2: URL fetch cache + negative cache | ✅ |
| Done | S3: adaptive fetch waves | ✅ |
| Done | S4: streaming final answer (opt-in) | ✅ |
| Done | S6: prompt-prefix stability + TTFT metric | ✅ |
| Done | Q4: cross-source reconciliation in synthesis prompt | ✅ |
| Done | C1: context compression for long runs | ✅ |
| Done | C2: conversation-memory summarization | ✅ |
| Done | C3: per-run token budget guard | ✅ |
| Done | M1: auto-remember research results to Chroma | ✅ |
| Done | M2: memory-first follow-ups | ✅ |
| Done | M3: memory dedup on write | ✅ |
| P1 | Q1: fetch fallback for JS-heavy/blocked pages (reader proxy) | ⬜ |
| P1 | Q2: search backend fallback (s.jina.ai/brave) | ⬜ |
| P1 | Q3: citation verification | ⬜ |
| P1 | S5: two-model split (fast decider + strong synthesizer) | ⬜ |
| P2 | Q5: markdown report export | ⬜ |
| P3 | O1–O3: JSONL traffic log, Gradio UI, human-in-the-loop | ⬜ |

---

## 13. Open Items (detailed)

This section expands on the not-yet-implemented improvements ("remaining improvements") identified while optimizing the pipeline. They are ordered roughly by impact on research quality. None require adding new graph nodes — most are within existing nodes, plus a few architectural items.

### 13.1 Current agent-loop items (active list)

Statuses verified against the code. Grouped by impact area: **Speed**,
**Quality**, **Context management**, **Memory**, **Ops**.

#### Implemented & verified (current loop)

- **Empty-output nudge retry** (was 13.16) — `agent.py` appends
  `config.AGENT_NUDGE` once on empty output, then stops with an error.
- **Token/time telemetry** (was 13.19–13.21) — `logutil.status_line` after
  every LLM call (tok/s · tokens · latency · context bar), `run_summary`
  footer, `abbr()` compact counters, normalized `info` dict in `llm.chat`.
- **Batched parallel tool calls + auto-fetch** (was 13.23) — one JSON array of
  searches executed in parallel; top hits auto-fetched without an extra LLM
  call; same-tool results no longer overwrite each other.
- **Prompt-growth control, part 1** (was 13.24) — fetched pages truncated to
  `MESSAGE_PAGE_CHARS` (2000) in the message history.
- **Auto-fetch hardening** (was 13.28, partial) — skip-domain list, >50%
  skip-domain quality gate, hard per-fetch timeout (`REQUEST_TIMEOUT + 10`,
  executor `shutdown(wait=False)`), volumes `TOP_N=3` / `MAX_TOTAL=6`.
- **Search result cache** (was 13.29, partial) — TTL in-memory cache in
  `web_search`/`web_search_batch` (`SEARCH_CACHE_TTL=1800`).
- **LLM-call failure handling** — the step loop catches timeout/connection
  errors, logs them, and fails the run fast (no more silent 3×180s retries).
- **Plain-text run logs** (was 13.15, partial) — `logs/run_<hash>.log` captures
  every printed line; structured JSONL replay is still open (see O1).
- **S1. Digest old tool results** (13.24) — `agent.py` compresses older
  tool-result messages into one-line digests after `MESSAGE_DIGEST_AFTER` steps.
- **S2. URL fetch cache + negative cache** (13.29) — TTL cache in
  `fetch_url()` keyed by URL; caches both successes and failures.
- **S3. Adaptive fetch waves** (13.28) — after the first auto-fetch wave,
  checks coverage heuristic and fetches additional pages only if thin.
- **S4. Streaming final answer** (13.27) — opt-in `STREAM_FINAL=1` in
  `llm.chat()`: buffers first tokens, suppresses if structured output, else
  streams live to stdout.
- **S6. Prompt-prefix stability + TTFT metric** (13.25) — messages are
  append-only for KV-cache reuse; TTFT measured via streaming and shown in
  `status_line`.
- **Q4. Cross-source reconciliation** — synthesis prompt now instructs the
  model to flag contradictions between sources and prefer authoritative/recent
  ones.

#### Open — Speed

- **S5. Two-model split: fast decider + strong synthesizer** (was 13.26) —
  tool-call turns only pick a tool + args; a 1–3B model does that at several×
  the tok/s of the 9B. Route tool turns to a small model, reserve the 9B for
  the final answer. Biggest raw speed win; needs a second model served.

#### Open — Quality

- **Q1. Fetch fallback for JS-heavy/blocked pages** (port of 13.10) — *Where:*
  `fetch_url` in `tools/base.py`. Run logs show ~half of auto-fetches fail.
  On empty trafilatura extraction, fall back to a reader proxy
  (`https://r.jina.ai/<url>`) before considering Playwright — no browser
  dependency. **Highest quality impact on this list.**
- **Q2. Search backend fallback** (port of 13.6) — *Where:* `run_ddg_search`.
  DDG is the only backend; when it returns < 3 results or errors, the run
  degrades. Add the DDG → `s.jina.ai` → partial chain behind
  `SEARCH_BACKEND`/`SEARCH_FALLBACK` (env keys already exist in `.env`).
- **Q3. Citation verification** (port of 13.8) — post-answer HEAD-check of
  cited URLs; flag dead/hallucinated sources in the report.
- **Q5. Markdown report export** (was 13.9) — `reports/*.md` with frontmatter,
  footnote citations, and a References section instead of flat `.txt`.

#### Open — Context management

_(All items implemented.)_

#### Open — Memory

_(All items implemented.)_

#### Open — Ops

- **O1. JSONL traffic log** (rest of 13.15) — structured request/response
  JSONL per run for offline replay/debug; plain-text logs already exist.
- **O2. Gradio web UI** (was 13.12) — browser UI with streaming progress.
- **O3. Human-in-the-loop checkpoints** (was 13.11) — approve/revise the
  planned searches before execution.

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
│   capped at MAX_AGENT_STEPS (default 6); empty output gets  │
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
2. For `step` in `1..MAX_AGENT_STEPS` (default 6):
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
