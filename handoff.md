# Handoff Document — lite-deep-research-agent

## Project Overview

**What it does:** A general-purpose local agent. The model answers questions by
calling tools — web search, page fetch, long-term memory — one JSON tool call
per turn, until it produces a final answer.

**Architecture (current):** A custom lightweight tool-calling loop —
`agent.py: run()` + `llm.py` + `research_agent/tools/` (`@tool` registry with
auto-discovery; factories/helpers in `tools/base.py`). **No LangGraph.** The
old LangGraph research pipeline (`nodes.py` / `graph.py` / `state.py`) has
been **removed**; HLD.md §1–§13 preserves it as design history.

**Target hardware:** Apple Silicon Mac with ~16 GB unified memory.

**Core models (configurable via `.env`):**
- LLM: `Jackrong/MLX-Qwen3.5-9B-DeepSeek-V4-Flash-4bit` served by `mlx_lm.server`
  (OpenAI-compatible `/v1` on `localhost:8080`). Chat model only.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` in-process (384d).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent loop | Custom tool-calling loop (`agent.py`); no framework |
| Tool registry | `@tool` decorator + auto-discovery (`research_agent/tools/`) |
| LLM | `mlx_lm.server` (OpenAI-compatible), client is `langchain_openai.ChatOpenAI` |
| Embeddings | `langchain_huggingface.HuggingFaceEmbeddings` (in-process) |
| Vector DB | Chroma (persisted to `./advanced_memory/`) |
| Text splitting | `RecursiveCharacterTextSplitter` (chunk=1000, overlap=100) |
| Search | DuckDuckGo (`ddgs`) |
| HTML fetching | `trafilatura` (clean markdown output) |
| Tracing | LangSmith (optional, via env vars) |
| Configuration | `.env` + `config.py` |

---

## Project Structure

```
lite-deep-research-agent/
├── HLD.md                  # High-Level Design (§14 documents the current loop)
├── handoff.md              # This file
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
│   ├── cli.py              # Interactive REPL, saves answer to reports/
│   └── tools/
│       ├── __init__.py     # @tool registry, auto-discovery, catalog, init_tools
│       ├── base.py         # factories + helpers (was tools.py)
│       ├── web_search.py   # web_search tool (DuckDuckGo)
│       ├── fetch_page.py   # fetch_page tool (trafilatura)
│       ├── memory_tools.py # recall_memory / remember tools
│       └── finalize.py     # final_answer tool + sentinel
│
└── scripts/
    ├── setup.sh            # pip install helper
    └── serve.sh            # Launch mlx_lm.server
```

---

## Agent Loop (`agent.py: run()`)

1. `messages = [system(build_system_prompt(catalog)), user(query)]`.
2. For `step` in `1..MAX_AGENT_STEPS` (default 12):
   - `text, call = llm.chat(messages, llm)`.
   - Empty output → append the nudge (`config.AGENT_NUDGE`) and retry once;
     still empty → stop with an error.
   - No tool call → `text` is the final answer; stop.
   - Tool call → look up `TOOL_REGISTRY`; execute with `**args`. `TypeError`
     (bad args) and other exceptions are caught and fed back to the model as
     the tool result. `final_answer` (or its sentinel) ends the loop.
3. Per-step logging via `logutil.tool_step`: `step N | tool=<name> | preview`.
   After any `web_search`, the top `AUTO_FETCH_TOP_N` URLs per search are
   fetched automatically in parallel (no extra model call) and appended as
   tool results — the fast research path is: batched searches → auto-fetch →
   final answer (~2 LLM calls).
4. Returns `{query, answer, steps, errors}`.

### Tool-call format

Robust for a weak 2-bit local model. The system prompt instructs: to use a
tool, reply with ONLY a JSON block; ONE tool call per turn:

    ```json
    {"tool": "<name>", "args": {"<arg>": <value>}}
    ```

Any other reply is treated as the final answer. Parsing reuses
`_extract_json_object` (tolerates code fences / surrounding prose).

### Adding a tool

Create `research_agent/tools/<name>.py` with a `@tool`-decorated function.
Submodules are auto-imported on package import; no other wiring. Shared
`ResearchTools` (llm/embedder/vectorstore/text_splitter) are built once via
`build_tools()` and passed to tool modules through `tools.init_tools()`.

---

## LLM Server (mlx_lm.server)

```bash
pip install -r requirements.txt
pip install --upgrade mlx-lm   # needs >= 0.31 for the qwen3_5 architecture

mlx_lm.server \
  --model "Jackrong/MLX-Qwen3.5-9B-DeepSeek-V4-Flash-4bit" \
  --port 8080
```

- **Lazy model load:** the server binds the port immediately and loads the
  ~5 GB weights on the first request. Silence after "Download complete"
  means *ready*, not stuck; the first generation request pays the load cost.
- **No `--chat-template-args`:** unlike the old 27B, this model's chat
  template has no `enable_thinking` switch, so the flag is a no-op and has
  been dropped from the launch command.
- The model's native 262K context comes from its `config.json`; there is no
  server flag for it.

---

## Memory Strategy

- **Storage:** Chroma persisted to `./advanced_memory/`, collection
  `research_memory`.
- **Tools:** `remember(text, title)` writes chunks; `recall_memory(query,
  top_k)` reads with a recency boost `1 / (1 + age_hours / 24)` on similarity
  scores.
- Chunks below `MEMORY_MIN_CHARS` are skipped on write; reads filter by
  `MEMORY_SIMILARITY_THRESHOLD`.

---

## Configuration

All env-read constants live in `config.py`; edit `.env` directly. Key keys:
`LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_TIMEOUT`, `EMBED_MODEL`,
`SEARCH_RESULTS_PER_QUERY`, `REQUEST_TIMEOUT`, `MAX_PAGE_CHARS`,
`MAX_AGENT_STEPS`, `MEMORY_DIR`, `MEMORY_TOP_K`,
`MEMORY_SIMILARITY_THRESHOLD`, `MEMORY_MIN_CHARS`, `CHUNK_SIZE`,
`CHUNK_OVERLAP`. Optional LangSmith: `LANGCHAIN_TRACING_V2`,
`LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`.

---

## Known Limitations / Roadmap

See HLD.md §13 for the detailed list. Highlights still relevant to the
current loop:

- **JSONL traffic log** — runs are not replayable/debuggable (§13.15).
- **Retry-with-backoff** around DDG / LLM / Chroma calls (§13.17).
- **Token/time telemetry footer** at end of `run()` (§13.19–13.21).
- **`s.jina.ai` / `brave` search fallback** — DDG is the only backend (§13.6).
- **Playwright JS fallback** for pages trafilatura can't extract (§13.10).
- **Gradio web UI** (§13.12) and **markdown report export** (§13.9).

Items tied to the removed LangGraph pipeline (coverage assessment, fact
dedup, sub-agent `Send` fan-out, etc.) are design history in HLD.md §13.
