# Handoff Document — lite-deep-research-agent

## Project Overview

**What it does:** A LangGraph-based deep research agent. An orchestrator decomposes the user query into search queries, runs a single-pass search → fetch → analyze pipeline, iterates when coverage is insufficient, and synthesizes a grounded, sourced research report.

**Architecture:** A single LangGraph `StateGraph` pipeline (monolithic, sequential nodes). The orchestrator plans queries, searches, fetches pages, extracts facts, decides whether to continue, queries memory, and synthesizes. All state is carried in one `ResearchState` TypedDict. (A parallel sub-agent `Send` fan-out design is a *future* target — see roadmap — but is **not** implemented yet.)

**Target hardware:** Mac with Apple Silicon (M1/M2/M3) with 16GB+ unified memory. Runs entirely via Ollama.

**Core models (configurable via `.env`):**
- LLM: `qwen3.5:4b` via Ollama by default in `.env` (config.py default `qwen3:8b-q4_K_M`). 256K context via `num_ctx` is recommended.
- Embeddings: `nomic-embed-text` via Ollama by default in `.env` (config.py default `mxbai-embed-large`).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph (`StateGraph`, sequential node pipeline) |
| LLM | Ollama — `langchain_ollama.ChatOllama` |
| Embeddings | Ollama — `langchain_ollama.OllamaEmbeddings` |
| Vector DB | Chroma (persisted to `./advanced_memory/`) |
| Text splitting | `RecursiveCharacterTextSplitter` (chunk=1000, overlap=100) |
| Search | DuckDuckGo (`ddgs`) |
| HTML fetching | `trafilatura` (multi-extractor pipeline, clean markdown output) |
| Structured output | Pydantic models via `with_structured_output()` for fact extraction |
| Tracing | LangSmith (optional, via env vars) |
| Configuration | `.env` + `config.py` |

> **Note:** The earlier design called for a `s.jina.ai` fallback and a `brave` backend. The current implementation uses DuckDuckGo only (no fallback). Structured-output fact extraction is implemented in `analyze_node`; the `plan_node` still uses YAML parsing.

---

## Project Structure

```
lite-deep-research-agent/
├── HLD.md                         # High-Level Design (updated to match implementation)
├── handoff.md                     # This file
├── requirements.txt
├── .env.example
│
├── research_agent/
│   ├── __init__.py
│   ├── __main__.py                # Entry: loads .env, runs CLI
│   ├── config.py                  # All env-read constants
│   ├── tools.py                   # ResearchTools, create_llm/embedder/vectorstore, DDG search, trafilatura fetch, helpers
│   ├── state.py                   # ResearchState TypedDict, helpers
│   ├── memory.py                  # Chroma add/query with recency boosting
│   ├── nodes.py                   # All orchestrator nodes: plan, search, fetch, analyze, should_continue, memory, synthesize
│   ├── graph.py                   # create_research_graph() — composes the pipeline
│   ├── agent.py                   # AdvancedResearchAgent — synchronous stream with logging
│   └── cli.py                     # Interactive REPL with report output
│
└── scripts/
    ├── setup.sh                   # pip install, ollama pull
    └── serve.sh                   # Launch Ollama (optional helper)
```

---

## Architecture

### Graph Flow (current implementation)

```
plan → search → fetch → analyze → should_continue
                                            │
                         ┌──────────────────┼──────────────────┐
                         ▼                                     ▼
                     "search"                            "synthesize"
                         │                                     │
                     [re-enter                          memory → synthesize → END
                     search → fetch → analyze]
```

### Orchestrator Nodes (all in `nodes.py`)

| Node | Role |
|---|---|
| `plan` | LLM decomposes the user query into 4–5 search queries + key aspects + gaps (YAML output, parsed). |
| `search` | Runs DuckDuckGo (`run_ddg_search`) for each query. Deduplicates by URL/host, then embedding-reranks with a recency bonus. |
| `fetch` | Fetches each result page with `trafilatura` in parallel (semaphore-limited). Writes content to Chroma memory. |
| `analyze` | For each fetched page, an LLM call extracts **structured facts** (`FactItem{claim, source_url}`) via `with_structured_output`. Larger context window than before. |
| `should_continue` | Heuristic: continues (back to `search`) if too few pages/facts were gathered or gaps remain; otherwise proceeds to `memory` → `synthesize`. Hard cap at `MAX_ITERATIONS`. |
| `memory` | Global Chroma query with the original user query (+ plan gaps/aspects) for broad context. |
| `synthesize` | LLM builds a structured report (Executive Summary → Findings → Analysis → Conclusion → Notes) with inline source markers from the extracted facts. |

---

## State Model (`ResearchState`)

Defined as a `TypedDict` (all fields `total=False`) in `state.py`. There are **no** accumulator reducers (e.g. `operator.add`); nodes overwrite the keys they own and the pipeline carries a single growing list of facts via plain reassignment.

| Key | Type | Purpose |
|---|---|---|
| `query` | `str` | Original user question |
| `research_plan` | `Dict[str, Any]` | Parsed plan (SEARCH_QUERIES, KEY_ASPECTS, GAPS_TO_ADDRESS) |
| `search_queries` | `List[str]` | Queries to run this round (accumulate across iterations) |
| `search_results` | `List[Dict[str, Any]]` | Reranked search results (url/title/snippet/published_at/score) |
| `fetched_content` | `List[Dict[str, Any]]` | Fetched pages (url/title/text/metadata) |
| `extracted_facts` | `List[str]` | Facts as strings, each formatted `claim (source: url)` |
| `relevant_memory` | `List[Dict[str, Any]]` | Chunks retrieved from Chroma in `memory_node` |
| `final_answer` | `str` | Synthesized report |
| `sources` | `List[str]` | Source URLs cited in the report |
| `iteration` | `int` | Current research round |
| `max_iterations` | `int` | Safety cap (default 2 in config.py; 5 in `.env`) |
| `plan_gaps` | `List[str]` | Gaps carried from the plan; drive continuation |
| `next_step` | `str` | "search" or "synthesize" |
| `errors` / `messages` | `List[str]` | Human-readable logs |

---

## Ollama Setup

### Quick Start

```bash
# One-time setup (installs deps; ensure langchain_ollama is present)
bash scripts/setup.sh

# Start Ollama
ollama serve

# Run agent
python -m research_agent
```

### Model Config (256K context)

Create a Modelfile for 256K context:

```dockerfile
# Modelfile.qwen3.5-4b-256k
FROM qwen3.5:4b
PARAMETER num_ctx 262144
PARAMETER temperature 0.25
PARAMETER num_thread 4
```

Apply: `ollama create qwen3.5-4b-256k -f Modelfile.qwen3.5-4b-256k`

Then use `qwen3.5-4b-256k` as `LLM_MODEL` in `.env`.

### Structured Output

Fact extraction (`analyze_node`) uses Pydantic `with_structured_output(AnalyzeOutput)` where:

```python
class FactItem(BaseModel):
    claim: str
    source_url: str

class AnalyzeOutput(BaseModel):
    facts: List[FactItem]
```

The `plan_node` still uses YAML parsing (a known fragility, see roadmap).

---

## Search & Fetch Strategy

### Search: DuckDuckGo

```python
# config.py
SEARCH_RESULTS_PER_QUERY = 8   # results per query
SEARCH_RERANK_TOP_N = 10       # keep top-N after embedding rerank
SEARCH_SINCE_DAYS / SEARCH_DATE_FROM / SEARCH_DATE_TO  # optional recency filters
```

DuckDuckGo is the sole backend (via the `ddgs` library). The earlier `s.jina.ai`/`brave` fallback is **not** wired in the current code.

> **Recency filter note:** `tools._inject_date_filters` appends DuckDuckGo `after:/before:` date tokens when recency is requested. A previous bug that also appended an invalid literal `site:news` token (which silently killed recall) has been **removed**.

### Fetch: trafilatura

```python
text = trafilatura.extract(
    downloaded,
    output_format="markdown",
    favor_precision=True,
    include_comments=False,
    include_tables=False,
)
```

- Pages are fetched **in parallel** with `asyncio.gather` + a semaphore (`FETCH_CONCURRENCY`, default 5).
- Content is truncated to `MAX_PAGE_CHARS` (default 10000).
- trafilatura produces clean markdown from server-rendered pages. JS-heavy pages may yield poor extraction (Playwright fallback is on the roadmap).

---

## Memory Strategy

- **Storage:** Chroma persisted to `./advanced_memory/`. Collection `research_memory`.
- **Writes:** `fetch_node` writes each fetched page's text (split into chunks) with metadata `{url, title, query, timestamp}`.
- **Reads:** `memory_node` queries Chroma with the original user query (+ aspects/gaps) for broad context before synthesis.
- **Recency boost:** `1 / (1 + age_hours / 24)` applied to similarity scores.
- **Persistence:** `add_to_memory()` calls `vectorstore.persist()` after each batch.

> **Caveat:** Because `fetch` writes to Chroma *before* `memory` reads, the memory node may re-retrieve the just-fetched content. This is currently acceptable but means memory acts more as a per-run store than a persistent cross-run knowledge base.

---

## Recent Improvements (quick-win cluster)

Implemented to optimize the existing pipeline without adding nodes:

1. **Parallel fetch** — `fetch_node` now fetches pages concurrently with a semaphore instead of one-at-a-time (`FETCH_CONCURRENCY`, default 5). Large speedup on multi-page rounds.
2. **trafilatura extraction** — `fetch_url` switched from `requests`+`BeautifulSoup` to `trafilatura` for cleaner, higher-quality markdown (better fact extraction downstream).
3. **Structured fact extraction** — `analyze_node` uses Pydantic `with_structured_output(AnalyzeOutput)` so each fact carries a `claim` + `source_url`, making facts citable and enabling future deduplication. Includes a plain-text fallback if structured parsing fails.
4. **Larger analysis context** — analysis snippet per page raised from a hard-coded 1500 chars to `ANALYSIS_SNIPPET_CHARS` (default 4000), so more of each page informs extracted facts.
5. **Lower analysis temperature** — analysis runs at `ANALYSIS_TEMPERATURE` (default 0.1) for more precise, less speculative facts.
6. **Date-filter bug fix** — removed the invalid `site:news` token injection that broke recency-filtered searches.

---

## Current State

### Implemented

- **Monolithic pipeline** (current stable): plan → search → fetch → analyze → should_continue → memory → synthesize. End-to-end with all local models.
- **Parallel trafilatura fetch** (see Recent Improvements).
- **Structured fact extraction** with source URLs (see Recent Improvements).
- **Chroma memory persistence** across runs.
- **LangSmith tracing**: optional, via env vars.

### Known Limitations / Not Yet Done

- **`plan_node`** still uses fragile YAML parsing instead of Pydantic structured output.
- **No `s.jina.ai`/`brave` fallback** for search.
- **No cross-source fact deduplication** — redundant facts may accumulate across pages.
- **`should_continue`** is heuristic (counts + gap presence), not an LLM coverage assessment. Next-round queries are naive concatenations (`f"{query} {gap}"`).
- **Memory self-retrieval** (see Memory Strategy caveat).
- **No Playwright JS fallback** for pages trafilatura can't extract.

### Future Roadmap (from original design)

1. **Orchestrator + sub-agent architecture** — replace the monolithic pipeline with parallel `Send` fan-out sub-agents. (This is the big architectural rewrite; not started in code.)
2. **Cross-encoder reranking** — two-stage ranking after embedding rerank in `search`.
3. **Human-in-the-loop checkpoints** — pause after plan and after fetch for approve/reject/revise.
4. **Source citation verification** — map each fact to its source URL (partially enabled by structured facts) and HEAD-check cited URLs.
5. **Markdown report export** — `.md` with frontmatter + references.
6. **Gradio web UI** — browser interface with streaming.
7. **Embedding fallback to CPU** — avoid OOM under tight memory.
8. **Playwright JS fallback** for trafilatura misses.
9. **Jina/Brave search fallback** and **Pydantic plan output**.
