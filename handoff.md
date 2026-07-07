# Handoff Document — lite-deep-research-agent

## Project Overview

**What it does:** A LangGraph-based deep research agent powered by an orchestrator that decomposes user queries into independent sub-topics, spawns parallel sub-agents to research each sub-topic, iterates when coverage is insufficient, and synthesizes a grounded, sourced research report.

**Architecture:** Orchestrator + parallel sub-agents via LangGraph `Send` fan-out. The orchestrator decomposes, dispatches, aggregates, and decides when to stop. Each sub-agent runs a single-pass search→fetch→analyze pipeline scoped to one sub-topic. All sub-agents run concurrently. Facts accumulate across research rounds via state channel reducers.

**Target hardware:** Mac with Apple Silicon (M1/M2/M3) with 16GB+ unified memory. Runs entirely via Ollama.

**Core models:**
- LLM: `qwen3.5:4b` via Ollama (256K context via `num_ctx`, Q4_K_M quantization, ~3.5GB)
- Embeddings: `nomic-embed-text` via Ollama (768d, ~0.5GB)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph (`StateGraph`, `Send` fan-out, subgraph composition) |
| LLM | Ollama — `langchain_ollama.ChatOllama` |
| Embeddings | Ollama — `langchain_ollama.OllamaEmbeddings` |
| Vector DB | Chroma (persisted to `./advanced_memory/`) |
| Text splitting | `RecursiveCharacterTextSplitter` (chunk=1000, overlap=100) |
| Search | DuckDuckGo (`ddgs`) with `s.jina.ai` auto-fallback |
| HTML fetching | `trafilatura` (multi-extractor pipeline, clean markdown output) |
| Structured output | Pydantic models via `with_structured_output()` (json_mode) |
| Tracing | LangSmith (optional, via env vars) |
| Configuration | `.env` + `config.py` |

---

## Project Structure

```
lite-deep-research-agent/
├── HLD.md                         # High-Level Design (comprehensive)
├── handoff.md                     # This file
├── requirements.txt
├── .env.example
│
├── research_agent/
│   ├── __init__.py
│   ├── __main__.py                # Entry: --iterations, --verbose, query selection
│   ├── config.py                  # All env-read constants
│   ├── llm.py                     # create_llm(), create_embedder(), ResearchTools dataclass
│   ├── state.py                   # ResearchState TypedDict, SubTask type, reducers, helpers
│   ├── search.py                  # search_web() — multi-backend with DDG→Jina fallback
│   ├── fetch.py                   # trafilatura async wrapper, parallel fetch_pages()
│   ├── memory.py                  # Chroma add/query with recency boosting
│   ├── sub_agent.py               # Sub-agent subgraph + sub_search/fetch/analyze/memory nodes
│   ├── nodes.py                   # Orchestrator nodes: plan, aggregate, should_continue, memory, synthesize
│   ├── graph.py                   # create_research_graph() — composes parent + subgraph
│   ├── agent.py                   # AdvancedResearchAgent — async generator with streaming
│   └── cli.py                     # Async REPL with streaming output
│
└── scripts/
    ├── setup.sh                   # pip install, ollama pull
    └── serve.sh                   # Launch Ollama (optional helper)
```

---

## Architecture

### Graph Flow

```
plan → research_round → aggregate → should_continue
                                           │
                        ┌──────────────────┼──────────────────┐
                        ▼                                     ▼
                    "continue"                           "synthesize"
                        │                                     │
                    [re-enter                            memory → synthesize → END
                    research_round]
```

### Orchestrator Nodes (parent graph)

| Node | Role |
|---|---|
| `plan` | LLM decomposes user query into 3–5 `SubTask` objects (structured output). Sets `max_pages` per task dynamically (2–8). |
| `research_round` | Compiled subgraph. Fans out `sub_tasks` to N parallel `sub_agent` instances via `Send`. Merges outputs via `operator.add` reducers. |
| `aggregate` | Deduplicates facts (cosine similarity, threshold 0.85) and sources. Prepares `aggregated_facts` + `unique_sources`. |
| `should_continue` | LLM assesses coverage. If insufficient → generates new focused `sub_tasks` → `Command(goto="research_round")`. If sufficient → `Command(goto="memory")`. Safety cap at `MAX_ITERATIONS`. |
| `memory` | Global Chroma query with original user query. Merges with per-topic memory from sub-agents. |
| `synthesize` | LLM builds structured report (Executive Summary → Findings → Analysis → Conclusion → Notes) with inline source markers. |

### Sub-Agent Subgraph (one instance per sub-task)

```
sub_search → sub_fetch → sub_analyze → sub_memory → END
```

| Node | Role |
|---|---|
| `sub_search` | Calls `search_web()` (DDG with Jina fallback) for the sub-task query. Deduplicates + embedding-reranks. |
| `sub_fetch` | Parallel trafilatura fetch of up to `max_pages` pages. Async with `asyncio.gather` + semaphore. Truncates at `MAX_PAGE_CHARS`. |
| `sub_analyze` | LLM extracts grounded facts per page via structured output. Facts tagged `[Topic: X]`. |
| `sub_memory` | Per-topic Chroma query with sub-task query + focus. Writes fetched content with topic-enriched metadata. |

### Research Rounds & Iteration

- **Round 1:** Initial plan → sub-agents research → aggregate → should_continue checks coverage
- **Round 2+:** should_continue generates new sub-tasks targeting gaps → plan → sub-agents research → aggregate → check
- **Facts accumulate across rounds** via `operator.add` reducers. Old facts are carried forward, not discarded.
- **Safety:** Hard cap at `MAX_ITERATIONS` (default 5). If reached, forces synthesis with available facts.

---

## State Model (`ResearchState`)

| Key | Type | Reducer | Purpose |
|---|---|---|---|
| `query` | `str` | (overwrite) | Original user question |
| `sub_tasks` | `List[SubTask]` | (overwrite) | Plan node output; replanned each round |
| `search_results` | `Annotated[List[Dict], operator.add]` | **accumulate** | Search results from all sub-agents |
| `fetched_content` | `Annotated[List[Dict], operator.add]` | **accumulate** | Fetched page content from all sub-agents |
| `extracted_facts` | `Annotated[List[str], operator.add]` | **accumulate** | Facts from all sub-agents, tagged `[Topic: X]` |
| `sub_agent_errors` | `Annotated[List[str], operator.add]` | **accumulate** | Errors from sub-agents |
| `aggregated_facts` | `List[str]` | (overwrite) | Deduplicated facts after aggregation |
| `unique_sources` | `List[str]` | (overwrite) | Deduplicated source URLs |
| `per_topic_memory` | `List[Dict]` | (overwrite) | Memory from sub-agent per-topic queries |
| `global_memory` | `List[Dict]` | (overwrite) | Memory from central memory_node |
| `final_answer` | `str` | (overwrite) | Synthesized report |
| `sources` | `List[str]` | (overwrite) | Sources cited in report |
| `iteration` | `int` | (overwrite) | Current research round |
| `max_iterations` | `int` | (overwrite) | Safety cap (default 5) |
| `next_step` | `str` | (overwrite) | "continue" or "synthesize" |
| `coverage_sufficient` | `bool` | (overwrite) | LLM assessment |
| `plan_gaps` | `List[str]` | (overwrite) | Identified gaps for replanning |
| `messages` / `errors` | `List[str]` | (overwrite) | Human-readable log |

### `SubTask`

```python
class SubTask(TypedDict):
    task_id: str       # unique ID (uuid4)
    topic: str         # label, e.g. "Pricing", "Performance"
    query: str         # focused search query
    focus: str         # what facts to extract
    max_pages: int     # page budget (2–8, dynamically allocated by plan LLM)
```

---

## Ollama Setup

### Quick Start

```bash
# One-time setup
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

### Memory Budget (Mac Unified Memory)

| Component | Estimate |
|---|---|
| LLM weights (Q4_K_M, 4B) | ~3.5GB |
| KV cache (256K context, fp16) | ~2.5GB |
| Embeddings (`nomic-embed-text`) | ~0.5GB |
| System + overhead | ~2GB |
| **Total** | **~8.5GB** — fits in 16GB with headroom |

### Structured Output

Ollama supports structured output via `json_mode` / `function_calling`. All LLM calls that produce structured data use `with_structured_output(PydanticModel)`:

```python
class PlanOutput(BaseModel):
    sub_tasks: list[SubTaskModel]

result = await llm.with_structured_output(PlanOutput).ainvoke(prompt)
# → PlanOutput(sub_tasks=[...]) — typed, no parsing needed
```

No YAML parsing. No `/no_think` hack. No fallback logic.

---

## Search & Fetch Strategy

### Search: DuckDuckGo + s.jina.ai

```python
# config.py
SEARCH_BACKEND = "duckduckgo"  # or "jina", "brave"
SEARCH_FALLBACK = True          # auto-fallback to Jina when DDG returns < 3 results
```

DuckDuckGo is primary (free, zero setup). If it returns < 3 results or fails:
1. Auto-switch to `s.jina.ai` (free, 100 RPM, returns pre-extracted content)
2. If both fail → return partial, let should_continue replan

### Fetch: trafilatura

```python
text = trafilatura.extract(html, output_format="markdown", favor_precision=True)
```

trafilatura uses a multi-extractor pipeline (readability, justext, boilerpy3) to produce clean markdown. No JavaScript rendering — most research content (news, papers, docs) is server-rendered. If JS-heavy pages are needed, a Playwright fallback is on the roadmap.

---

## Memory Strategy

- **Storage:** Chroma persisted to `./advanced_memory/`. Collection `research_memory`.
- **Per-topic writes:** Each sub-agent writes fetched content with metadata `{url, title, query, topic, focus, timestamp}`.
- **Per-topic reads:** Each sub-agent queries Chroma with its sub-task query + focus before analyzing.
- **Global read:** Memory node queries Chroma with the original user query for broad context.
- **Recency boost:** `1 / (1 + age_hours / 24)` applied to similarity scores.
- **Persistence:** `add_to_memory()` calls `vectorstore.persist()` after each batch.

---

## Current State

### Implemented

- **Monolithic pipeline** (current stable): plan → search → fetch → analyze → should_continue → memory → synthesize. Working end-to-end with all local models.
- **Memory persistence**: Chroma survives restarts.
- **LangSmith tracing**: Optional, via env vars.

### In Progress (architectural rewrite)

- **Orchestrator + sub-agent architecture** — replacing the monolithic pipeline with parallel fan-out. See HLD.md for full design.
- **Ollama integration** — using Ollama for both LLM and embeddings on Mac.
- **trafilatura** — replacing requests+BeautifulSoup with trafilatura for better extraction.
- **Multi-backend search** — DuckDuckGo + s.jina.ai fallback, configurable.
- **Structured output** — replacing YAML parsing with Pydantic typed output models.

### Remaining Improvements (post-rewrite)

1. **Cross-encoder reranking** — Add `sentence-transformers.CrossEncoder` pass after embedding rerank in sub_search_node. Two-stage ranking for higher search precision.

2. **Human-in-the-loop checkpoints** — Pause after plan (approve decomposition) and after sub-agents (review sources). CLI prompts for approve/reject/revise.

3. **Source citation verification** — Map each fact back to its source URL. Post-synthesis HEAD check on cited URLs. Flag broken/hallucinated sources.

4. **Markdown report export** — Export reports as `.md` with YAML frontmatter, proper footnote-style citations, and a references section.

5. **Playwright JS fallback** — Optional trafilatura → Playwright fallback for pages that trafilatura can't extract cleanly.

6. **Gradio web UI** — Browser-based interface with streaming research progress and report display.

7. **Embedding fallback to CPU** — Under tight VRAM, fall back to CPU-based embedding to avoid OOM during bulk fetches.