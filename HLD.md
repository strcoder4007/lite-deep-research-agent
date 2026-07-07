# High-Level Design — lite-deep-research-agent

## 1. System Overview

**lite-deep-research-agent** is an orchestrator-driven research system built on LangGraph. It decomposes a user query into independent sub-topics, spawns parallel sub-agents to research each sub-topic, aggregates results across rounds, and synthesizes a single grounded, sourced report.

The orchestrator uses **LLM-driven decomposition and iteration control**. Sub-agents are **single-pass research pipelines** (search → fetch → analyze → per-topic memory) that run concurrently via LangGraph's `Send` fan-out mechanism. Facts accumulate across research rounds using state channel reducers.

### Design Goals

| Goal | How |
|---|---|
| **Lightweight** | 4B model via vLLM, 137MB embeddings, zero cloud dependencies |
| **Parallel by default** | LangGraph `Send` fan-out spawns N sub-agents concurrently |
| **LLM-driven iteration** | Orchestrator decides when coverage is sufficient; generates follow-up sub-tasks |
| **Grounded output** | Every fact traced to a source URL; per-topic memory enriches context |
| **Single GPU** | vLLM serves both LLM and embeddings under 16GB VRAM total |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR GRAPH                        │
│                                                                  │
│  ┌─────────┐   ┌───────────────────┐   ┌────────────┐           │
│  │  plan   │──▶│  research_round   │──▶│  aggregate  │           │
│  │ (decomp)│   │  (subgraph, N     │   │ (dedup,     │           │
│  │         │   │   parallel sub-   │   │  merge,     │           │
│  │         │   │   agents via Send)│   │  gap check) │           │
│  └─────────┘   └───────────────────┘   └─────┬──────┘           │
│                                              │                   │
│                                     ┌────────▼──────────┐       │
│                                     │ should_continue    │       │
│                                     │ LLM decides:       │       │
│                                     │ coverage ok?       │       │
│                                     └──┬────────────┬───┘       │
│                                  "yes" │            │ "no"      │
│                                        ▼            ▼           │
│                              ┌──────────┐  ┌─────────────────┐  │
│                              │  memory  │  │ generate new     │  │
│                              │ (global) │  │ sub_tasks +      │  │
│                              └────┬─────┘  │ goto plan        │  │
│                                   ▼         └─────────────────┘  │
│                              ┌──────────┐                        │
│                              │synthesize│                        │
│                              └────┬─────┘                        │
│                                   ▼                              │
│                                  END                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────┐
│     research_round SUBGRAPH         │
│                                     │
│  dispatch ──► [Send fan-out to      │
│                N sub_agent          │
│                instances]           │
│                                     │
│  sub_agent SUBGRAPH (N instances):  │
│  ┌──────────────────────────────┐   │
│  │ sub_search → sub_fetch →     │   │
│  │ sub_analyze → sub_memory     │   │
│  │ (per-topic)                  │   │
│  └──────────────────────────────┘   │
│                                     │
│  All instances complete → fan-in    │
│  Reducers (operator.add) merge:     │
│    search_results, fetched_content, │
│    extracted_facts                  │
└─────────────────────────────────────┘
```

---

## 3. State Design

```python
from typing import Annotated, Any, Dict, List, Optional, TypedDict
import operator

class SubTask(TypedDict):
    """A self-contained research task for a sub-agent."""
    task_id: str
    topic: str         # sub-topic label (e.g., "Pricing", "Performance")
    query: str          # focused search query
    focus: str          # what facts to extract
    max_pages: int      # dynamic page budget (2–8)

class ResearchState(TypedDict, total=False):
    # ── Input ──
    query: str                              # original user question

    # ── Decomposition ──
    sub_tasks: List[SubTask]                 # plan_node outputs; replan rewrites

    # ── Sub-agent outputs (accumulated via operator.add reducers) ──
    search_results: Annotated[List[Dict[str, Any]], operator.add]
    fetched_content: Annotated[List[Dict[str, Any]], operator.add]
    extracted_facts: Annotated[List[str], operator.add]
    sub_agent_errors: Annotated[List[str], operator.add]

    # ── Aggregation ──
    aggregated_facts: List[str]              # deduplicated fact list
    unique_sources: List[str]                # deduplicated URLs
    coverage_sufficient: bool                # LLM assessment
    plan_gaps: List[str]                     # what's still missing

    # ── Memory ──
    per_topic_memory: List[Dict[str, Any]]   # from sub-agent per-topic queries
    global_memory: List[Dict[str, Any]]      # from central memory_node

    # ── Synthesis ──
    final_answer: str
    sources: List[str]

    # ── Control ──
    iteration: int
    max_iterations: int                      # safety cap (default 5)
    next_step: str                           # "continue" | "synthesize"

    # ── Observability ──
    messages: List[str]
    errors: List[str]
```

### Reducer Semantics

Fields annotated with `Annotated[List[...], operator.add]` accumulate across parallel branches. When N sub-agents complete, each one's `extracted_facts` list is concatenated via `operator.add`, producing a unified list for the orchestrator.

Fields without a reducer (like `final_answer`, `next_step`, `aggregated_facts`) are overwritten by the last node that writes them. The orchestrator nodes after the fan-in barrier own these.

---

## 4. Component Design

### 4.1 LLM: qwen3.5:4b via vLLM

| Property | Value |
|---|---|
| Model | `cyankiwi/Qwen3.5-4B-AWQ-4bit` |
| Context window | 256K tokens |
| VRAM | ~2.5GB (AWQ-4bit) |
| Server | vLLM OpenAI-compatible API (`localhost:8000`) |
| Structured output | `guided_json` via vLLM's native support |
| LangChain client | `langchain_openai.ChatOpenAI(base_url=...)` |

**Concurrency model:** vLLM uses continuous batching. When N sub-agents make concurrent LLM calls (planning, fact extraction), vLLM batches them into a single forward pass. This is the primary reason to use vLLM over Ollama — Ollama queues requests, vLLM parallelizes them.

### 4.2 Embeddings: nomic-embed-text via vLLM

| Property | Value |
|---|---|
| Model | `nomic-ai/nomic-embed-text-v1.5` |
| Dimensions | 768 |
| VRAM | ~275MB |
| Server | Separate vLLM instance (`localhost:8001`) with `--task embedding` |
| LangChain client | `langchain_openai.OpenAIEmbeddings(base_url=...)` |

Both servers run on the same GPU. vLLM's PagedAttention memory management allows them to coexist under the 16GB budget.

### 4.3 Search: DuckDuckGo + s.jina.ai

```python
SEARCH_BACKEND = "duckduckgo"  # "duckduckgo" | "jina" | "brave"
SEARCH_FALLBACK = True         # DDG → Jina auto-fallback when DDG returns < 3 results
```

| Backend | Protocol | Results | Cost |
|---|---|---|---|
| DuckDuckGo | `ddgs` library scraping | 5–8 results, title+snippet | Free |
| s.jina.ai | `GET https://s.jina.ai/search?q=...` | 5 results with extracted content | Free, 100 RPM |
| Brave Search | `GET https://api.search.brave.com/...` | 10 results, LLM context endpoint | $5/mo free credit |

**Fallback logic:**
1. Try configured `SEARCH_BACKEND`
2. If < 3 results OR exception → try `s.jina.ai` (if `SEARCH_FALLBACK=1`)
3. If still insufficient → return partial results, let `should_continue` replan

### 4.4 Fetch: trafilatura

Replaces `requests` + `BeautifulSoup`. trafilatura uses a multi-extractor pipeline (readability, justext, boilerpy3) and outputs clean text or markdown.

```python
import trafilatura

async def fetch_page(url: str, timeout: int = 15) -> Optional[Tuple[str, str]]:
    downloaded = trafilatura.fetch_url(url, timeout=timeout)
    if not downloaded:
        return None
    text = trafilatura.extract(
        downloaded,
        output_format="markdown",
        favor_precision=True,       # prefer accuracy over quantity
        include_links=False,        # strip raw links
        include_images=False,
        include_tables=False,
    )
    if not text or len(text) < 100:
        return None
    text = text[:MAX_PAGE_CHARS]
    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title.strip() if metadata and metadata.title else url
    return title, text
```

**Parallel fetching within a sub-agent:** `asyncio.gather()` with a semaphore limiting concurrency (default 5). Since trafilatura is CPU-bound (not I/O), `run_in_executor` wraps the sync `trafilatura.fetch_url`.

**Why not crawl4ai:** crawl4ai adds ~500MB Chromium install and 300-500MB RAM overhead. For a lightweight agent targeting 16GB VRAM, trafilatura is sufficient. JavaScript-heavy pages that trafilatura can't handle are rare for research content (news articles, docs, papers are typically server-rendered).

### 4.5 Memory: Chroma

Unchanged from current architecture. Persisted to `./advanced_memory/`. Collection `research_memory`.

**Per-topic memory (sub_memory_node in each sub-agent):**
- Queries Chroma with the sub-task query + focus
- Returns up to 5 chunks with recency boost
- Writes newly fetched content to Chroma with metadata `{url, title, query, topic, focus, timestamp}`

**Global memory (orchestrator's memory_node):**
- Queries Chroma with the original user query
- Provides broader context for synthesis
- Uses the same recency boost formula: `1 / (1 + age_hours / 24)`

---

## 5. Graph Design

### 5.1 Parent Graph Nodes

#### `plan_node` (async)

```
Input:  state.query
Action: LLM with structured output decomposes query into sub-topics
Output: state.sub_tasks (3–5 SubTask objects with dynamic max_pages)
```

Pydantic model:
```python
class PlanOutput(BaseModel):
    sub_tasks: list[SubTaskModel]
```

Prompt template:
```
Decompose this research question into 3–5 independent sub-topics.
For each sub-topic, provide:
- topic: short label (1-3 words)
- query: focused web search query
- focus: what specific facts/information to extract from each page
- max_pages: how many sources to consult (2–8, based on complexity/importance)
Cover diverse angles: factual, analytical, comparative, practical.
```

#### `research_round` (compiled subgraph)

A compiled subgraph added as a parent node. Internally fans out `sub_tasks` to N parallel `sub_agent` instances via `Send`. Each sub-agent runs `sub_search → sub_fetch → sub_analyze → sub_memory`. Outputs are merged via `operator.add` reducers.

**Dispatch logic:**
```python
def fan_out(state: ResearchState) -> List[Send]:
    return [
        Send("sub_agent", {
            "query": st["query"],
            "topic": st["topic"],
            "focus": st["focus"],
            "max_pages": st["max_pages"],
        })
        for st in state["sub_tasks"]
    ]
```

#### `aggregate_node` (sync, async wrapper)

- Deduplicates `extracted_facts` using cosine similarity (threshold 0.85) against `nomic-embed-text` embeddings
- Deduplicates source URLs
- Prepares `aggregated_facts` and `unique_sources`
- Assesses rough coverage (fact count per topic) but delegates final assessment to `should_continue`

#### `should_continue_node` (async)

```
Input:  aggregated_facts, unique_sources, plan_gaps, iteration
Action: LLM with structured output assesses coverage
Output: coverage_sufficient + new_sub_tasks OR goto synthesize
```

Pydantic model:
```python
class CoverageAssessment(BaseModel):
    coverage_sufficient: bool
    reasoning: str
    new_sub_tasks: Optional[list[SubTaskModel]]

class SubTaskModel(BaseModel):
    topic: str
    query: str
    focus: str
    max_pages: int
```

Routing via `Command`:
```python
if assessment.coverage_sufficient or state["iteration"] >= state["max_iterations"]:
    return Command(goto="memory", update={"next_step": "synthesize"})
else:
    return Command(goto="plan", update={
        "sub_tasks": assessment.new_sub_tasks,
        "iteration": state["iteration"] + 1,
        "next_step": "continue",
    })
```

**Safety:** Hard cap at `MAX_ITERATIONS` (default 5) prevents unbounded loops from small models.

#### `memory_node` (async)

Central memory query using original `state.query`. Merges with per-topic memory from sub-agents (already in `per_topic_memory`). Returns combined `global_memory`.

#### `synthesize_node` (async)

Builds a structured report prompt from:
- `aggregated_facts` (tagged with `[Topic: X]` prefixes)
- `global_memory`
- `unique_sources`

Report structure: Executive Summary → Numbered Findings → Analysis → Conclusion → Notes.

Sources are listed with inline markers `[1]`, `[2]` pointing to a reference section.

### 5.2 Sub-Agent Subgraph

```
sub_search → sub_fetch → sub_analyze → sub_memory → END
```

All nodes are async. The sub-agent uses `state.query` (sub-task query, set by `Send`) and `state.topic` for scoping.

#### `sub_search_node`

- Calls `search_web(state.query, max_results=config.SEARCH_PER_QUERY)` via configured backend
- Deduplicates by URL + hostname
- Reranks using embedding cosine similarity + recency bonus
- Writes to `search_results` (operator.add accumulator)

#### `sub_fetch_node`

- Fetches up to `state.max_pages` results via trafilatura
- Parallelizes with `asyncio.gather` + semaphore
- Truncates to `MAX_PAGE_CHARS` (default 10000)
- Writes to `fetched_content` (operator.add accumulator)

#### `sub_analyze_node`

- For each fetched page, calls LLM with structured output
- Extracts grounded facts tagged with `[Topic: {topic}]` prefix
- Pydantic model: `{facts: list[str], source_title: str, source_url: str}`
- Writes to `extracted_facts` (operator.add accumulator)

#### `sub_memory_node`

- Query Chroma with sub-task query + focus (per-topic retrieval)
- Write fetched content to Chroma with topic-enriched metadata
- Returns per-topic memory chunks → `per_topic_memory`

### 5.3 Graph Wiring (graph.py)

```python
def create_research_graph(tools: ResearchTools):
    graph = StateGraph(ResearchState)

    # Parent nodes
    graph.add_node("plan", partial(plan_node, tools=tools))

    # Research round: subgraph that internally fans out
    research_round = create_research_round_graph(tools)
    graph.add_node("research_round", research_round)

    graph.add_node("aggregate", aggregate_node)
    graph.add_node("should_continue", partial(should_continue_node, tools=tools))
    graph.add_node("memory", partial(memory_node, tools=tools))
    graph.add_node("synthesize", partial(synthesize_node, tools=tools))

    # Entry
    graph.set_entry_point("plan")

    # Plan → research_round via Send fan-out (inside research_round subgraph)
    graph.add_conditional_edges("plan", dispatch_to_round, ["research_round"])

    # research_round → aggregate
    graph.add_edge("research_round", "aggregate")
    graph.add_edge("aggregate", "should_continue")

    # should_continue routes via Command (no static edges needed)
    graph.add_node("memory", partial(memory_node, tools=tools))
    graph.add_edge("memory", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def create_research_round_graph(tools: ResearchTools):
    """Subgraph that fans out sub_tasks to parallel sub_agent instances."""
    sub_agent = create_sub_agent_graph(tools)

    graph = StateGraph(ResearchState)
    graph.add_node("dispatch", dispatch_identity)  # pass-through for fan-out
    graph.add_node("sub_agent", sub_agent)

    graph.set_entry_point("dispatch")
    graph.add_conditional_edges("dispatch", fan_out_to_sub_agents, ["sub_agent"])
    graph.add_edge("sub_agent", END)

    return graph.compile()


def create_sub_agent_graph(tools: ResearchTools):
    """Single-pass sub-agent: search → fetch → analyze → per-topic memory."""
    graph = StateGraph(ResearchState)
    graph.add_node("sub_search", partial(sub_search_node, tools=tools))
    graph.add_node("sub_fetch", partial(sub_fetch_node, tools=tools))
    graph.add_node("sub_analyze", partial(sub_analyze_node, tools=tools))
    graph.add_node("sub_memory", partial(sub_memory_node, tools=tools))

    graph.set_entry_point("sub_search")
    graph.add_edge("sub_search", "sub_fetch")
    graph.add_edge("sub_fetch", "sub_analyze")
    graph.add_edge("sub_analyze", "sub_memory")
    graph.add_edge("sub_memory", END)

    return graph.compile()
```

---

## 6. Data Flow (Request Lifecycle)

### Initial Request
```
User query
  → plan_node: decompose into 3–5 SubTasks
  → research_round: fan out via Send
    → [SubAgent 1] DDG search → trafilatura fetch(5 pages) → LLM extract facts → memory store/query
    → [SubAgent 2] DDG search → trafilatura fetch(4 pages) → LLM extract facts → memory store/query
    → [SubAgent 3] DDG search → trafilatura fetch(6 pages) → LLM extract facts → memory store/query
  → reducers merge: 45 search results, 15 fetched pages, 60 extracted facts
  → aggregate_node: dedup → 12 unique pages, 48 unique facts
  → should_continue: "Coverage insufficient on pricing and benchmarks"
  → Command(goto="plan", sub_tasks=[SubTask("pricing"), SubTask("benchmarks")])

### Round 2
  → plan_node: receives new sub_tasks (carried forward from should_continue)
  → research_round: fan out again
    → [SubAgent 4] focused pricing search → fetch → analyze → memory
    → [SubAgent 5] focused benchmark search → fetch → analyze → memory
  → reducers merge again (facts accumulate across rounds!)
  → aggregate_node: now 80 facts across 20 pages
  → should_continue: "Coverage sufficient"
  → Command(goto="memory")

### Final (after fan-in)
  → memory_node: global query + merge per-topic memory
  → synthesize_node: LLM generates structured report with inline citations
  → END
```

---

## 7. vLLM Integration

### Setup

```bash
# One-time install
pip install vllm

# Pull models
huggingface-cli download cyankiwi/Qwen3.5-4B-AWQ-4bit
huggingface-cli download nomic-ai/nomic-embed-text-v1.5
```

### Server Launch

```bash
# LLM server (port 8000)
vllm serve cyankiwi/Qwen3.5-4B-AWQ-4bit \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.80 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# Embeddings server (port 8001)
vllm serve nomic-ai/nomic-embed-text-v1.5 \
    --port 8001 \
    --task embedding \
    --gpu-memory-utilization 0.15
```

**VRAM budget** (16GB GPU):
- LLM (`qwen3.5:4b` AWQ-4bit): ~2.5GB
- Embeddings (`nomic-embed-text`): ~1GB
- KV cache overhead: ~4GB
- **Total: ~7.5GB** — fits under 16GB with generous headroom

### CLI Flags

```bash
# Auto-launch vLLM servers as subprocesses
python -m research_agent --start-vllm

# Connect to already-running servers
python -m research_agent --llm-url http://localhost:8000/v1 --embed-url http://localhost:8001/v1
```

The `--start-vllm` flag launches both vLLM servers as subprocesses, waits for them to be healthy (health-check loop on `/v1/models`), runs the agent, and tears down the servers on exit. This is optional — the default mode expects vLLM to already be running.

### Client Code

```python
# research_agent/llm.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        base_url=config.LLM_BASE_URL,      # http://localhost:8000/v1
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        api_key="not-needed",              # vLLM doesn't require auth locally
    )

def create_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBED_MODEL_NAME,
        base_url=config.EMBED_BASE_URL,    # http://localhost:8001/v1
        api_key="not-needed",
    )
```

### Structured Output with vLLM

vLLM supports `guided_json` natively. LangChain's `ChatOpenAI.with_structured_output()` sends a `response_format` with JSON Schema, which vLLM handles via its `--enable-auto-tool-choice` flag:

```python
class PlanOutput(BaseModel):
    sub_tasks: list[SubTaskModel]

result = await llm.with_structured_output(PlanOutput).ainvoke(prompt)
# → PlanOutput(sub_tasks=[...])  # typed, no parsing needed
```

---

## 8. Configuration

### Environment Variables

```bash
# ── vLLM ──
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL_NAME=cyankiwi/Qwen3.5-4B-AWQ-4bit
EMBED_BASE_URL=http://localhost:8001/v1
EMBED_MODEL_NAME=nomic-ai/nomic-embed-text-v1.5
VLLM_START_SERVERS=0              # 1 = auto-launch vLLM subprocesses

# ── LLM params ──
LLM_TEMPERATURE=0.25
LLM_MAX_TOKENS=2048
SYNTH_MAX_TOKENS=4096

# ── Search ──
SEARCH_BACKEND=duckduckgo         # duckduckgo | jina | brave
SEARCH_FALLBACK=1                 # DDG → jina auto-fallback
SEARCH_PER_QUERY=8                # results per sub-agent search
SEARCH_RECENCY_BOOST=0.05
SEARCH_HOST_DEDUP=1

# ── Fetch ──
MAX_PAGE_CHARS=10000              # chars per fetched page
FETCH_TIMEOUT=15                  # seconds per page
FETCH_CONCURRENCY=5               # parallel fetches per sub-agent

# ── Sub-agents ──
MAX_SUB_TASKS=5                   # max sub-tasks per plan
DEFAULT_PAGES_PER_TASK=5          # fallback when LLM doesn't set max_pages
MIN_PAGES_PER_TASK=2
MAX_PAGES_PER_TASK=8

# ── Iteration ──
MAX_ITERATIONS=5                  # safety cap for research rounds
MIN_FACTS_FOR_COVERAGE=20         # rough threshold

# ── Memory ──
MEMORY_DIR=advanced_memory
MEMORY_TOP_K=5
MEMORY_SIMILARITY_THRESHOLD=0.35
MEMORY_MIN_CHARS=200
CHUNK_SIZE=1000
CHUNK_OVERLAP=100

# ── Dedup ──
FACT_DEDUP_THRESHOLD=0.85         # cosine similarity for duplicate facts

# ── Tracing ──
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=lite-deep-research
```

### config.py Constants

All values read from env vars with sensible defaults. The `config.py` module is the single source of truth — no constants hardcoded in node implementations.

---

## 9. File Map

```
lite-deep-research-agent/
├── HLD.md                         # This document
├── handoff.md                     # Updated project handoff
├── requirements.txt               # Updated dependencies
├── .env.example                   # Template for configuration
│
├── research_agent/
│   ├── __init__.py
│   ├── __main__.py                # Entry: handles --start-vllm, --llm-url etc.
│   ├── config.py                  # All env+constants
│   ├── llm.py                     # create_llm(), create_embedder(), ResearchTools
│   ├── state.py                   # ResearchState, SubTask, reducers, helpers
│   ├── search.py                  # search_web() — multi-backend + DDG→Jina fallback
│   ├── fetch.py                   # trafilatura async wrapper, fetch_pages()
│   ├── memory.py                  # Chroma add/query with recency boost
│   ├── sub_agent.py               # Sub-agent subgraph + 4 nodes
│   ├── nodes.py                   # Orchestrator nodes: plan, aggregate, should_continue, memory, synthesize
│   ├── graph.py                   # create_research_graph() — composes all subgraphs
│   ├── agent.py                   # AdvancedResearchAgent — async research() with streaming
│   └── cli.py                     # Async CLI with REPL + streaming output
│
└── scripts/
    ├── setup.sh                   # One-time: pip install, vllm pull models
    └── serve.sh                   # Launch both vLLM servers
```

---

## 10. Streaming & Observability

### Node-by-node streaming

The `AdvancedResearchAgent.research()` method is an `async generator` that yields `StreamEvent` objects per node:

```python
async for event in agent.research(query):
    print(f"[{event.node}] {event.summary}")
    # → [plan] Decomposed into 4 sub-tasks
    # → [sub_agent] Researching "Pricing": 6 pages, 12 facts
    # → [sub_agent] Researching "Performance": 5 pages, 8 facts
    # → [aggregate] 20 unique facts, 3 gaps identified
```

### LangSmith

When `LANGCHAIN_TRACING_V2=true`, all nodes, sub-graphs, and LLM calls are traced. Each sub-agent's internal nodes appear as nested traces under the `research_round` span.

---

## 11. Error Handling

| Failure | Strategy |
|---|---|
| DDG search fails | Auto-fallback to s.jina.ai |
| s.jina.ai also fails | Return empty results, let should_continue replan |
| trafilatura fetch timeout | Skip page, log error, continue with remaining pages |
| LLM structured output parse failure | Retry once with temperature=0; if still fails, skip that LLM call |
| vLLM server unreachable | Exit with clear error: "vLLM not running. Run with --start-vllm or start servers manually." |
| All sub-agents return 0 facts | should_continue replans with more focused queries |
| Max iterations reached | Force synthesize with available facts, note gaps in report |
| Chroma persistence error | Log warning, continue with in-memory operation |

---

## 12. Future Roadmap

| Priority | Feature |
|---|---|
| P1 | Human-in-the-loop checkpoints (approve plan, review sources) |
| P1 | Cross-encoder reranking for search results |
| P2 | Gradio web UI |
| P2 | Markdown report export with DOI-like citations |
| P3 | Playwright-based fallback for JS pages trafilatura misses |
| P3 | Source URL accessibility verification (HEAD requests post-synthesis) |