# Handoff Document — lite-deep-research-agent

## Project Overview

**What it does:** A LangGraph-based deep research agent that runs entirely locally (under 16GB VRAM). Given a user query, it plans search queries, executes web searches, fetches and analyzes pages, retrieves relevant memory from a vector store, and synthesizes a grounded, sourced research report.

**Target hardware:** Single GPU with ~16GB VRAM (e.g., RTX 4070, RTX 3090, or a Mac with unified memory).

**Core models (defaults):**
- LLM: `qwen3:8b-q4_K_M` via Ollama
- Embeddings: `mxbai-embed-large` via Ollama

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph (`StateGraph`) |
| LLM | Ollama — `ChatOllama` (LangChain) |
| Embeddings | Ollama — `OllamaEmbeddings` (LangChain) |
| Vector DB | Chroma (persisted to `./advanced_memory/`) |
| Text splitting | `RecursiveCharacterTextSplitter` (chunk=1000, overlap=100) |
| Search | DuckDuckGo (`ddgs` / `duckduckgo_search`) |
| HTML fetching | `requests` + `BeautifulSoup` |
| Tracing | LangSmith (optional, via env vars) |
| Configuration | `.env` + `config.py` constants |

---

## Project Structure

```
lite-deep-research-agent/
├── research_agent/
│   ├── __init__.py          # Package marker
│   ├── __main__.py          # Entry point (runs CLI)
│   ├── agent.py            # AdvancedResearchAgent wrapper + visualize_graph()
│   ├── cli.py              # REPL CLI logic
│   ├── config.py           # All tunable constants (VRAM-profile defaults)
│   ├── graph.py            # StateGraph factory: creates/compiles the research graph
│   ├── memory.py           # Chroma add/query helpers with recency boosting
│   ├── nodes.py            # All 7 LangGraph nodes + should_continue
│   ├── state.py            # ResearchState TypedDict + append_message/append_error helpers
│   └── tools.py            # ResearchTools dataclass + LLM/embedder/vectorstore factories + search/fetch utilities
├── .env                    # Model names, limits, memory dir, search settings
├── langgraph.json          # LangGraph deployment manifest (references get_graph factory)
├── requirements.txt
└── README.md
```

---

## How It Works — Detailed Architecture

### Pipeline Stages

```
plan → search → fetch → analyze → should_continue → memory → synthesize → END
                                        ↑                                     
                                        └── (loops back to search if coverage insufficient)
```

### State Model (`ResearchState`)

A `TypedDict` carrying these keys through the pipeline:

| Key | Type | Purpose |
|---|---|---|
| `query` | `str` | Original user question |
| `research_plan` | `Dict` | YAML-parsed plan: `SEARCH_QUERIES`, `KEY_ASPECTS`, `GAPS_TO_ADDRESS` |
| `search_queries` | `List[str]` | Queries to execute (can grow on loop iteration) |
| `search_results` | `List[Dict]` | Deduplicated + embedding-reranked DuckDuckGo results |
| `fetched_content` | `List[Dict]` | `{url, title, text, metadata}` for each fetched page |
| `extracted_facts` | `List[str]` | Bullet facts per page, prefixed with `[Source Title]` |
| `relevant_memory` | `List[Dict]` | Retrieved chunks from Chroma with score + metadata |
| `final_answer` | `str` | Synthesized report |
| `sources` | `List[str]` | Unique URLs used in the report |
| `iteration` / `max_iterations` | `int` | Loop control |
| `messages` / `errors` | `List[str]` | Human-readable log |
| `plan_gaps` | `List[str]` | Gaps identified during planning |
| `next_step` | `str` | Routing decision from `should_continue` |

### Node Behaviors

**`plan_node`**: Sends the user query + date to the LLM with a YAML-output system prompt. Parses `SEARCH_QUERIES`, `KEY_ASPECTS`, `GAPS_TO_ADDRESS`. Falls back to raw query if parsing fails. Appends a message to the state log.

**`search_node`**: For each query in `search_queries`, calls `run_ddg_search()` (5–8 results). Collects all results, deduplicates by URL and hostname (host-level dedup via config flag), then reranks by cosine similarity of title+snippet embeddings against the main query, plus a recency bonus for pages with a `published_at` date. Returns top-10.

**`fetch_node`**: Iterates the top-N results (config: 15), calls `fetch_url()` to get cleaned text (strips `<script>`, `<style>`, nav, footer, etc., truncates at 5000 chars). For each page, calls `add_to_memory()` to chunk and persist into Chroma with metadata `{url, title, query, timestamp}`. Returns the fetched list.

**`analyze_node`**: For each fetched page, sends a `/no_think` prompt (qwen3 instruction-truncation trick) asking the LLM to extract only grounded, source-prefixed bullet facts relevant to the query. Appends any parse/analysis errors to the state.

**`should_continue_node`**: Decides next_step based on:
- If `iteration >= max_iterations` → `synthesize`
- If `fetched < 3` OR `facts < 5` OR `plan_gaps` non-empty → `search` (appends gap-based queries, increments iteration)
- Otherwise → `synthesize`

**`memory_node`**: Queries Chroma with the user query + plan aspects, enforcing a similarity threshold (default 0.35) and applying a recency boost (older chunks score lower). Returns up to `MEMORY_TOP_K` (default 5) chunks.

**`synthesize_node`**: Builds a prompt from `extracted_facts` + `relevant_memory` + `sources`. Instructs the LLM to produce a structured report (Executive Summary, Numbered Findings, Analysis, Conclusion, Notes) with inline source markers. Returns `final_answer` and `sources`.

### Graph Wiring

```
plan → search → fetch → analyze → should_continue
                                           │
                      ┌────────────────────┴────────────────────┐
                      ↓                                         ↓
                   "search"                                   "synthesize"
                   (loop)                                       ↓
                 memory ← synthesize → END
```

### Memory Strategy

- **Storage:** Chroma persisted to `./advanced_memory/`. Collection `research_memory`.
- **Metadata:** `{url, title, query, timestamp}` — enables recency boosting and source tracking.
- **Retrieval:** Similarity search with extended query (query + aspects), threshold 0.35, recency boost `1/(1 + age_hours/24)`.
- **Persistence:** `add_to_memory()` calls `vectorstore.persist()` after each batch.

---

## Current State / What's Working

- **Fully functional end-to-end pipeline**: `python -m research_agent` launches an interactive REPL.
- **Local-only execution**: All models run via Ollama; no external API calls (except to DuckDuckGo).
- **Memory persistence**: `./advanced_memory/` survives restarts; prior context informs new queries.
- **Conditional loop**: `should_continue` routes back to search when coverage is insufficient.
- **LangSmith tracing**: Works when env vars are set (can be disabled by omitting them).
- **Config tunability**: All model names, chunk sizes, thresholds, iteration limits are in `config.py` and `.env`.
- **Report output**: Saved to `report_<hash>.txt` with sources listed.

### Known Gaps / Rough Edges

1. **No human-in-the-loop**: No checkpoints for query refinement, source approval, or draft review.
2. **DuckDuckGo scraping**: Relies on HTML scraping; breaks on JavaScript-heavy pages.
3. **No cross-encoder reranking**: Uses embedding cosine similarity for reranking; a cross-encoder would be more accurate.
4. **No streaming output**: The agent returns the full report; no incremental streaming.
5. **Fallback plan parsing**: If YAML parsing fails, it falls back to raw query (acceptable but inelegant).
6. **`/no_think` hack**: The `_with_no_think()` mechanism to strip think tags from qwen3 is a fragile convention.

---

## Improvements

### 1. Add Cross-Encoder Reranking
**What to do:** After the embedding-based cosine rerank in `search_node`, add a cross-encoder pass using `sentence-transformers.CrossEncoder` to rescore the top-20 candidates. Replace the cosine similarity reranking in `_rerank_results` with a cross-encoder that scores `(query, title + snippet)` pairs, then take the top-N.

Install: `pip install sentence-transformers`. Download a lightweight cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) in `config.py` or `.env`. In `search_node`, after deduping: use the cross-encoder to score the top-20 candidates, then take top-10.

**Why it matters:** Cross-encoders jointly encode the query-document pair and produce much more accurate relevance scores than separate embedding + cosine. This directly improves the quality of sources fed to the synthesizer. For Staff Engineer growth: demonstrates understanding of information retrieval internals, two-stage ranking pipelines, and how to integrate new model types into an existing LangGraph agent.

**Files likely to touch:** `research_agent/nodes.py` (`_rerank_results`), `research_agent/tools.py` (add cross-encoder factory), `research_agent/config.py` (new env vars).

**Verification:** Run the agent on a known query; check that the top results are more relevant. Compare the reranked list before/after. Run unit tests on `_rerank_results`.

---

### 2. Add Streaming Output for Real-Time Progress
**What to do:** Modify `AdvancedResearchAgent.research()` to yield intermediate state snapshots via a generator pattern, and update `cli.py` to display incremental results. In `graph.py`, change `stream()` to `stream()` (already a stream, but surface it as a generator). Add a `--stream` flag to the CLI. Show node-by-node progress as each node completes.

The `graph.stream()` already yields events per node — expose this through `AdvancedResearchAgent.research()` as a generator, and update `cli.py` to print node completion messages as they arrive rather than waiting for the full run.

**Why it matters:** For research tasks that take 2–5 minutes, showing no output until the end is a poor UX. Streaming intermediate results (facts extracted so far, pages fetched) builds trust and lets users abort early if the direction is wrong. Staff Engineer context: designing streaming/chunked APIs, understanding async event emission, UX-oriented system design.

**Files likely to touch:** `research_agent/agent.py` (add `@generator` pattern), `research_agent/cli.py` (streaming display), `research_agent/graph.py` (document streaming behavior).

**Verification:** Run the CLI and observe node-by-node output appearing in real time.

---

### 3. Implement Human-in-the-Loop Checkpoints
**What to do:** Add two new node types to the graph: `review_plan_node` (after `plan_node`) and `review_sources_node` (after `fetch_node` / before `analyze_node`). Both pause the graph and prompt the user via the CLI to approve/revise the plan or select which sources to include.

Implement a `human_review()` function in `cli.py` that prompts in the terminal. Insert these as edges in `graph.py`:

```
plan → review_plan → search (if approved)
fetch → review_sources → analyze (if approved)
```

Add a `reviewed` boolean to `ResearchState`. If the user rejects at any checkpoint, route to `synthesize` with a "review incomplete" note.

**Why it matters:** Production research agents need human oversight to catch incorrect search directions early. This is a core pattern in AI engineering workflows (RLHF, human feedback loops). Staff Engineer context: designing fault-tolerant pipelines with human override points, state machine design.

**Files likely to touch:** `research_agent/graph.py` (new edges/nodes), `research_agent/nodes.py` (new checkpoint nodes), `research_agent/cli.py` (human review prompts), `research_agent/state.py` (new fields).

**Verification:** Run the agent; observe a pause after planning asking for approval. Reject a plan and verify the agent handles the rejection gracefully.

---

### 4. Switch to Structured Output for Plan Parsing
**What to do:** Replace the YAML parsing in `plan_node` with an LLM `with_structured_output` call using a Pydantic model (`ResearchPlan` with `search_queries: list[str]`, `key_aspects: list[str]`, `gaps_to_address: list[str]`). This removes the `yaml` dependency for plan parsing, eliminates the fallback-to-raw-query logic, and makes plan parsing robust even when the LLM adds commentary.

In `nodes.py`, define `class SearchQuery(TypedDict)`, `class ResearchPlan(TypedDict)` and use `tools.llm.with_structured_output(ResearchPlan)` in `plan_node`. Keep the YAML approach as a fallback only for debugging.

**Why it matters:** Structured output is the modern, reliable way to get typed data from LLMs. YAML parsing with fallback logic is fragile and adds unnecessary complexity. Staff Engineer context: LLM API design, structured output patterns, reducing unnecessary dependencies.

**Files likely to touch:** `research_agent/nodes.py` (`plan_node`), `research_agent/state.py` (possibly new type hints), `research_agent/config.py`.

**Verification:** Run 20 queries of varying complexity; check that plan parsing failures drop to near-zero. Compare the structure quality of generated plans.

---

### 5. Add Source Citation Tracking and Verification
**What to do:** Extend `ResearchState` to include a `source_map: Dict[str, Dict]` that maps each fact back to its source URL and title. In `analyze_node`, extract not just the fact but also the source metadata and build this map. In `synthesize_node`, verify that every inline citation exists in `source_map`. Add a post-synthesis validation step that logs any citation without a corresponding source.

Also add a `verify_sources` function in `tools.py` that checks, for each source URL, whether the page is still accessible (lightweight HEAD request, catch failures). Run this after `fetch_node` and flag inaccessible sources.

**Why it matters:** Grounded reports require verified citations. Broken links or hallucinated sources undermine the agent's credibility. Staff Engineer context: building reliability into AI systems, citation verification, trust-but-verify patterns.

**Files likely to touch:** `research_agent/state.py` (new `source_map` field), `research_agent/nodes.py` (`analyze_node`, `synthesize_node`), `research_agent/tools.py` (new `verify_sources` function).

**Verification:** Run the agent and check the report's citations against the actual sources. Verify that no hallucinated sources appear.

---

### 6. Add Markdown Report Export with DOI/Semantic Citations
**What to do:** In `synthesize_node`, after generating the report, convert inline `[Source N]` markers to semantic citations using a standard format (e.g., footnote-style `[^1]` with a references section). Also export the report as a proper `.md` file with YAML front matter containing metadata (query, date, sources, model used). Update the report saving logic in `agent.py` to write `report_<hash>.md` instead of `.txt`.

Add a `report_format` config option (`"plain"` / `"markdown"`). Default to `"markdown"`.

**Why it matters:** Reports shared as plain `.txt` files are hard to read and lack proper citation formatting. Markdown with semantic citations is the standard for research outputs and can be rendered in any viewer. Staff Engineer context: output format design, document generation, standard formats.

**Files likely to touch:** `research_agent/nodes.py` (`synthesize_node`), `research_agent/agent.py` (report saving), `research_agent/config.py`.

**Verification:** Open the generated `.md` report in a Markdown viewer; confirm citations render correctly.

---

### 7. Implement Smart Query Expansion in should_continue
**What to do:** Currently `should_continue_node` expands queries by concatenating `base_query + gap`. This is simplistic. Improve it by sending the current state (query, extracted facts, plan gaps) to the LLM with a prompt: "Based on what we've found so far, generate 2–3 focused search queries that would fill the remaining gaps. Return as a JSON list." Parse and add these queries to `search_queries`.

This replaces the naive `f"{base_query} {gap}"` concatenation with semantically-driven query generation.

**Why it matters:** Query expansion is the core of iterative research. Naive concatenation produces poor queries; LLM-driven expansion produces targeted follow-up searches. Staff Engineer context: iterative refinement loops, query understanding, research pipeline optimization.

**Files likely to touch:** `research_agent/nodes.py` (`should_continue_node`), `research_agent/tools.py` (add query expansion helper).

**Verification:** Run the agent on a complex multi-aspect query; observe that follow-up queries are specific and diverse, not just the original query concatenated with gap names.

---

### 8. Add Playwright-Based Fetching for JavaScript Pages
**What to do:** Add an optional fetch mode in `tools.py` that uses Playwright (already used in the browser-agent project) to render JavaScript-heavy pages that BeautifulSoup cannot handle. Add a `USE_PLAYWRIGHT_FETCH` config flag.

Implement `fetch_url_playwright(url)` that launches a headless Chromium page, waits for network idle, extracts the rendered HTML, and passes it to the existing cleanup pipeline. Fall back to the existing `fetch_url` if Playwright is unavailable or times out.

**Why it matters:** Many modern sites (news, forums, SPAs) render content client-side. The current requests+BeautifulSoup approach misses this content entirely. Staff Engineer context: hybrid fetching strategies, browser automation integration, fallback design patterns.

**Files likely to touch:** `research_agent/tools.py` (new `fetch_url_playwright`), `research_agent/config.py` (new `USE_PLAYWRIGHT_FETCH` flag), `requirements.txt` (add `playwright`).

**Verification:** Run the agent against a JS-heavy site (e.g., a React-based blog); confirm the fetched content includes text that would be missing with requests-only fetching.

---

### 9. Add Embedding Model Fallback to CPU
**What to do:** In `build_tools()` in `tools.py`, wrap embedder creation in a try/except. If VRAM is constrained and embedding fails, fall back to a CPU-based embedding model (e.g., `nomic-embed-text` or `bge-small`). Add a `EMBED_ON_CPU` env var to force CPU embedding.

Also add a `embed_batch_size` config that caps how many texts are embedded at once, to avoid VRAM spikes during bulk fetching.

**Why it matters:** Under tight VRAM conditions, the embedder can cause OOM during the fetch phase when many chunks are being stored simultaneously. CPU fallback keeps the agent running. Staff Engineer context: resource management, graceful degradation, memory-aware pipeline design.

**Files likely to touch:** `research_agent/tools.py` (`build_tools`, `create_embedder`), `research_agent/config.py` (new env vars).

**Verification:** Run with `EMBED_ON_CPU=1` and observe that embedding happens on CPU without crashing.

---

### 10. Add Web UI (Gradio) for the Research Agent
**What to do:** Add a Gradio web interface to the project, similar to the `browser-agent` project's `webui.py`. The UI should have a query input, a "Run Research" button, a streaming log output area, and a final report display panel. Support starting/stopping the research task.

Create `research_agent/webui.py` with Gradio components. Add a `run_webui()` function in `cli.py` to launch it.

**Why it matters:** A CLI-only tool limits adoption. Non-technical users want a web UI. This brings the project into parity with browser-agent's UX. Staff Engineer context: web UI development, Gradio integration, multi-interface architecture.

**Files likely to touch:** New `research_agent/webui.py`, updated `research_agent/cli.py`, new entry in `requirements.txt` (`gradio`).

**Verification:** Launch the web UI; run a research query through the browser; observe streaming output and final report.