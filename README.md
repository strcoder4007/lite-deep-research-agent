# Lite Deep Research Agent

Local web research agent built with LangGraph. It plans, searches, fetches, extracts facts, recalls prior memory, and synthesizes a grounded report while staying within a 16 GB VRAM budget.

> This README is the source of truth for architecture, implementation progress, and usage.

## Setup
- Install dependencies: `pip install -r requirements.txt` (includes `langsmith` + `python-dotenv` for tracing).
- Edit `.env` to set model names/limits; defaults target a 16 GB VRAM box. The CLI auto-loads `.env` on startup.
- Set LangSmith env vars in `.env` if you want tracing: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=<key>`, optional `LANGSMITH_PROJECT=lite-deep-research-agent` and `LANGSMITH_WORKSPACE_ID=<id>`.
- Run an Ollama server with small models available (defaults: `qwen3:8b-q4_K_M` for LLM, `mxbai-embed-large` for embeddings). Override with `LLM_MODEL`/`EMBED_MODEL` env vars.
- Vector memory persists to `./advanced_memory`; keep this directory to reuse historical context.

## Run
```bash
python -m research_agent  # interactive REPL
# or
python -m research_agent.cli
```
Reports are saved to `report_<hash>.txt` and include sources.

### Tracing (LangSmith)
- With `.env` populated and `LANGSMITH_TRACING=true`, runs automatically emit LangGraph/LangChain traces (per-node spans plus DuckDuckGo search/fetch spans).
- The agent sets a stable `thread_id` per query and attaches the query as metadata for grouping in LangSmith.
- If traces don’t appear, confirm env vars are loaded: `python - <<'PY'\nimport os; print(os.getenv('LANGSMITH_API_KEY'), os.getenv('LANGSMITH_TRACING'))\nPY`.

## LangGraph application structure
The repo follows the layout described in the LangGraph docs: a package that holds all graph code, a dependency file, a `langgraph.json`, and an optional `.env`.
```
lite-deep-research-agent/
├── research_agent/
│   ├── __init__.py
│   ├── __main__.py
│   ├── agent.py
│   ├── cli.py
│   ├── config.py
│   ├── graph.py
│   ├── memory.py
│   ├── nodes.py
│   ├── state.py
│   └── tools.py
├── .env
├── langgraph.json
├── requirements.txt
└── README.md
```
- Graph factory: `research_agent/graph.py:get_graph` (compiled `StateGraph` referenced in `langgraph.json`).
- Nodes/utilities: `research_agent/nodes.py`, `research_agent/memory.py`, `research_agent/tools.py`, `research_agent/state.py`.
- Agent wrapper/CLI: `research_agent/agent.py`, `research_agent/cli.py`.
- Deployment config: `langgraph.json` (lists dependencies, graphs, env file), `.env` for runtime settings.

## Purpose and behavior
- LangGraph agent that turns a user query into a grounded, sourced research report.
- Pipeline: plan → search/rerank → fetch/ingest → analyze → memory recall → synthesize, with iteration via `should_continue`.
- Long-term vector memory (`./advanced_memory`) is reused across runs to add historical context.

## Hardware/model profile (16 GB VRAM)
- LLM: prefer a 7B quantized instruct model (e.g., `qwen3:8b-q4_K_M` or `llama3.1:8b-instruct-q4_K_M`); keep `concurrency=1` and modest max tokens.
- Embeddings: small-footprint model such as `mxbai-embed-large` or `bge-small`; avoid large embedding models alongside the LLM.
- Batching: keep chunk batches small during ingestion to limit VRAM spikes; fall back to CPU embeddings if needed.

## High-level data flow
1) Receive query; generate a structured research plan with multiple search queries.
2) Run web searches, rerank for relevance, and deduplicate.
3) Fetch and clean the best pages; chunk and persist to vector memory with metadata.
4) Extract key facts from fetched pages with the LLM.
5) Retrieve similar prior chunks from memory to enrich context.
6) Synthesize a grounded report using only extracted facts + retrieved memory.

## Architecture diagram
```
 User Query
     |
     v
  plan_node ----> search_node ----> fetch_node ----> analyze_node ----> memory_node ----> synthesize_node
     |                |                 |                  |                |                   |
     |                |                 |                  |                |                   v
     |                |                 |                  |                |             Final Report
     |                |                 |                  |                v
     |                |                 |                  |         Retrieved Memory
     |                |                 |                  v
     |                |                 |         Extracted Facts
     |                |                 v
     |                |         Cleaned Page Content
     |                v
     v         Search Results (ranked/deduped)
  Research Plan (structured)
```

## State model (`ResearchState`)
- Input: `query`; Planning: `research_plan`, `search_queries`; Searching: `search_results`, `fetched_content`.
- Analysis: `extracted_facts`, `key_findings`; Memory: `relevant_memory`; Output: `final_answer`, `sources`.
- Control/logging: `iteration`, `max_iterations`, `errors`, `messages`, `plan_gaps`, `next_step`.

## Tooling (`ResearchTools`)
- LLM: lighter instruct model for planning, extraction, and synthesis (low temperature).
- Embeddings: small model; keep batch sizes modest.
- Vector store: Chroma persisted to disk (`./advanced_memory`) with metadata `{url, title, query, timestamp}`.
- Text splitting: `RecursiveCharacterTextSplitter` chunk 1000 / overlap 100.
- Search/fetch: DuckDuckGo (`DDGS`) plus `requests` + `BeautifulSoup` cleanup; cap text length around 5000 chars.
- Memory helpers: `add_to_memory` (chunks + metadata) and `query_memory` (top-k similarity).

## Agent nodes (current/expected behavior)
- `plan_node`: prompt for structured YAML/JSON with SEARCH_QUERIES, KEY_ASPECTS, GAPS_TO_ADDRESS; expect 4–5 queries; fallback to raw user query if parsing fails.
- `search_node`: run searches per query (5–8 results), dedupe by URL/host, rerank against the main query with embeddings, keep top N (~10).
- `fetch_node`: fetch top URLs (aim 8–10), clean HTML, discard short pages, chunk and store to vector memory with metadata.
- `analyze_node`: extract concise, source-prefixed facts per page relevant to the original query.
- `memory_node`: query memory with user query (and plan aspects) for top-k (e.g., 5); enforce similarity threshold; retrieved memory informs synthesis and follow-up passes.
- `synthesize_node`: low temperature; grounded to `extracted_facts` + `relevant_memory`; structure output (Executive Summary, Numbered Findings, Analysis, Conclusion, Notes) with inline source markers.
- `should_continue`: loop when coverage is thin (e.g., fetched < 3, facts < 5, or gaps remain); stop when `iteration >= max_iterations` or facts are sufficient; expand queries around gaps.

## Graph wiring
- Entry: `plan`; edges: `plan → search → fetch → analyze → memory → synthesize → END`.
- Conditional: from `should_continue` route `{ "search": "search", "synthesize": "memory" }` (used after `analyze`).

## Grounding and answer quality
- Synthesis must cite from `extracted_facts` or `relevant_memory`; avoid injecting new claims.
- Inline markers (e.g., `[Source 1]`) map to URLs captured in fetched/memory sources.
- Keep temperatures low for synthesis; modest for planning/extraction to reduce variance.

## Memory strategy
- Store richer metadata `{url, title, query, timestamp}` and persist to reuse across runs.
- Apply similarity threshold and slight recency boost; drop low-signal recalls.
- Feed retrieved memory into planning/search prompts to reduce redundant searches.

## Planning output format
```
SEARCH_QUERIES:
  - ...
KEY_ASPECTS:
  - ...
GAPS_TO_ADDRESS:
  - ...
```
- Validate structure; if missing, default to the user query as the only search query and log the fallback in `messages`.

## Configuration checklist
- Use lighter LLM/embedding models noted above; `concurrency=1`, modest max tokens.
- Increase search breadth with dedupe + embedding rerank; fetch 8–10 pages when available.
- Chunk + store with metadata; enforce similarity threshold and recency boost on retrieval.
- Keep synthesis grounded with low temperature and structured output plus inline source markers.
- Keep `should_continue` enabled to loop when coverage is thin or gaps remain.

## Future improvements
- Human-in-the-loop checkpoints for query refinement, source approval, and draft review.
- Improve DuckDuckGo integrations (news timelimits, richer metadata, better dedupe).
- Use a cross-encoder reranker to re-score top candidates before truncation.

## Agent wrapper (`AdvancedResearchAgent`)
- Builds the graph with `create_research_graph()`.
- `research(query, max_iterations, verbose=True)` returns `{query, report, sources, plan, message_log, errors}` with console progress when verbose.
- `visualize_graph()` renders the compiled graph if `pygraphviz` is installed.

## CLI flow (`main`)
- Simple REPL: pick custom or example queries, run `agent.research(...)`, print the report and source list, and save to `report_{hash(query)}.txt`.
