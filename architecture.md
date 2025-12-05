# Lite Deep Research Agent Architecture

## Overview
- Implements a single advanced research agent built with LangGraph. The agent orchestrates planning, web search, content harvesting, analysis, memory retrieval, and synthesis to produce a sourced research report.
- The graph is linear by default (plan → search → fetch → analyze → memory → synthesize), but a conditional loop hook exists for iterative runs.
- Persistent vector memory is stored in `./advanced_memory` using Chroma with Ollama embeddings, enabling reuse of past fetches across sessions.

## State Model (`ResearchState`)
- **Input:** `query`
- **Planning:** `research_plan`, `search_queries`
- **Searching:** `search_results`, `fetched_content`
- **Analysis:** `extracted_facts`, `key_findings`
- **Memory:** `relevant_memory`
- **Output:** `final_answer`, `sources`
- **Control/Logging:** `iteration`, `max_iterations`, `errors`, `messages`

## Tooling (`ResearchTools`)
- **LLM:** Ollama `qwen2.5:14b-instruct-q4_K_M` via `Ollama`.
- **Embeddings:** `nomic-embed-text` via `OllamaEmbeddings`.
- **Vector store:** Chroma persisted to disk for long-term recall.
- **Text splitting:** `RecursiveCharacterTextSplitter` (chunk 1000/overlap 100) for memory ingestion.
- **Search:** DuckDuckGo text search (`DDGS`).
- **Fetching:** HTTP GET with BeautifulSoup cleaning; strips script/style/nav/footer/header and returns up to 5000 chars.
- **Memory helpers:** `add_to_memory` (chunks + metadata), `query_memory` (similarity search, top-k).

## Agent Nodes (graph steps)
- **plan_node:** Uses the LLM to draft a research plan, derive up to 4 search queries, and log planning status.
- **search_node:** Runs DuckDuckGo searches for each query, deduplicates by URL, and stores the top 10 combined results.
- **fetch_node:** Fetches the top 5 URLs, cleans HTML, saves successful pages into vector memory, and records fetched content.
- **analyze_node:** For each fetched page, prompts the LLM to extract concise facts/insights relevant to the original query.
- **memory_node:** Retrieves up to 5 similar chunks from persisted vector memory to supply historical context.
- **synthesize_node:** Prompts the LLM to write the final structured report (executive summary, findings, analysis, conclusion, notes) using the plan, extracted facts, and memory; collects source URLs.
- **should_continue (unused in current flow):** Optional conditional router for iterative research; would decide to loop back to search or proceed to synthesis based on iteration count and data sufficiency.

## Graph Wiring
- Constructed via `StateGraph(ResearchState)`.
- Entry point: `plan`.
- Edges: `plan → search → fetch → analyze → memory → synthesize → END`.
- Commented conditional edges are ready to enable iterative refinement when `should_continue` is activated.

## Agent Wrapper (`AdvancedResearchAgent`)
- Builds the graph with `create_research_graph()`.
- `research(query, max_iterations=1, verbose=True)` initializes state, runs the graph, and returns `{query, report, sources, plan, message_log}`. Prints progress when verbose.
- `visualize_graph()` optionally renders the compiled graph if `pygraphviz` is installed.

## CLI Flow (`main`)
- Provides a simple loop to run custom or example queries.
- Executes `agent.research(...)`, prints the report and source list, and saves results to `report_{hash(query)}.txt`.

