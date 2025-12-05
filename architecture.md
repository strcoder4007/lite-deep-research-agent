# Lite Deep Research Agent Architecture

## Purpose and Behavior
- Single LangGraph-based agent that turns a user query into a sourced research report.
- Orchestrates planning, web search, content fetch, fact extraction, memory recall, and synthesis with streaming-friendly LangGraph nodes.
- Maintains long-term vector memory (`./advanced_memory`) so past crawls can inform future answers.

## High-Level Data Flow
1) Receive query and generate a research plan + search queries.
2) Run web searches and deduplicate URLs.
3) Fetch and clean the top pages, chunk them, and persist chunks to vector memory.
4) Extract key facts from each fetched page with the LLM.
5) Pull similar prior chunks from memory for extra context.
6) Synthesize a structured report with sources.

## Architecture Diagram
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
     v         Search Results
  Research Plan
```

## State Model (`ResearchState`)
- **Input:** `query`
- **Planning:** `research_plan`, `search_queries`
- **Searching:** `search_results`, `fetched_content`
- **Analysis:** `extracted_facts`, `key_findings`
- **Memory:** `relevant_memory`
- **Output:** `final_answer`, `sources`
- **Control/Logging:** `iteration`, `max_iterations`, `errors`, `messages`

## Tooling (`ResearchTools`)
- **LLM:** Ollama `qwen2.5:14b-instruct-q4_K_M` via `Ollama` for planning, extraction, synthesis.
- **Embeddings:** `nomic-embed-text` via `OllamaEmbeddings`.
- **Vector store:** Chroma persisted to disk for long-term recall (`./advanced_memory`).
- **Text splitting:** `RecursiveCharacterTextSplitter` (chunk 1000 / overlap 100) before storage.
- **Search:** DuckDuckGo text search (`DDGS`).
- **Fetching:** `requests` + `BeautifulSoup`; strips script/style/nav/footer/header; returns up to 5000 chars per page.
- **Memory helpers:** `add_to_memory` (chunks + metadata) and `query_memory` (top-k similarity).

## Agent Nodes (graph steps)
- **plan_node:** LLM builds a plan and 3–4 search queries; logs planning status.
- **search_node:** Runs searches for each query, dedupes by URL, keeps the top 10 combined results.
- **fetch_node:** Fetches the top 5 URLs, cleans HTML, stores successful pages into vector memory, records fetched content.
- **analyze_node:** LLM extracts concise facts/insights relevant to the original query from each fetched page.
- **memory_node:** Retrieves up to 5 similar chunks from vector memory to supply historical context.
- **synthesize_node:** LLM writes the final structured report (exec summary, numbered findings, analysis, conclusion, notes) using the plan, extracted facts, and memory; collects source URLs.
- **should_continue (currently unused):** Optional conditional router to loop back to search when more data is needed or iteration limits allow.

## Graph Wiring
- Constructed via `StateGraph(ResearchState)`.
- Entry point: `plan`.
- Edges: `plan → search → fetch → analyze → memory → synthesize → END`.
- Commented conditional edges are ready to enable iterative refinement when `should_continue` is activated.

## Agent Wrapper (`AdvancedResearchAgent`)
- Builds the graph with `create_research_graph()`.
- `research(query, max_iterations=1, verbose=True)` initializes state, runs the graph, and returns `{query, report, sources, plan, message_log}` with console progress when verbose.
- `visualize_graph()` can render the compiled graph if `pygraphviz` is installed.

## CLI Flow (`main`)
- Simple REPL: pick custom or example queries, run `agent.research(...)`, print the report and source list, and save to `report_{hash(query)}.txt`.

## Configuration Tips
- Swap models by adjusting `ResearchTools.llm` and `ResearchTools.embeddings`.
- Increase `max_iterations` and enable `should_continue` (in the commented conditional edges) for iterative enrichment.
- Tune `chunk_size`, `chunk_overlap`, and Chroma parameters to balance recall vs. storage.
