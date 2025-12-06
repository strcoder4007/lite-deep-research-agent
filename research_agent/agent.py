from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional

from . import config
from .graph import create_research_graph
from .state import append_message
from .tools import ResearchTools, build_tools


class AdvancedResearchAgent:
    def __init__(self, tools: Optional[ResearchTools] = None):
        self.tools = tools or build_tools()
        self.graph = create_research_graph(self.tools)

    def research(
        self, query: str, max_iterations: int = config.MAX_ITERATIONS, verbose: bool = True
    ) -> Dict[str, Any]:
        query = query.strip()
        initial_state: Dict[str, Any] = {
            "query": query,
            "iteration": 0,
            "max_iterations": max_iterations,
            "messages": [],
            "errors": [],
            "search_queries": [],
        }
        run_id = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
        append_message(initial_state, f"Starting research for: {query}")
        latest_state = dict(initial_state)
        last_time = time.perf_counter()

        def _log_node(name: str, payload: Dict[str, Any], elapsed: float) -> None:
            duration_ms = elapsed
            summary_parts = []
            if payload.get("search_results"):
                summary_parts.append(f"results={len(payload['search_results'])}")
            if payload.get("fetched_content"):
                summary_parts.append(f"fetched={len(payload['fetched_content'])}")
            if payload.get("extracted_facts"):
                summary_parts.append(f"facts={len(payload['extracted_facts'])}")
            if payload.get("relevant_memory"):
                summary_parts.append(f"memory={len(payload['relevant_memory'])}")
            if payload.get("sources"):
                summary_parts.append(f"sources={len(payload['sources'])}")
            if payload.get("errors"):
                summary_parts.append(f"errors={len(payload['errors'])}")
            summary = " | ".join(summary_parts) if summary_parts else ""
            print(f"[{name:<15}] {duration_ms:7.1f} seconds" + (f" | {summary}" if summary else ""))

        stream_config = {
            "configurable": {"thread_id": run_id},
            "metadata": {"query": query},
        }

        for event in self.graph.stream(initial_state, config=stream_config):
            for node, data in event.items():
                latest_state = {**latest_state, **data}
                if verbose:
                    now = time.perf_counter()
                    _log_node(node, data, now - last_time)
                    last_time = now
        final_state = latest_state
        return {
            "query": query,
            "report": final_state.get("final_answer", ""),
            "sources": final_state.get("sources", []),
            "plan": final_state.get("research_plan", {}),
            "message_log": final_state.get("messages", []),
            "errors": final_state.get("errors", []),
        }

    def visualize_graph(self, output_path: str = "graph.png") -> Optional[Path]:
        try:
            graph = self.graph.get_graph()
            data = graph.draw_png()
        except Exception:
            return None
        path = Path(output_path)
        path.write_bytes(data)
        return path
