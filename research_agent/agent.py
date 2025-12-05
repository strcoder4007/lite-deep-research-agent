from __future__ import annotations

import hashlib
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
        append_message(initial_state, f"Starting research for: {query}")
        latest_state = dict(initial_state)
        for event in self.graph.stream(initial_state):
            for node, data in event.items():
                latest_state = {**latest_state, **data}
                if verbose:
                    print(f"[{node}]")
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
