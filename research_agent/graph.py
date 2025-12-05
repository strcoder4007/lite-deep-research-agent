from functools import partial

from langgraph.graph import END, StateGraph

from .nodes import (
    analyze_node,
    fetch_node,
    memory_node,
    plan_node,
    search_node,
    should_continue_node,
    synthesize_node,
)
from .state import ResearchState
from .tools import ResearchTools, build_tools


def create_research_graph(tools: ResearchTools):
    graph = StateGraph(ResearchState)
    graph.add_node("plan", partial(plan_node, tools=tools))
    graph.add_node("search", partial(search_node, tools=tools))
    graph.add_node("fetch", partial(fetch_node, tools=tools))
    graph.add_node("analyze", partial(analyze_node, tools=tools))
    graph.add_node("should_continue", should_continue_node)
    graph.add_node("memory", partial(memory_node, tools=tools))
    graph.add_node("synthesize", partial(synthesize_node, tools=tools))

    graph.set_entry_point("plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "fetch")
    graph.add_edge("fetch", "analyze")
    graph.add_edge("analyze", "should_continue")
    graph.add_conditional_edges(
        "should_continue",
        lambda state: state.get("next_step", "synthesize"),
        {"search": "search", "synthesize": "memory"},
    )
    graph.add_edge("memory", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def get_graph():
    """Factory used by LangGraph deployments to build the compiled graph."""
    tools = build_tools()
    return create_research_graph(tools)
