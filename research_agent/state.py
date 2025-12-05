from typing import Any, Dict, List, Optional, TypedDict


class ResearchState(TypedDict, total=False):
    query: str
    research_plan: Dict[str, Any]
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    fetched_content: List[Dict[str, Any]]
    extracted_facts: List[str]
    key_findings: List[str]
    relevant_memory: List[Dict[str, Any]]
    final_answer: str
    sources: List[str]
    iteration: int
    max_iterations: int
    errors: List[str]
    messages: List[str]
    plan_gaps: List[str]
    next_step: str


Message = Dict[str, Any]


def append_message(state: ResearchState, text: str) -> None:
    messages = list(state.get("messages", []))
    messages.append(text)
    state["messages"] = messages


def append_error(state: ResearchState, text: str) -> None:
    errors = list(state.get("errors", []))
    errors.append(text)
    state["errors"] = errors
