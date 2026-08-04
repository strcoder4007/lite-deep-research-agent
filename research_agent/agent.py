"""General-purpose local agent with a custom lightweight tool-calling loop.

No LangGraph: a simple step loop where the model either emits one JSON tool
call (```json {"tool": name, "args": {...}} ```) or a plain-text final answer.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from . import config, llm, logutil
from .tools import (
    TOOL_REGISTRY,
    ResearchTools,
    build_catalog,
    build_tools,
    get_tool,
    init_tools,
)
from .tools import finalize as _finalize


def _preview(result: Any) -> str:
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        return str(result)


def run(
    query: str,
    tools: Optional[ResearchTools] = None,
    max_steps: int = config.MAX_AGENT_STEPS,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the agent loop on ``query`` and return a result dict."""
    tools = tools or build_tools()
    init_tools(tools)

    messages = [
        SystemMessage(content=llm.build_system_prompt(build_catalog())),
        HumanMessage(content=query.strip()),
    ]
    steps: List[Dict[str, Any]] = []
    errors: List[str] = []
    answer = ""

    for step in range(1, max_steps + 1):
        started = time.perf_counter()
        text, call = llm.chat(messages, tools.llm)

        # Nudge retry once on empty/degenerate output (HLD §13.16 pattern).
        if not text.strip():
            messages.append(HumanMessage(content=config.AGENT_NUDGE))
            text, call = llm.chat(messages, tools.llm)
            if not text.strip():
                errors.append(f"step {step}: empty model output after nudge")
                break

        if call is None:
            # No tool call: treat the text as the final answer.
            answer = text
            steps.append({"step": step, "tool": None, "answer": text})
            if verbose:
                print(logutil.tool_step(step, "final(text)", text))
            break

        name = call["tool"]
        args = call["args"]
        entry = get_tool(name)
        if entry is None:
            result: Any = {"error": f"unknown tool: {name}"}
            errors.append(f"step {step}: unknown tool {name}")
        else:
            try:
                result = entry["fn"](**args)
            except TypeError as exc:
                # Bad args from the model: feed the error back as the result.
                result = {"error": f"bad arguments: {exc}"}
                errors.append(f"step {step}: {name} bad args: {exc}")
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
                errors.append(f"step {step}: {name} failed: {exc}")

        steps.append({"step": step, "tool": name, "args": args, "result": result})
        if verbose:
            elapsed = time.perf_counter() - started
            print(
                logutil.tool_step(step, name, _preview(result))
                + logutil.dim(f"  {elapsed:.1f}s")
            )

        messages.append(AIMessage(content=text))
        messages.append(
            HumanMessage(content=f"Tool result ({name}):\n{_preview(result)}")
        )

        if name == "final_answer":
            answer = str(args.get("answer", result))
            break
        sentinel = _finalize.take_final_answer()
        if sentinel is not None:
            answer = sentinel
            break
    else:
        errors.append(f"stopped after {max_steps} steps without a final answer")

    if not answer and steps:
        # Fall back to the last model text if the loop never finalized.
        answer = str(steps[-1].get("answer") or steps[-1].get("result") or "")

    return {"query": query, "answer": answer, "steps": steps, "errors": errors}


__all__ = ["run", "TOOL_REGISTRY"]
