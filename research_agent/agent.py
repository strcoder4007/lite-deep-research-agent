"""General-purpose local agent with a custom lightweight tool-calling loop.

No LangGraph: a simple step loop where the model either emits one or more JSON
tool calls (```json [{"tool": name, "args": {...}}, ...] ```) or a plain-text
final answer.  Multiple tool calls in a single turn are executed in parallel.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from . import config, llm, logutil
from .memory import ConversationMemory
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


def _execute_tool(
    name: str, args: Dict[str, Any], step: int
) -> tuple[str, Dict[str, Any], List[str], float]:
    """Execute a single tool call and return (name, result, errors, elapsed)."""
    started = time.perf_counter()
    entry = get_tool(name)
    if entry is None:
        return name, {"error": f"unknown tool: {name}"}, [f"step {step}: unknown tool {name}"], 0.0
    try:
        result = entry["fn"](**args)
        return name, result, [], time.perf_counter() - started
    except TypeError as exc:
        return name, {"error": f"bad arguments: {exc}"}, [f"step {step}: {name} bad args: {exc}"], time.perf_counter() - started
    except Exception as exc:
        return name, {"error": f"{type(exc).__name__}: {exc}"}, [f"step {step}: {name} failed: {exc}"], time.perf_counter() - started


def run(
    query: str,
    tools: Optional[ResearchTools] = None,
    max_steps: int = config.MAX_AGENT_STEPS,
    verbose: bool = True,
    history: Optional[ConversationMemory] = None,
) -> Dict[str, Any]:
    """Run the agent loop on ``query`` and return a result dict.

    ``history`` is an optional ``ConversationMemory`` whose previous
    turns are prepended to the message list so the model sees context.
    """
    tools = tools or build_tools()
    init_tools(tools)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    log_path = log_dir / f"run_{digest}.log"
    logutil.set_log_file(str(log_path))

    logutil._print(logutil.header(f"Query: {query}"))

    messages: List[Any] = [
        SystemMessage(content=llm.build_system_prompt(build_catalog())),
    ]
    if history is not None:
        messages.extend(history.messages())
    messages.append(HumanMessage(content=query.strip()))

    steps: List[Dict[str, Any]] = []
    errors: List[str] = []
    answer = ""
    total_tokens = 0
    total_prompt_tokens = 0
    total_time = 0.0
    context_total = config.LLM_NUM_CTX

    for step in range(1, max_steps + 1):
        started = time.perf_counter()
        text, calls, info = llm.chat(messages, tools.llm)
        total_time += info["elapsed"]
        total_tokens += info["tokens"]
        total_prompt_tokens += info["prompt_tokens"]

        # Nudge retry once on empty/degenerate output (HLD §13.16 pattern).
        if not text.strip() and not calls:
            messages.append(HumanMessage(content=config.AGENT_NUDGE))
            text, nudge_calls, nudge_info = llm.chat(messages, tools.llm)
            total_time += nudge_info["elapsed"]
            total_tokens += nudge_info["tokens"]
            total_prompt_tokens += nudge_info["prompt_tokens"]
            if not text.strip() and not nudge_calls:
                errors.append(f"step {step}: empty model output after nudge")
                break
            calls = nudge_calls

        if not calls:
            # No tool call: treat the text as the final answer.
            answer = text
            steps.append({"step": step, "tool": None, "answer": text})
            if verbose:
                logutil._print(logutil.tool_step(step, "final(text)", text))
            break

        # Execute all tool calls in parallel.
        step_errors: List[str] = []
        step_results: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(len(calls), config.FETCH_CONCURRENCY)) as executor:
            futures = {
                executor.submit(_execute_tool, call["tool"], call["args"], step): call
                for call in calls
            }
            for future in futures:
                name, result, call_errors, elapsed = future.result()
                step_errors.extend(call_errors)
                step_results[name] = result
                if verbose:
                    preview = _preview(result)
                    logutil._print(
                        logutil.tool_step(step, name, preview)
                        + logutil.dim(f"  {elapsed:.1f}s")
                    )
                    logutil._print(logutil.tool_result(result))

        errors.extend(step_errors)

        # Add all results to messages and check for termination.
        has_final = False
        for call in calls:
            name = call["tool"]
            args = call["args"]
            result = step_results.get(name, {"error": "missing result"})
            steps.append({"step": step, "tool": name, "args": args, "result": result})

            messages.append(AIMessage(content=text))
            messages.append(
                HumanMessage(content=f"Tool result ({name}):\n{_preview(result)}")
            )

            if name == "final_answer":
                answer = str(args.get("answer", result))
                has_final = True
            sentinel = _finalize.take_final_answer()
            if sentinel is not None:
                answer = sentinel
                has_final = True

        if has_final:
            break

    else:
        errors.append(f"stopped after {max_steps} steps without a final answer")

    if not answer and steps:
        # Fall back to the last model text if the loop never finalized.
        answer = str(steps[-1].get("answer") or steps[-1].get("result") or "")

    context_used = total_prompt_tokens
    logutil._print(logutil.run_summary(
            total_steps=len(steps),
            total_time=total_time,
            total_tokens=total_tokens,
            prompt_tokens=total_prompt_tokens,
            context_used=context_used,
            context_total=context_total,
            errors=errors,
        )
    )
    logutil._print(logutil.agent(f"Answer: {answer}"))
    logutil._print(logutil.separator())
    logutil.close_log()

    if history is not None:
        history.add(query, answer)

    return {"query": query, "answer": answer, "steps": steps, "errors": errors}


__all__ = ["run", "TOOL_REGISTRY"]
