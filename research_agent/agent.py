"""General-purpose local agent with a custom lightweight tool-calling loop.

No LangGraph: a simple step loop where the model either emits one or more JSON
tool calls (```json [{"tool": name, "args": {...}}, ...] ```) or a plain-text
final answer.  Multiple tool calls in a single turn are executed in parallel.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
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


def _for_message(result: Any) -> str:
    """Render a tool result for the message history, truncating page text.

    Fetched pages can be MAX_PAGE_CHARS long; appending several verbatim
    makes the next prompt huge (prompt tokens are the dominant cost and
    can push the LLM call past the client timeout).  The full result is
    still kept in ``steps`` and passed to the tools themselves.
    """
    if isinstance(result, dict) and isinstance(result.get("text"), str):
        text = result["text"]
        if len(text) > config.MESSAGE_PAGE_CHARS:
            result = {
                **result,
                "text": text[: config.MESSAGE_PAGE_CHARS] + "…[truncated]",
            }
    return _preview(result)


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
    total_tool_time = 0.0
    total_fetch_time = 0.0
    context_total = config.LLM_NUM_CTX

    for step in range(1, max_steps + 1):
        step_started = time.perf_counter()
        llm_time = 0.0
        tool_time = 0.0
        fetch_time = 0.0
        try:
            text, calls, info = llm.chat(messages, tools.llm)
        except Exception as exc:
            # Timeout / connection failure: fail the run fast instead of
            # hanging (the openai client's internal retries already ran).
            errors.append(f"step {step}: llm call failed: {type(exc).__name__}: {exc}")
            logutil._print(logutil.error(f"llm call failed: {exc}"))
            break
        llm_time = info["elapsed"]
        total_time += llm_time
        total_tokens += info["tokens"]
        total_prompt_tokens += info["prompt_tokens"]

        # Nudge retry once on empty/degenerate output (HLD §13.16 pattern).
        if not text.strip() and not calls:
            messages.append(HumanMessage(content=config.AGENT_NUDGE))
            try:
                text, nudge_calls, nudge_info = llm.chat(messages, tools.llm)
            except Exception as exc:
                errors.append(f"step {step}: llm call failed: {type(exc).__name__}: {exc}")
                logutil._print(logutil.error(f"llm call failed: {exc}"))
                break
            nudge_llm_time = nudge_info["elapsed"]
            llm_time += nudge_llm_time
            total_time += nudge_llm_time
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

        # Execute all tool calls in parallel, keeping results aligned with
        # the calls (several calls to the same tool must not overwrite).
        with ThreadPoolExecutor(max_workers=min(len(calls), config.FETCH_CONCURRENCY)) as executor:
            futures = [
                executor.submit(_execute_tool, call["tool"], call["args"], step)
                for call in calls
            ]
            executed = [future.result() for future in futures]

        for (name, result, call_errors, elapsed) in executed:
            errors.extend(call_errors)
            if verbose:
                preview = _preview(result)
                logutil._print(
                    logutil.tool_step(step, name, preview)
                    + logutil.dim(f"  {elapsed:.1f}s")
                )
                logutil._print(logutil.tool_result(result))

        # Auto-fetch: pull the top URLs from any web_search results in
        # parallel, without another model round-trip (fast research path).
        fetched_urls = {
            str(call["args"].get("url", ""))
            for call in calls
            if call["tool"] == "fetch_page"
        }
        auto_urls: List[str] = []
        if config.AUTO_FETCH_TOP_N > 0:
            from urllib.parse import urlparse

            def _skipped(url: str) -> bool:
                domain = urlparse(url).netloc.lower()
                return any(skip in domain for skip in config.AUTO_FETCH_SKIP_DOMAINS)

            for call, (name, result, _, _) in zip(calls, executed):
                if name != "web_search" or not isinstance(result, list):
                    continue
                top = [i for i in result[: config.AUTO_FETCH_TOP_N] if isinstance(i, dict) and i.get("url")]
                if not top:
                    continue
                # Quality gate: if most top hits are skip-domains (YouTube,
                # social, …), this search has no fetchable content — skip it.
                if sum(1 for i in top if _skipped(i["url"])) > len(top) / 2:
                    if verbose:
                        logutil._print(logutil.dim("  auto-fetch skipped (low-quality results)"))
                    continue
                for item in top:
                    url = item["url"]
                    if _skipped(url):
                        continue
                    if (
                        url not in fetched_urls
                        and url not in auto_urls
                        and len(auto_urls) < config.AUTO_FETCH_MAX_TOTAL
                    ):
                        auto_urls.append(url)
        if auto_urls:
            fetch_fn = get_tool("fetch_page")
            if fetch_fn is not None:
                if verbose:
                    logutil._print(
                        logutil.dim(f"  auto-fetching {len(auto_urls)} top result(s)…")
                    )
                # Not a `with` block: shutdown(wait=False) so a hung fetch
                # thread can't block the run after we time it out.
                executor = ThreadPoolExecutor(
                    max_workers=min(len(auto_urls), config.FETCH_CONCURRENCY)
                )
                fetch_futures = {
                    executor.submit(fetch_fn["fn"], url): url for url in auto_urls
                }
                auto_results: List[tuple[str, Any]] = []
                try:
                    for future, url in fetch_futures.items():
                        started_f = time.perf_counter()
                        try:
                            # Hard cap: trafilatura's own timeout is not always
                            # honored, so enforce one around the future.
                            res = future.result(timeout=config.REQUEST_TIMEOUT + 10)
                        except FuturesTimeoutError:
                            res = {"error": f"fetch timed out after {config.REQUEST_TIMEOUT + 10}s"}
                        except Exception as exc:
                            res = {"error": f"{type(exc).__name__}: {exc}"}
                        auto_results.append((url, res))
                        if verbose:
                            logutil._print(
                                logutil.tool_step(step, "fetch_page(auto)", _preview(res))
                                + logutil.dim(f"  {time.perf_counter() - started_f:.1f}s")
                            )
                            logutil._print(logutil.tool_result(res))
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

        step_elapsed = time.perf_counter() - step_started
        if verbose:
            logutil._print(
                logutil.step_summary(
                    step, llm_time, tool_time, fetch_time, step_elapsed,
                    len(calls), len(auto_urls),
                )
            )

        # Add all results to messages and check for termination.
        has_final = False
        for call, (name, result, _, _) in zip(calls, executed):
            args = call["args"]
            steps.append({"step": step, "tool": name, "args": args, "result": result})

            messages.append(AIMessage(content=text))
            messages.append(
                HumanMessage(content=f"Tool result ({name}):\n{_for_message(result)}")
            )

            if name == "final_answer":
                answer = str(args.get("answer", result))
                has_final = True
            sentinel = _finalize.take_final_answer()
            if sentinel is not None:
                answer = sentinel
                has_final = True

        if auto_urls:
            for url, res in auto_results:
                steps.append(
                    {"step": step, "tool": "fetch_page", "args": {"url": url}, "result": res}
                )
                messages.append(
                    HumanMessage(
                        content=f"Tool result (fetch_page, auto-fetched for {url}):\n{_for_message(res)}"
                    )
                )

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
            total_tool_time=total_tool_time,
            total_fetch_time=total_fetch_time,
            total_tokens=total_tokens,
            prompt_tokens=total_prompt_tokens,
            context_used=context_used,
            context_total=context_total,
            errors=errors,
        )
    )
    logutil.log_only(logutil.agent(f"Answer: {answer}"))
    logutil.log_only(logutil.separator())
    logutil.close_log()

    if history is not None:
        history.add(query, answer)

    return {"query": query, "answer": answer, "steps": steps, "errors": errors}


__all__ = ["run", "TOOL_REGISTRY"]
