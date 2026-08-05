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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from . import config, llm, logutil, memtrack, memory
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


# Search/fetch tool blocks are noise in the terminal: log them to the run
# log only.  Errors still surface on the terminal (red line).
QUIET_TOOLS = {"web_search", "web_search_batch", "fetch_page", "fetch_page(auto)", "fetch_page(wave2)", "fetch_pages"}


def _log_tool(step: int, name: str, result: Any, elapsed: float = 0.0) -> None:
    """Print a tool step line + result, or log it only for quiet tools."""
    if name in QUIET_TOOLS and not (isinstance(result, dict) and "error" in result):
        logutil.log_only(
            logutil.tool_step(step, name, _preview(result))
            + logutil.dim(f"  {elapsed:.1f}s")
            + "\n"
            + logutil.tool_result(result)
        )
        return
    preview = _preview(result)
    logutil._print(
        logutil.tool_step(step, name, preview)
        + logutil.dim(f"  {elapsed:.1f}s")
    )
    logutil._print(logutil.tool_result(result))


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


def _digest_message(msg: Any) -> str:
    """Return a one-line digest of a tool-result HumanMessage."""
    content = getattr(msg, "content", "")
    if not isinstance(content, str):
        return str(content)[: config.MESSAGE_DIGEST_MAX_CHARS]
    # Extract tool name from "Tool result (<name>):\n..."
    if content.startswith("Tool result ("):
        end = content.find("):")
        if end != -1:
            tool_name = content[len("Tool result ("):end]
            # Get first line of the actual result
            body = content[end + 2 :].strip()
            first_line = body.split("\n")[0][: config.MESSAGE_DIGEST_MAX_CHARS]
            return f"Tool result ({tool_name}): {first_line}"
    return content[: config.MESSAGE_DIGEST_MAX_CHARS]


def _digest_old_messages(
    messages: List[Any], current_step: int
) -> None:
    """Replace older tool-result messages with one-line digests.

    Keeps the most recent ``MESSAGE_DIGEST_AFTER`` steps' tool results
    verbatim; older ones are compressed to save prompt tokens.
    """
    if config.MESSAGE_DIGEST_AFTER <= 0 or current_step <= config.MESSAGE_DIGEST_AFTER:
        return
    # Walk backwards to find the boundary of recent tool results.
    # We keep the last MESSAGE_DIGEST_AFTER tool-result messages.
    tool_result_indices: List[int] = []
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.startswith("Tool result ("):
            tool_result_indices.append(i)
            if len(tool_result_indices) >= config.MESSAGE_DIGEST_AFTER:
                break
    # Digest all tool-result messages older than the kept ones.
    keep_set = set(tool_result_indices)
    for i, msg in enumerate(messages):
        if i in keep_set:
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.startswith("Tool result ("):
            messages[i] = HumanMessage(content=_digest_message(msg))


def _compress_messages(
    messages: List[Any],
    tools: ResearchTools,
    total_prompt_tokens: int,
    context_total: int,
) -> None:
    """Summarize older tool results + history when prompt tokens exceed
    ~50% of the context window.  Replaces old messages with a single
    summary so the run stays bounded.
    """
    if total_prompt_tokens < context_total * 0.5:
        return
    # Find the oldest non-system message index.
    oldest_idx = None
    for i, msg in enumerate(messages):
        if not isinstance(msg, SystemMessage):
            oldest_idx = i
            break
    if oldest_idx is None or oldest_idx >= len(messages) - 2:
        return
    # Keep the system prompt and the last 2 messages (most recent user + assistant).
    # Summarize everything in between.
    old_msgs = messages[oldest_idx : len(messages) - 2]
    if not old_msgs:
        return
    # Build a compact summary of the older messages.
    summary_text = _summarize_old_messages(old_msgs, tools)
    # Replace old messages with a single summary.
    del messages[oldest_idx : len(messages) - 2]
    messages.insert(oldest_idx, HumanMessage(content=summary_text))


def _summarize_old_messages(
    old_msgs: List[Any], tools: ResearchTools
) -> str:
    """Summarize older messages into a compact paragraph using the LLM."""
    # Collect the content of older messages as a single text block.
    parts: List[str] = []
    for msg in old_msgs:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            # Truncate very long tool results in the summary input.
            if content.startswith("Tool result ("):
                parts.append(content[: 500])
            else:
                parts.append(content[: 1000])
        else:
            parts.append(str(content)[: 500])
    combined = "\n".join(parts)
    if len(combined) > 8000:
        combined = combined[:8000]

    prompt = (
        "Summarize the following conversation and tool results in a few "
        "sentences. Include key facts, URLs, and any decisions made. "
        "Be concise.\n\n"
        f"{combined}"
    )
    try:
        from langchain_core.messages import HumanMessage as HM

        response = tools.llm.invoke([SystemMessage(content="Summarize the user's "
            "conversation and tool results briefly."), HM(content=prompt)])
        return getattr(response, "content", str(response))[: 2000]
    except Exception:
        return f"(earlier {len(old_msgs)} messages summarized)"


def _is_follow_up(query: str) -> bool:
    """Detect short/deictic follow-up questions that likely relate
    to prior research (e.g. 'and the price?', 'what about the
    specs?').  These should hit recall_memory before new searches.
    """
    q = query.strip().lower()
    if len(q) > 120:
        return False
    follow_up_markers = [
        "and the", "what about", "how about", "tell me more",
        "what else", "any more", "further", "also", "too",
        "the same", "similar", "like that", "regarding that",
        "on that", "about that", "for that",
    ]
    return any(q.startswith(m) for m in follow_up_markers) or (
        len(q) < 60 and not q.startswith("what") and not q.startswith("how")
        and "?" in q
        and not any(q.startswith(w) for w in ("tell", "explain", "describe", "summarize", "compare"))
    )


def _maybe_recall_memory(
    messages: List[Any],
    query: str,
    tools: ResearchTools,
) -> None:
    """Prepend memory recall results for short follow-up queries."""
    if not _is_follow_up(query):
        return
    try:
        results = memory.query_memory(tools, query, top_k=3)
    except Exception:
        return
    if not results:
        return
    parts = []
    for r in results:
        content = r.get("content", "")
        score = r.get("score", 0)
        parts.append(f"- {content} (relevance: {score:.2f})")
    summary = "Prior research context:\n" + "\n".join(parts)
    messages.insert(1, HumanMessage(content=summary))


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
        # Arg-name mismatch (e.g. model sent {"q": ...} instead of
        # {"query": ...}): if the tool has exactly ONE required parameter
        # and exactly one arg was provided under a different name, remap
        # and retry once.
        error: Exception = exc
        required = [a["name"] for a in entry["schema"] if a["required"] == "True"]
        if len(required) == 1 and len(args) == 1 and required[0] not in args:
            try:
                result = entry["fn"](**{required[0]: next(iter(args.values()))})
                return name, result, [], time.perf_counter() - started
            except Exception as retry_exc:
                error = retry_exc
        return name, {"error": f"bad arguments: {error}"}, [f"step {step}: {name} bad args: {error}"], time.perf_counter() - started
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

    # M2: Memory-first follow-ups — prepend memory recall for
    # short/deictic follow-up questions.
    _maybe_recall_memory(messages, query.strip(), tools)

    steps: List[Dict[str, Any]] = []
    errors: List[str] = []
    answer = ""
    answer_streamed = False
    total_tokens = 0
    total_prompt_tokens = 0
    total_time = 0.0
    total_tool_time = 0.0
    total_fetch_time = 0.0
    context_total = config.LLM_NUM_CTX
    token_budget = context_total * config.TOKEN_BUDGET_GUARD
    consecutive_error_steps = 0
    # Circuit breaker below reads these from the previous step, so they
    # must exist before the loop starts (step 1 included).
    auto_urls: List[str] = []
    auto_results: List[tuple[str, Any]] = []
    # Track domains that consistently fail extraction so we
    # can skip them in future auto-fetches within this run.
    failed_domains: set = set()

    for step in range(1, max_steps + 1):
        step_started = time.perf_counter()
        llm_time = 0.0
        tool_time = 0.0
        fetch_time = 0.0
        try:
            text, calls, info = llm.chat(messages, tools.llm, stream_final=config.STREAM_FINAL)
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
                text, nudge_calls, nudge_info = llm.chat(messages, tools.llm, stream_final=config.STREAM_FINAL)
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
            info = nudge_info

        if not calls:
            # No tool call: treat the text as the final answer.
            answer = text
            answer_streamed = bool(info.get("streamed"))
            steps.append({"step": step, "tool": None, "answer": text})
            if verbose:
                preview = "(streamed above)" if answer_streamed else text
                logutil._print(logutil.tool_step(step, "final(text)", preview))
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
            tool_time += elapsed
            errors.extend(call_errors)
            if verbose:
                _log_tool(step, name, result, elapsed)

        # Circuit breaker: if every tool call errored for 2 consecutive
        # steps, the model is stuck producing invalid calls — fail fast
        # instead of burning steps until MAX_AGENT_STEPS.
        # Also trigger if every auto-fetch result errored (fetch failures
        # indicate the sources are unusable and more rounds won't help).
        all_tool_errors = all(
            isinstance(result, dict) and "error" in result
            for _, result, _, _ in executed
        )
        all_fetch_errors = (
            auto_urls
            and all(
                isinstance(res, dict) and "error" in res
                for _, res in auto_results
            )
            if auto_results
            else False
        )
        if all_tool_errors or all_fetch_errors:
            consecutive_error_steps += 1
        else:
            consecutive_error_steps = 0
        if consecutive_error_steps >= 2:
            errors.append(f"step {step}: stopped: model repeatedly produced invalid tool calls or all fetches failed")
            logutil._print(logutil.error("stopped: model repeatedly produced invalid tool calls or all fetches failed"))
            break

        # Auto-fetch: pull the top URLs from any web_search results in
        # parallel, without another model round-trip (fast research path).
        fetched_urls = {
            str(call["args"].get("url", ""))
            for call in calls
            if call["tool"] == "fetch_page"
        }
        auto_urls: List[str] = []
        from urllib.parse import urlparse

        def _skipped(url: str) -> bool:
            domain = urlparse(url).netloc.lower()
            return any(skip in domain for skip in config.AUTO_FETCH_SKIP_DOMAINS) or domain in failed_domains

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
                ):
                    auto_urls.append(url)
        # Prioritize reputable domains: preferred domains first.
        def _preferred(url: str) -> bool:
            domain = urlparse(url).netloc.lower()
            return any(pref in domain for pref in config.PREFERRED_DOMAINS)

        auto_urls.sort(key=lambda u: (0 if _preferred(u) else 1, u))
        auto_urls = auto_urls[: config.AUTO_FETCH_MAX_TOTAL]
        if auto_urls:
            fetch_fn = get_tool("fetch_page")
            if fetch_fn is not None:
                if verbose:
                    logutil._print(
                        logutil.dim(f"  auto-fetching {len(auto_urls)} top result(s)…")
                    )
                # Not a `with` block: shutdown(wait=False) so a hung fetch
                # thread can't block the run after we time it out.
                fetch_start = time.perf_counter()
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
                        # Track domains whose fetches consistently fail
                        # so we skip them in future waves.
                        if isinstance(res, dict) and "error" in res:
                            failed_domains.add(urlparse(url).netloc.lower())
                        if verbose:
                            _log_tool(step, "fetch_page(auto)", res, time.perf_counter() - started_f)
                finally:
                    fetch_time += time.perf_counter() - fetch_start
                    executor.shutdown(wait=False, cancel_futures=True)

        # Adaptive fetch waves (S3): after the first wave, check whether
        # the fetched content looks thin (total chars / successful sources).
        # If thin and we haven't hit the max, fetch the next wave.
        if (
            auto_urls
            and config.AUTO_FETCH_WAVE_THRESHOLD > 0
            and len(auto_results) < config.AUTO_FETCH_MAX_TOTAL
        ):
            total_chars = sum(
                len(r.get("text", ""))
                for _, r in auto_results
                if isinstance(r, dict) and "text" in r
            )
            successful = sum(1 for _, r in auto_results if isinstance(r, dict) and "text" in r)
            if (
                successful >= config.AUTO_FETCH_MIN_SUCCESSFUL
                and total_chars / max(successful, 1) < config.AUTO_FETCH_WAVE_THRESHOLD
            ):
                # Content looks thin — fetch more.
                remaining = config.AUTO_FETCH_MAX_TOTAL - len(auto_urls)
                extra_urls = [url for url in auto_urls if url not in {a[0] for a in auto_results} and urlparse(url).netloc.lower() not in failed_domains]
                for url in extra_urls[: remaining]:
                    if verbose:
                        logutil._print(logutil.dim(f"  adaptive wave: fetching {url}…"))
                    wave_start = time.perf_counter()
                    try:
                        res = fetch_fn["fn"](url)
                        fetch_time += time.perf_counter() - wave_start
                        auto_results.append((url, res))
                        if isinstance(res, dict) and "error" in res:
                            failed_domains.add(urlparse(url).netloc.lower())
                        if verbose:
                            _log_tool(step, "fetch_page(wave2)", res)
                    except Exception as exc:
                        fetch_time += time.perf_counter() - wave_start
                        auto_results.append((url, {"error": str(exc)}))
                        failed_domains.add(urlparse(url).netloc.lower())

        step_elapsed = time.perf_counter() - step_started
        if verbose:
            logutil._print(
                logutil.step_summary(
                    step, llm_time, tool_time, fetch_time, step_elapsed,
                    len(calls), len(auto_urls),
                )
            )
            memtrack.print_memory_stats(
                step, total_prompt_tokens, context_total,
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

        _digest_old_messages(messages, step)

        # C1: Context compression — when prompt tokens exceed ~50%
        # of the context window, summarize older messages.
        _compress_messages(messages, tools, total_prompt_tokens, context_total)

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

    # M1: Auto-remember research results — write a dated summary
    # of the final answer to Chroma so follow-up turns start from
    # memory instead of zero.
    if answer and any(s.get("tool") == "web_search" for s in steps):
        try:
            memory.add_to_memory(
                tools,
                answer,
                metadata={
                    "title": query,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
        except Exception:
            pass

    return {"query": query, "answer": answer, "steps": steps, "errors": errors, "streamed": answer_streamed}


__all__ = ["run", "TOOL_REGISTRY"]
