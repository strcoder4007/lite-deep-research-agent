"""Chat helper for the agent loop: system prompt + tool-call parsing.

Tool-call format, kept robust for a weak 2-bit local model: to use a tool
the model replies with ONLY a JSON block:

    ```json
    {"tool": "<name>", "args": {"<arg>": <value>}}
    ```

Multiple tool calls per turn are supported by returning a JSON array:

    ```json
    [
      {"tool": "<name1>", "args": {"<arg>": <value>}},
      {"tool": "<name2>", "args": {"<arg>": <value>}}
    ]
    ```

Any other reply is treated as the final answer.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from . import config, logutil
from .tools import _extract_json_object, count_tokens
_PROMPT_TEMPLATE = """You are a research assistant running on the user's local machine. Your job is to produce thorough, well-sourced answers to research questions.

Current date: {current_date}
Day of week: {day_of_week}

## Personality
- Be direct, factual, and well-organized. No fluff.
- This is a multi-turn chat: remember earlier context and follow up coherently.
- Match the user's language and tone.

## Tools
{catalog}

To use a tool, reply with ONLY a JSON block, nothing else:
```json
{{"tool": "<name>", "args": {{"<arg>": <value>}}}}
```

You can call multiple tools in a single turn by returning a JSON array:
```json
[
  {{"tool": "<name1>", "args": {{"<arg>": <value>}}}},
  {{"tool": "<name2>", "args": {{"<arg>": <value>}}}}
]
```

## Search strategy
- On the FIRST turn, plan the COMPLETE set of independent searches needed to cover the topic from multiple angles, and emit ALL of them in a single JSON array. Do not search one query at a time.
- One search round is enough: plan ALL searches upfront. Do NOT run another round of searches afterwards unless the results were genuinely missing or irrelevant AND you have a specific new sub-question that was not already covered. Never repeat or lightly rephrase an earlier query.
- Vary your search queries: use different phrasings, include the current date/year for time-sensitive topics, and cover sub-topics separately.
- Aim for 4-8 diverse `web_search` calls on the first turn. More is fine for complex topics.
- Use `since_days` and `time_limit` parameters on `web_search` for time-sensitive queries (e.g. since_days=7 for "this week", time_limit="w" for past week).
- After search results arrive, the top URLs are fetched automatically. Only call `fetch_page` yourself for specific extra URLs beyond those.
- Prefer authoritative sources: official docs, academic papers, reputable news outlets, and primary sources over blog posts or social media.
- If a search returned mostly low-quality or irrelevant results, rephrase ONLY that specific query once with different keywords — do not launch a whole new round.
- Use `recall_memory` when the question may relate to earlier conversations or previously researched topics.
- Use `remember` to store durable facts, preferences, and research findings worth keeping.
- Do not invent tool results; wait for them. If a tool errors, adapt (rephrase the query, try another source) instead of repeating the same call.
- Ground research answers in what the tools returned and cite source URLs when you use fetched information.
- After auto-fetched pages are available, synthesize an answer from them. Do not launch another search round just to find more sources — the auto-fetch already pulled the top results.

## Source quality
- Prefer authoritative sources: official docs, academic papers, reputable news outlets, and primary sources over blog posts or social media.
- When sources disagree, flag the contradiction and prefer the more recent or authoritative source.
- Cite source URLs inline so the user can verify each claim.

## Finishing
- When you have everything needed to answer, call the `final_answer` tool with the complete answer, or reply with the answer as plain text.
- Never output partial answers as tool calls; the final answer must stand on its own.
"""


def build_system_prompt(catalog: str) -> str:
    now = datetime.now()
    return _PROMPT_TEMPLATE.format(
        current_date=now.strftime("%Y-%m-%d"),
        day_of_week=now.strftime("%A"),
        catalog=catalog,
    )


def _parse_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse one or more tool calls from model output.

    Returns a list of {"tool": name, "args": {...}} dicts, or None if
    the text does not contain a valid tool call.
    """
    parsed = _extract_json_object(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("tool"), str):
        args = parsed.get("args")
        return [{"tool": parsed["tool"], "args": args if isinstance(args, dict) else {}}]
    if isinstance(parsed, list):
        calls: List[Dict[str, Any]] = []
        for item in parsed:
            if isinstance(item, dict) and isinstance(item.get("tool"), str):
                args = item.get("args")
                calls.append({"tool": item["tool"], "args": args if isinstance(args, dict) else {}})
        return calls if calls else None
    return None


def chat(
    messages: List[BaseMessage], llm: BaseChatModel, stream_final: bool = False
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Send messages to the LLM.

    Returns (text, toolcalls, info) where:
      - toolcalls is a list of {"tool": name, "args": {...}} (empty list if no tool call)
      - info is {"elapsed": float, "tokens": int, "prompt_tokens": int, "ttft": float}
    """
    logutil._print(logutil.stage("llm") + " calling model ...")
    started = time.perf_counter()
    first_token_time: Optional[float] = None
    chunks: List[str] = []
    last_chunk: Any = None
    token_usage: Dict[str, Any] = {}
    _buffer: List[str] = []
    _streaming = False
    _structured_detected = False

    for chunk in llm.stream(messages):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        # Usage usually arrives on a final content-less chunk, so collect
        # it from every chunk instead of only the last content chunk.
        usage_meta = getattr(chunk, "usage_metadata", None)
        if usage_meta:
            token_usage = {
                "prompt_tokens": usage_meta.get("input_tokens", 0),
                "completion_tokens": usage_meta.get("output_tokens", 0),
                "total_tokens": usage_meta.get("total_tokens", 0),
            }
        else:
            fallback_usage = (getattr(chunk, "response_metadata", None) or {}).get("token_usage")
            if fallback_usage:
                token_usage = fallback_usage
        content = getattr(chunk, "content", "")
        if content:
            chunks.append(str(content))
            last_chunk = chunk
            # Heuristic: decide from the accumulated buffer whether this
            # turn is structured (JSON tool call / fence) or plain text.
            # Only plain text is streamed live; structured output never is.
            if stream_final and not _streaming and not _structured_detected:
                _buffer.append(str(content))
                stripped = "".join(_buffer).strip()
                if stripped.startswith(("{", "[", "```")):
                    _structured_detected = True
                elif len(stripped) > 50 and not any(c in stripped for c in "{["):
                    # No structured markers after 50 chars — plain text
                    # (the final answer), start streaming it live.
                    _streaming = True
                    print(logutil.C.CYAN, end="", flush=True)
                    print(stripped, end="", flush=True)
            elif _streaming and not _structured_detected:
                # Continue streaming: chunks are deltas, print as-is.
                print(str(content), end="", flush=True)

    elapsed = time.perf_counter() - started
    ttft = first_token_time - started if first_token_time is not None else 0.0

    text = "".join(chunks).strip()

    # If we were streaming plain text, print a newline at the end.
    streamed = _streaming and not _structured_detected
    if streamed:
        print(logutil.C.RESET)
        print()

    response = last_chunk if last_chunk is not None else chunks[-1] if chunks else ""
    total_tokens = token_usage.get("total_tokens", 0)
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    estimated = False
    if not total_tokens and not prompt_tokens:
        # Server ignored include_usage — fall back to a rough estimate so
        # the budget guard / compression trigger still work.
        estimated = True
        prompt_tokens = sum(
            count_tokens(None, str(getattr(m, "content", m))) for m in messages
        )
        total_tokens = prompt_tokens + count_tokens(None, text)
    info = {"elapsed": elapsed, "tokens": total_tokens, "prompt_tokens": prompt_tokens, "ttft": ttft, "streamed": streamed}

    logutil._print(
        logutil.stage("llm")
        + logutil.dim(" done")
        + "\n"
        + logutil.status_line(elapsed, total_tokens, prompt_tokens, config.LLM_NUM_CTX, ttft)
        + (logutil.dim(" (estimated)") if estimated else "")
    )

    reasoning = getattr(response, "additional_kwargs", {}).get("reasoning", "")
    if reasoning:
        logutil._print(logutil.dim(f"  [reasoning]\n{reasoning}"))

    calls = _parse_tool_calls(text)
    return text, calls if calls else [], info

