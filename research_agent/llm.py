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
from .tools import _extract_json_object
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
- Vary your search queries: use different phrasings, include the current date/year for time-sensitive topics, and cover sub-topics separately.
- Aim for 4-8 diverse `web_search` calls on the first turn. More is fine for complex topics.
- After search results arrive, the top URLs are fetched automatically. Only call `fetch_page` yourself for specific extra URLs beyond those.
- Prefer authoritative sources: official docs, academic papers, reputable news outlets, and primary sources over blog posts or social media.
- If a search returns mostly low-quality or irrelevant results, rephrase the query and search again with different keywords.
- Use `recall_memory` when the question may relate to earlier conversations or previously researched topics.
- Use `remember` to store durable facts, preferences, and research findings worth keeping.
- Do not invent tool results; wait for them. If a tool errors, adapt (rephrase the query, try another source) instead of repeating the same call.
- Ground research answers in what the tools returned and cite source URLs when you use fetched information.

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
    messages: List[BaseMessage], llm: BaseChatModel
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Send messages to the LLM.

    Returns (text, toolcalls, info) where:
      - toolcalls is a list of {"tool": name, "args": {...}} (empty list if no tool call)
      - info is {"elapsed": float, "tokens": int, "prompt_tokens": int}
    """
    logutil._print(logutil.stage("llm") + " calling model ...")
    started = time.perf_counter()
    response = llm.invoke(messages)
    elapsed = time.perf_counter() - started

    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    text = str(content).strip()

    token_usage: Dict[str, Any] = (
        response.response_metadata.get("token_usage", {})
        if hasattr(response, "response_metadata")
        else {}
    )
    total_tokens = token_usage.get("total_tokens", 0)
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    info = {"elapsed": elapsed, "tokens": total_tokens, "prompt_tokens": prompt_tokens}

    logutil._print(
        logutil.stage("llm")
        + logutil.dim(" done")
        + "\n"
        + logutil.status_line(elapsed, total_tokens, prompt_tokens, config.LLM_NUM_CTX)
    )

    reasoning = response.additional_kwargs.get("reasoning", "")
    if reasoning:
        logutil._print(logutil.dim(f"  [reasoning]\n{reasoning}"))

    calls = _parse_tool_calls(text)
    return text, calls if calls else [], info

