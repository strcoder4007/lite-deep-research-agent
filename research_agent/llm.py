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

from . import logutil
from .tools import _extract_json_object
_PROMPT_TEMPLATE = """You are a general-purpose local agent. Answer the user's request.

Current date: {current_date}
Day of week: {day_of_week}

You have these tools:
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

Rules:
- You may call one or more tools per turn. After all tool results come back, continue.
- When you have everything needed to answer, call the `final_answer` tool
  with the complete answer, or reply with the answer as plain text.
- Do not invent tool results; wait for them.
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
        + f" done in {elapsed:.2f}s"
        + (f" ({total_tokens} tok)" if total_tokens else "")
    )

    reasoning = response.additional_kwargs.get("reasoning", "")
    if reasoning:
        logutil._print(logutil.dim(f"  [reasoning]\n{reasoning}"))

    calls = _parse_tool_calls(text)
    return text, calls if calls else [], info
