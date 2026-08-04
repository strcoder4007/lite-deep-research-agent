"""Chat helper for the agent loop: system prompt + tool-call parsing.

Tool-call format, kept robust for a weak 2-bit local model: to use a tool the
model replies with ONLY a JSON block:

    ```json
    {"tool": "<name>", "args": {"<arg>": <value>}}
    ```

ONE tool call per turn. Any other reply is treated as the final answer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from .tools import _extract_json_object

_PROMPT_TEMPLATE = """You are a general-purpose local agent. Answer the user's request.

You have these tools:
{catalog}

To use a tool, reply with ONLY a JSON block, nothing else:
```json
{{"tool": "<name>", "args": {{"<arg>": <value>}}}}
```

Rules:
- ONE tool call per turn. After the tool result comes back, continue.
- When you have everything needed to answer, call the `final_answer` tool
  with the complete answer, or reply with the answer as plain text.
- Do not invent tool results; wait for them.
"""


def build_system_prompt(catalog: str) -> str:
    return _PROMPT_TEMPLATE.format(catalog=catalog)


def chat(
    messages: List[BaseMessage], llm: BaseChatModel
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Send messages to the LLM.

    Returns (text, toolcall) where toolcall is {"tool": name, "args": {...}}
    or None when the reply was plain text (treated as a final answer).
    """
    response = llm.invoke(messages)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    text = str(content).strip()
    parsed = _extract_json_object(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("tool"), str):
        args = parsed.get("args")
        return text, {"tool": parsed["tool"], "args": args if isinstance(args, dict) else {}}
    return text, None
