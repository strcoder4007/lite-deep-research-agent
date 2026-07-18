from __future__ import annotations

import os
import re
from typing import Any, List

_USE_COLOR = os.getenv("NO_COLOR") is None and os.getenv("TERM", "") != "dumb"


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{code}{text}{C.RESET}"


def bold(text: str) -> str:
    return _c(C.BOLD, text)


def dim(text: str) -> str:
    return _c(C.DIM, text)


def red(text: str) -> str:
    return _c(C.RED, text)


def green(text: str) -> str:
    return _c(C.GREEN, text)


def yellow(text: str) -> str:
    return _c(C.YELLOW, text)


def blue(text: str) -> str:
    return _c(C.BLUE, text)


def magenta(text: str) -> str:
    return _c(C.MAGENTA, text)


def cyan(text: str) -> str:
    return _c(C.CYAN, text)


def node_label(name: str) -> str:
    return _c(C.CYAN + C.BOLD, f"[{name}]")


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, limit: int = 200) -> str:
    text = _collapse_ws(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _truncate_list(items: List[Any], limit: int = 200) -> str:
    parts = [truncate(str(i), 120) for i in items]
    joined = "  •  ".join(parts)
    return truncate(joined, limit)


def model_preview(label: str, content: str, limit: int = 200) -> str:
    preview = truncate(content, limit)
    return (
        "\n      "
        + magenta(f"{label}: ")
        + dim(preview)
    )


def facts_preview(facts: List[str], limit: int = 220) -> str:
    if not facts:
        return "\n      " + red("model output: (empty)")
    return "\n      " + magenta("facts: ") + dim(_truncate_list(facts, limit))


def preview_for_node(node: str, payload: dict) -> str:
    """Return a colored, truncated preview of the node's model output."""
    if node == "plan":
        plan = payload.get("research_plan") or {}
        raw = yaml_safe_dump(plan)
        return model_preview("plan", raw)
    if node == "analyze":
        return facts_preview(payload.get("extracted_facts", []))
    if node == "synthesize":
        return model_preview("report", payload.get("final_answer") or "", 280)
    return ""


def yaml_safe_dump(data: Any) -> str:
    try:
        import yaml

        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
    except Exception:
        return str(data)
