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


def abbr(n: int) -> str:
    """Compact number abbreviation: 1500 -> 1.5K, 78000 -> 78K."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}K".rstrip("0").rstrip(".")
    return f"{n / 1_000_000:.1f}M".rstrip("0").rstrip(".")


def context_bar(used: int, total: int, width: int = 20) -> str:
    """Render a colored context-usage bar (used/total)."""
    if total <= 0:
        pct = 0.0
    else:
        pct = min(used / total, 1.0)
    filled = int(round(pct * width))
    bar = "█" * filled + "░" * (width - filled)
    pct_str = f"{pct * 100:.1f}%"
    color = C.GREEN
    if pct > 0.85:
        color = C.RED
    elif pct > 0.6:
        color = C.YELLOW
    return _c(color, bar) + dim(f" {abbr(used)}/{abbr(total)} ({pct_str})")


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


def tool_step(step: int, tool_name: str, preview: str, limit: int = 200) -> str:
    """Colored per-step line for the agent loop: step N | tool=<name> | preview."""
    header = node_label(f"step {step}") + " " + blue(f"tool={tool_name}")
    if preview:
        header += dim(" | " + truncate(preview, limit))
    return header


def yaml_safe_dump(data: Any) -> str:
    try:
        import yaml

        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
    except Exception:
        return str(data)
