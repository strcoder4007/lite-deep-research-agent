from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, List, Optional, TextIO

_USE_COLOR = os.getenv("NO_COLOR") is None and os.getenv("TERM", "") != "dumb"
_log_file: Optional[TextIO] = None


def set_log_file(path: str) -> None:
    global _log_file
    _log_file = open(path, "a", encoding="utf-8")


def close_log() -> None:
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def _print(msg: str) -> None:
    print(msg)
    if _log_file is not None:
        _log_file.write(strip_ansi(msg) + "\n")
        _log_file.flush()


def log_only(msg: str) -> None:
    """Write to the log file without printing to the terminal."""
    if _log_file is not None:
        _log_file.write(strip_ansi(msg) + "\n")
        _log_file.flush()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{code}{text}{C.RESET}"


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


def strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


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


def user(text: str) -> str:
    return green(bold(f" you: {text}"))


def agent(text: str) -> str:
    return cyan(bold(f" agent: {text}"))


def success(text: str) -> str:
    return green(f" ✓ {text}")


def error(text: str) -> str:
    return red(bold(f" ✗ {text}"))


def header(text: str) -> str:
    return magenta(bold(text))


def separator() -> str:
    return dim("─" * 50)


def tool_result(result: Any) -> str:
    if isinstance(result, dict) and "error" in result:
        return red(f"  error: {result['error']}")
    if isinstance(result, dict):
        keys = ", ".join(result.keys())
        return green(f"  ok ({keys})")
    return green(f"  ok: {truncate(str(result), 120)}")


def thinking() -> str:
    return cyan("  thinking…")


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
    header = node_label(f"step {step}") + " " + yellow(f"tool={tool_name}")
    if preview:
        header += dim(" | " + truncate(preview, limit))
    return header


def stage(label: str) -> str:
    """Return a colored stage indicator."""
    return cyan(f"[{label}]")


def status_line(
    elapsed: float,
    tokens: int,
    prompt_tokens: int,
    context_total: int,
) -> str:
    """Persistent one-line status: tok/s, tokens, latency, context usage."""
    tps = tokens / elapsed if elapsed > 0 and tokens else 0.0
    parts = [
        dim("  ── "),
        yellow(bold(f"{tps:.1f} tok/s")),
        dim("  ·  "),
        cyan(f"{abbr(tokens)} tok"),
        dim("  ·  "),
        blue(f"{elapsed:.1f}s"),
    ]
    if context_total > 0:
        parts += [dim("  ·  "), dim("ctx ") + context_bar(prompt_tokens, context_total, width=14)]
    return "".join(parts)


def step_summary(step: int, llm_time: float, tool_time: float, fetch_time: float, total_time: float, tool_count: int, fetch_count: int) -> str:
    """Return a colored one-line summary of a step's timing breakdown."""
    parts = [
        dim("──"),
        magenta(bold(f" step {step} ")),
        dim("|"),
        cyan(f" LLM {llm_time:.1f}s"),
        dim("|"),
        yellow(f" tools {tool_time:.1f}s"),
        dim("|"),
        green(f" fetch {fetch_time:.1f}s"),
        dim("|"),
        bold(f" total {total_time:.1f}s"),
        dim(f" ({tool_count} tools"),
        dim(f" + {fetch_count} fetch)"),
    ]
    return "".join(parts)


def run_summary(
    total_steps: int,
    total_time: float,
    total_tool_time: float,
    total_fetch_time: float,
    total_tokens: int,
    prompt_tokens: int,
    context_used: int,
    context_total: int,
    errors: List[str],
) -> str:
    """Return a colored end-of-run summary block."""
    lines = [
        "",
        header("═══ Run Summary ═══"),
        f"  Steps:           {total_steps}",
        f"  Total time:      {total_time:.1f}s",
    ]
    if total_tool_time > 0:
        lines.append(f"  Tool time:       {total_tool_time:.1f}s")
    if total_fetch_time > 0:
        lines.append(f"  Auto-fetch time: {total_fetch_time:.1f}s")
    if total_tokens > 0:
        avg = total_tokens / total_time if total_time > 0 else 0
        lines.append(f"  Avg speed:       {avg:.1f} tok/s")
        lines.append(f"  Tokens:          {total_tokens} total ({prompt_tokens} prompt)")
    if context_total > 0:
        lines.append(
            "  Context:         "
            + context_bar(context_used, context_total)
        )
    if errors:
        lines.append(f"  Errors:          {len(errors)}")
        for err in errors:
            lines.append(f"    {error(err)}")
    else:
        lines.append(f"  Errors:          {success('none')}")
    lines.append(header("═══"))
    return "\n".join(lines)


def yaml_safe_dump(data: Any) -> str:
    try:
        import yaml

        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
    except Exception:
        return str(data)