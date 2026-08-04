"""Lightweight tool registry for the general-purpose agent loop.

Decorating a function with ``@tool`` in any ``tools/*.py`` submodule is the
ONLY wiring needed: submodules are auto-discovered and imported when this
package is imported, and each decorated function is registered in
``TOOL_REGISTRY``.

Everything from the original ``tools.py`` (factories + helpers) lives in
``tools/base.py`` and is re-exported here, so existing imports like
``from .tools import ResearchTools`` keep working unchanged.
"""
from __future__ import annotations

import inspect
import pkgutil
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional

# Re-export the original factories + helpers (run_ddg_search, fetch_url,
# _extract_json_object, cosine_similarity, count_tokens, create_llm,
# create_embedder, create_vectorstore, create_text_splitter, build_tools,
# ResearchTools, timestamp, ...).
from .base import *  # noqa: F401,F403
from .base import ResearchTools, _extract_json_object  # explicit re-exports

# name -> {"name", "description", "fn", "schema"}
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Shared ResearchTools (llm/embedder/vectorstore/text_splitter), set once at
# agent startup via init_tools().
shared_tools: Optional[ResearchTools] = None


def init_tools(tools: ResearchTools) -> None:
    """Make the shared ResearchTools available to all tool modules."""
    global shared_tools
    shared_tools = tools


def get_shared() -> ResearchTools:
    """Return the shared ResearchTools, raising if init_tools() wasn't called."""
    if shared_tools is None:
        raise RuntimeError("tools.init_tools() must be called before using tools")
    return shared_tools


def _arg_schema(fn: Callable[..., Any]) -> List[Dict[str, str]]:
    schema = []
    for name, param in inspect.signature(fn).parameters.items():
        ann = param.annotation
        ann_str = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
        schema.append(
            {
                "name": name,
                "type": ann_str,
                "required": str(param.default is inspect.Parameter.empty),
            }
        )
    return schema


def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Register ``fn`` as an agent-callable tool."""
    TOOL_REGISTRY[fn.__name__] = {
        "name": fn.__name__,
        "description": (inspect.getdoc(fn) or "").strip().splitlines()[0]
        if inspect.getdoc(fn)
        else "",
        "fn": fn,
        "schema": _arg_schema(fn),
    }
    return fn


def get_tool(name: str) -> Optional[Dict[str, Any]]:
    return TOOL_REGISTRY.get(name)


def list_tools() -> List[str]:
    return sorted(TOOL_REGISTRY)


def build_catalog() -> str:
    """Render the tool catalog as a markdown list for the system prompt."""
    lines = []
    for name in list_tools():
        entry = TOOL_REGISTRY[name]
        args = ", ".join(
            f"{a['name']}: {a['type']}" + ("" if a["required"] == "True" else " (optional)")
            for a in entry["schema"]
        )
        lines.append(f"- **{name}**({args}) — {entry['description']}")
    return "\n".join(lines)


def _discover_tool_modules() -> None:
    """Import every ``tools/*.py`` submodule so their @tool decorators run."""
    skip = {"base"}
    for info in pkgutil.iter_modules(__path__):
        if info.name in skip or info.name.startswith("_"):
            continue
        import_module(f"{__name__}.{info.name}")


_discover_tool_modules()
