from __future__ import annotations

from typing import Dict, List

from .. import config
from . import tool
from .base import run_ddg_search


@tool
def web_search(query: str, max_results: int = config.SEARCH_RESULTS_PER_QUERY) -> List[Dict[str, str]]:
    """Search the web with DuckDuckGo and return matching pages."""
    results = run_ddg_search(query, max_results=max_results)
    return [
        {"url": r["url"], "title": r["title"], "snippet": r["snippet"]}
        for r in results
    ]
