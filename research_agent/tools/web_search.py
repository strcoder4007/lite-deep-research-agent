from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


@tool
def web_search_batch(queries: List[str], max_results: int = config.SEARCH_RESULTS_PER_QUERY) -> List[Dict[str, str]]:
    """Run multiple DuckDuckGo searches in parallel and return all results."""
    results: List[Dict[str, str]] = []

    def _search(q: str) -> List[Dict[str, str]]:
        return [
            {"url": r["url"], "title": r["title"], "snippet": r["snippet"], "query": q}
            for r in run_ddg_search(q, max_results=max_results)
        ]

    with ThreadPoolExecutor(max_workers=min(len(queries), config.FETCH_CONCURRENCY)) as executor:
        futures = {executor.submit(_search, q): q for q in queries}
        for future in futures:
            try:
                results.extend(future.result())
            except Exception:
                q = futures[future]
                results.append({"url": "", "title": "", "snippet": f"search failed for: {q}", "query": q})

    return results
