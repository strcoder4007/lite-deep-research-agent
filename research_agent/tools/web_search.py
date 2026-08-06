from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from .. import config
from . import tool
from .base import run_ddg_search

# TTL in-memory cache: (query, max_results, since_days, time_limit) -> (timestamp, results).
_CACHE: Dict[Tuple[str, int, int, Optional[str]], Tuple[float, List[Dict[str, str]]]] = {}


def _detect_time_sensitive(query: str) -> Tuple[int, Optional[str]]:
    """Infer the recency window the query implies, or (0, None) if it
    doesn't ask for fresher results."""
    q = query.strip().lower()

    # Explicit windows win over vague recency words.
    if any(kw in q for kw in ["last 24 hours", "past day", "past 24 hours", "today", "yesterday"]):
        return 1, None
    if any(kw in q for kw in ["this week", "last week", "last 7 days", "past week", "past 7 days"]):
        return 7, None
    if any(kw in q for kw in ["this month", "last month", "past 30 days", "last 30 days"]):
        return 30, None
    if any(kw in q for kw in ["last year", "past year", "this year", "past 365 days"]):
        return 365, None

    # Vague recency signals ("latest", "recent", "current", "newest",
    # "up to date", "as of now", "any news on") default to the past week.
    if any(kw in q for kw in [
        "latest", "recent", "newest", "breaking", "current",
        "updated", "up to date", "as of now", "right now",
        "what's new", "whats new", "any news", "news on",
        "new version", "new release", "released", "announced",
        "launched", "unveiled",
    ]):
        return 7, None

    return 0, None


def _search_cached(query: str, max_results: int, since_days: int = 0, time_limit: Optional[str] = None) -> List[Dict[str, str]]:
    key = (query.strip().lower(), max_results, since_days, time_limit)
    # Recency-filtered queries must never be served from cache: "the
    # latest X" has a short shelf life and cached hits go stale fast.
    cached_ok = config.SEARCH_CACHE_TTL > 0 and since_days == 0 and time_limit is None
    if cached_ok:
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < config.SEARCH_CACHE_TTL:
            return hit[1]
    results = [
        {"url": r["url"], "title": r["title"], "snippet": r["snippet"]}
        for r in run_ddg_search(query, max_results=max_results, since_days=since_days, time_limit=time_limit)
    ]
    if cached_ok:
        _CACHE[key] = (time.time(), results)
    return results


@tool
def web_search(query: str, max_results: int = config.SEARCH_RESULTS_PER_QUERY, since_days: int = 0, time_limit: Optional[str] = None) -> List[Dict[str, str]]:
    """Search the web with DuckDuckGo and return matching pages.

    For time-sensitive queries (e.g. "this week", "last 7 days"), set
    since_days=7 to filter results to the last 7 days.  time_limit
    accepts DuckDuckGo time filters: "d" (day), "w" (week), "m" (month), "y" (year).
    """
    if since_days == 0 and time_limit is None:
        detected_days, detected_limit = _detect_time_sensitive(query)
        if detected_days > 0:
            since_days = detected_days
        if detected_limit is not None:
            time_limit = detected_limit
    return _search_cached(query, max_results, since_days, time_limit)


@tool
def web_search_batch(queries: List[str], max_results: int = config.SEARCH_RESULTS_PER_QUERY, since_days: int = 0, time_limit: Optional[str] = None) -> List[Dict[str, str]]:
    """Run multiple DuckDuckGo searches in parallel and return all results."""
    results: List[Dict[str, str]] = []

    def _search(q: str) -> List[Dict[str, str]]:
        q_since = since_days
        q_limit = time_limit
        if since_days == 0 and time_limit is None:
            q_since, q_limit = _detect_time_sensitive(q)
        return [{**r, "query": q} for r in _search_cached(q, max_results, q_since, q_limit)]

    with ThreadPoolExecutor(max_workers=min(len(queries), config.FETCH_CONCURRENCY)) as executor:
        futures = {executor.submit(_search, q): q for q in queries}
        for future in futures:
            try:
                results.extend(future.result())
            except Exception:
                q = futures[future]
                results.append({"url": "", "title": "", "snippet": f"search failed for: {q}", "query": q})

    return results
