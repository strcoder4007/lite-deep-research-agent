from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from .. import config
from . import tool
from .fetch_page import fetch_page


@tool
def fetch_pages(
    urls: List[str],
    max_workers: int = config.FETCH_CONCURRENCY,
) -> List[Dict[str, str]]:
    """Fetch multiple web pages concurrently and return their content."""
    results: List[Dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_page, url): url for url in urls}
        for future in futures:
            try:
                results.append(future.result())
            except Exception:
                url = futures[future]
                results.append({"url": url, "error": "fetch failed"})

    return results