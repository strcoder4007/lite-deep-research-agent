from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from .. import config
from . import tool
from .base import fetch_url


@tool
def fetch_pages(
    urls: List[str],
    max_workers: int = config.FETCH_CONCURRENCY,
) -> List[Dict[str, str]]:
    """Fetch multiple web pages concurrently and return their content."""
    results: List[Dict[str, str]] = []

    def _fetch(url: str) -> Optional[Dict[str, str]]:
        result = fetch_url(url)
        if result is None:
            return {"url": url, "error": "could not fetch or extract page content"}
        title, text = result
        return {"url": url, "title": title, "text": text}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch, url): url for url in urls}
        for future in futures:
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception:
                url = futures[future]
                results.append({"url": url, "error": "fetch failed"})

    return results