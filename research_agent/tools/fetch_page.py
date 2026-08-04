from __future__ import annotations

from typing import Dict, Union

from . import tool
from .base import fetch_url


@tool
def fetch_page(url: str) -> Dict[str, str]:
    """Fetch a web page and return its title and clean text content."""
    result = fetch_url(url)
    if result is None:
        return {"url": url, "error": "could not fetch or extract page content"}
    title, text = result
    return {"url": url, "title": title, "text": text}
