from __future__ import annotations

from typing import Dict

from . import tool
from .base import fetch_url


@tool
def fetch_page(url: str) -> Dict[str, str]:
    """Fetch a web page and return its title and clean text content."""
    title, text = fetch_url(url)
    if title == "error":
        return {"url": url, "error": text}
    return {"url": url, "title": title, "text": text}
