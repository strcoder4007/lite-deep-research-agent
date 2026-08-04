from __future__ import annotations

from typing import Any, Dict, List

from .. import memory
from . import get_shared, tool
from .base import timestamp


@tool
def recall_memory(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Recall relevant notes from long-term memory (past research runs)."""
    return memory.query_memory(get_shared(), query, top_k=top_k)


@tool
def remember(text: str, title: str = "") -> Dict[str, Any]:
    """Save an important piece of information to long-term memory."""
    docs = memory.add_to_memory(
        get_shared(), text, metadata={"title": title, "timestamp": timestamp()}
    )
    return {"stored_chunks": len(docs)}
