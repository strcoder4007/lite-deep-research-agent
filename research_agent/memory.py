from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from . import config
from .tools import ResearchTools


def add_to_memory(
    tools: ResearchTools,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    if not text or len(text) < config.MEMORY_MIN_CHARS:
        return []
    docs = tools.text_splitter.create_documents(
        [text], metadatas=[metadata or {}]
    )
    if not docs:
        return []
    tools.vectorstore.add_documents(docs)
    tools.vectorstore.persist()
    return docs


def _recency_boost(meta: Dict[str, Any]) -> float:
    ts = meta.get("timestamp")
    if not ts:
        return 0.0
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        return 0.0
    age_hours = max((datetime.utcnow() - dt).total_seconds() / 3600, 0.1)
    return 1.0 / (1.0 + age_hours / 24.0)


def query_memory(
    tools: ResearchTools,
    query: str,
    aspects: Optional[List[str]] = None,
    top_k: int = config.MEMORY_TOP_K,
    threshold: float = config.MEMORY_SIMILARITY_THRESHOLD,
) -> List[Dict[str, Any]]:
    if not query:
        return []
    extended = query
    if aspects:
        extended = extended + " | aspects: " + "; ".join(aspects)
    results = tools.vectorstore.similarity_search_with_relevance_scores(
        extended, k=top_k
    )
    filtered = []
    for doc, score in results:
        meta = doc.metadata or {}
        boosted = score + 0.05 * _recency_boost(meta)
        if boosted < threshold:
            continue
        filtered.append(
            {
                "content": doc.page_content,
                "metadata": meta,
                "score": boosted,
            }
        )
    return filtered
