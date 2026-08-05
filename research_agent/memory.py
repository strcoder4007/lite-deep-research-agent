from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage

from . import config
from .tools import ResearchTools, cosine_similarity


class Scratchpad:
    """Tiny in-memory working notes for the agent loop."""

    def __init__(self) -> None:
        self._notes: List[str] = []

    def add(self, note: str) -> None:
        note = (note or "").strip()
        if note:
            self._notes.append(note)

    def render(self) -> str:
        return "\n".join(f"- {note}" for note in self._notes)


class ConversationMemory:
    """Simple in-memory store of previous query/answer turns.

    Older turns beyond ``max_turns`` are compressed into a rolling
    summary so long REPL sessions don't grow the prompt linearly.
    """

    def __init__(self, max_turns: int = 10) -> None:
        self._turns: List[Tuple[str, str]] = []
        self._max_turns = max_turns
        self._summary: str = ""

    def add(self, query: str, answer: str) -> None:
        self._turns.append((query, answer))
        if len(self._turns) > self._max_turns:
            self._compress()

    def _compress(self) -> None:
        """Compress the oldest turns into a rolling summary."""
        # Keep the most recent half of the turns verbatim.
        keep = len(self._turns) // 2
        old_turns = self._turns[: len(self._turns) - keep]
        self._turns = self._turns[len(self._turns) - keep :]
        # Build a summary of the compressed turns.
        parts = []
        for q, a in old_turns:
            q_short = q[: 200]
            a_short = a[: 200]
            parts.append(f"Q: {q_short}\nA: {a_short}")
        old_summary = "\n".join(parts)
        if self._summary:
            self._summary = f"{self._summary}\n\n{old_summary}"
        else:
            self._summary = old_summary

    def messages(self) -> List[Any]:
        msgs: List[Any] = []
        if self._summary:
            msgs.append(HumanMessage(content=f"(Earlier conversation summary):\n{self._summary}"))
        for q, a in self._turns:
            msgs.append(HumanMessage(content=q))
            msgs.append(AIMessage(content=a))
        return msgs

    def clear(self) -> None:
        self._turns.clear()
        self._summary = ""

    def __len__(self) -> int:
        return len(self._turns)


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
    existing = tools.vectorstore.similarity_search(
        text, k=100
    )
    filtered: List[Document] = []
    for doc in docs:
        if _is_duplicate(doc, existing, tools.embedder):
            continue
        filtered.append(doc)
    if not filtered:
        return []
    tools.vectorstore.add_documents(filtered)
    if hasattr(tools.vectorstore, "persist"):
        tools.vectorstore.persist()
    return filtered


def _is_duplicate(
    doc: Document,
    existing: List[Document],
    embedder: Embeddings,
) -> bool:
    """Return True if doc's content is a near-duplicate of an
    existing chunk (cosine similarity >= threshold).
    """
    try:
        new_emb = embedder.embed_query(doc.page_content)
    except Exception:
        return False
    for ex in existing:
        try:
            ex_emb = embedder.embed_query(ex.page_content)
        except Exception:
            continue
        sim = cosine_similarity(new_emb, ex_emb)
        if sim >= config.MEMORY_SIMILARITY_THRESHOLD:
            return True
    return False


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
