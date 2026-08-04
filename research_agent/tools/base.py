from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, cast

from langsmith import traceable
try:  # prefer new package name
    from ddgs import DDGS
except ImportError:  # fallback to legacy package
    from duckduckgo_search import DDGS
try:  # langchain 0.3+
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # fallback for older langchain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter
try:  # OpenAI-compatible chat endpoint (mlx_lm.server, llama.cpp, vLLM, etc.)
    from langchain_openai import ChatOpenAI
except ImportError:  # fallback to community if older
    from langchain_community.chat_models import ChatOpenAI  # type: ignore
try:  # local in-process embeddings (no server required)
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # fallback to community if older
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
from langchain_core.embeddings import Embeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_core.language_models.chat_models import BaseChatModel

import trafilatura

from .. import config


_ddg_session: Optional[DDGS] = None
_ddg_session_lock = threading.Lock()


def _get_ddg_session() -> DDGS:
    """Return a thread-local DDGS session, creating one if needed."""
    global _ddg_session
    if _ddg_session is None:
        with _ddg_session_lock:
            if _ddg_session is None:
                _ddg_session = DDGS()
    return _ddg_session


@dataclass
class ResearchTools:
    llm: BaseChatModel
    embedder: Embeddings
    vectorstore: Chroma
    text_splitter: RecursiveCharacterTextSplitter


def create_llm(
    model: str = config.LLM_MODEL,
    temperature: float = config.LLM_TEMPERATURE,
    max_tokens: int = config.LLM_MAX_TOKENS,
) -> BaseChatModel:
    return ChatOpenAI(
        model=model,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=config.LLM_TIMEOUT,
        max_retries=2,
    )


def create_embedder(model: str = config.EMBED_MODEL) -> Embeddings:
    return HuggingFaceEmbeddings(model_name=model)


class CachingEmbeddings(Embeddings):
    """Wrap an embedder and cache embedding results to avoid recomputation."""

    def __init__(self, embedder: Embeddings, max_size: int = config.EMBED_CACHE_MAX) -> None:
        self._embedder = embedder
        self._max_size = max_size
        self._cache: Dict[str, List[float]] = {}

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        uncached: List[str] = []
        indices: List[int] = []
        results: List[Optional[List[float]]] = [None] * len(texts)
        for i, text in enumerate(texts):
            key = self._key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached.append(text)
                indices.append(i)
        if uncached:
            cached_results = self._embedder.embed_documents(uncached)
            for idx, emb in zip(indices, cached_results):
                k = self._key(texts[idx])
                self._cache[k] = emb
                if len(self._cache) > self._max_size:
                    self._cache.pop(next(iter(self._cache)))
                results[idx] = emb
        return cast(List[List[float]], results)

    def embed_query(self, text: str) -> List[float]:
        key = self._key(text)
        if key in self._cache:
            return self._cache[key]
        result = self._embedder.embed_query(text)
        self._cache[key] = result
        if len(self._cache) > self._max_size:
            self._cache.pop(next(iter(self._cache)))
        return result


def create_vectorstore(embedder: Embeddings) -> Chroma:
    config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        embedding_function=embedder,
        persist_directory=str(config.MEMORY_DIR),
        collection_name="research_memory",
    )


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )


def _messages_to_text(messages: Any) -> str:
    parts: List[str] = []
    for m in messages:
        content = getattr(m, "content", m)
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in content
            )
        parts.append(str(content))
    return "\n".join(parts)


def count_tokens(embedder: Embeddings, text: str) -> int:
    """Estimate token count for a prompt using the embedder's tokenizer when
    available, falling back to a ~4 chars/token heuristic."""
    text = text or ""
    try:
        if hasattr(embedder, "encode"):
            return len(embedder.encode(text))  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        if hasattr(embedder, "tokenizer") and embedder.tokenizer is not None:  # type: ignore[attr-defined]
            return len(embedder.tokenizer.tokenize(text))  # type: ignore[attr-defined]
    except Exception:
        pass
    return max(1, len(text) // 4)


def build_tools() -> ResearchTools:
    embedder = CachingEmbeddings(create_embedder())
    tools = ResearchTools(
        llm=create_llm(),
        embedder=embedder,
        vectorstore=create_vectorstore(embedder),
        text_splitter=create_text_splitter(),
    )
    return tools


def _normalize_date_str(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).isoformat()
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).isoformat()
        except Exception:
            continue
    return None


def _inject_date_filters(query: str, since_days: int = 0, date_from: Optional[str] = None, date_to: Optional[str] = None) -> str:
    tokens = []
    if since_days > 0 and not date_from:
        start = (datetime.utcnow() - timedelta(days=since_days)).date().isoformat()
        tokens.append(f"after:{start}")
    if date_from:
        tokens.append(f"after:{date_from}")
    if date_to:
        tokens.append(f"before:{date_to}")
    if not tokens:
        return query
    return f"{query} {' '.join(tokens)}"


@traceable(run_type="tool", name="DuckDuckGo Search")
def run_ddg_search(
    query: str,
    max_results: int,
    since_days: int = 0,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    time_limit: Optional[str] = None,
) -> List[Dict[str, str]]:
    search_query = _inject_date_filters(query, since_days=since_days, date_from=date_from, date_to=date_to)
    search = _get_ddg_session()
    if time_limit:
        results = list(
            search.news(
                safesearch="off",
                keywords=search_query,
                timelimit=time_limit,
                max_results=max_results,
            )
        )
    else:
        results = list(search.text(search_query, max_results=max_results))
    cleaned = []
    for item in results:
        url = item.get("href") or item.get("url")
        if not url:
            continue
        cleaned.append(
            {
                "url": url,
                "title": item.get("title", "").strip(),
                "snippet": item.get("body") or item.get("snippet", ""),
                "published_at": _normalize_date_str(
                    item.get("date")
                    or item.get("published")
                    or item.get("published_at")
                    or item.get("year")
                ),
            }
        )
    return cleaned


@traceable(run_type="retriever", name="Fetch URL")
def fetch_url(url: str, timeout: int = config.REQUEST_TIMEOUT) -> Optional[Tuple[str, str]]:
    try:
        # trafilatura >= 2.1 has no `timeout` kwarg on fetch_url; the download
        # timeout is set via the config object instead.
        dl_config = trafilatura.settings.use_config()
        dl_config.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(timeout))
        downloaded = trafilatura.fetch_url(url, config=dl_config)
    except Exception:
        return None
    if not downloaded:
        return None
    text = trafilatura.extract(
        downloaded,
        output_format="markdown",
        favor_precision=True,
        include_comments=False,
        include_tables=False,
    )
    if not text:
        return None
    if len(text) > config.MAX_PAGE_CHARS:
        text = text[: config.MAX_PAGE_CHARS]
    metadata = trafilatura.extract_metadata(downloaded)
    title = (metadata.title.strip() if metadata and metadata.title else None) or url
    return title, text


def timestamp() -> str:
    return datetime.utcnow().isoformat()


def _extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of the first JSON object from an LLM response.

    Handles the common cases where the model wraps JSON in markdown code
    fences (```json ... ```) or emits prose around it. Returns None if no
    balanced JSON object can be found.
    """
    if not text:
        return None
    cleaned = text.strip()
    # Strip markdown code fences if present.
    if cleaned.startswith("```"):
        # drop the opening fence line (``` or ```json)
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3]
        cleaned = cleaned.strip()
    # Try direct parse first.
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Find the first balanced {...} span.
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
