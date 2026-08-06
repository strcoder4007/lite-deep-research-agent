from __future__ import annotations

import gzip
import hashlib
import json
import math
import threading
import time
import urllib.parse
import urllib.request
import zlib
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

_fetch_cache: Dict[str, Tuple[float, Optional[Tuple[str, str]]]] = {}
_fetch_cache_lock = threading.Lock()


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
        # Ask the server for token usage on streamed responses
        # (stream_options: {include_usage: true}); without it the agent's
        # token stats, budget guard, and compression trigger never fire.
        stream_usage=True,
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


_tools: Optional[ResearchTools] = None
_tools_lock = threading.Lock()


def build_tools() -> ResearchTools:
    global _tools
    if _tools is not None:
        return _tools
    with _tools_lock:
        if _tools is not None:
            return _tools
        embedder = CachingEmbeddings(create_embedder())
        _tools = ResearchTools(
            llm=create_llm(),
            embedder=embedder,
            vectorstore=create_vectorstore(embedder),
            text_splitter=create_text_splitter(),
        )
        return _tools


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


def _days_to_timelimit(since_days: int) -> Optional[str]:
    """Map a `since_days` window to the DDG timelimit filter it implies."""
    if since_days <= 0:
        return None
    if since_days <= 1:
        return "d"
    if since_days <= 7:
        return "w"
    if since_days <= 30:
        return "m"
    return "y"


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
    # A `since_days` window only takes effect through the real DDG time
    # filter (timelimit); an "after:" operator in the query text is
    # ignored by DDG and leaves the search unfiltered (stale results).
    effective_limit = time_limit or _days_to_timelimit(since_days)
    search_query = _inject_date_filters(
        query,
        since_days=0 if effective_limit else since_days,
        date_from=date_from,
        date_to=date_to,
    )
    search = _get_ddg_session()
    if effective_limit:
        results = list(
            search.text(
                search_query,
                safesearch="off",
                timelimit=effective_limit,
                max_results=max_results,
            )
        )
    else:
        results = list(search.text(search_query, safesearch="off", max_results=max_results))
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


def _parse_reddit_json(raw: str, url: str) -> Tuple[str, str]:
    """Parse reddit's ``.json`` response into (text, title)."""
    data = json.loads(raw)
    parts: List[str] = []
    title = url
    if isinstance(data, list):
        items = [i for i in data if isinstance(i, dict)]
    elif isinstance(data, dict):
        items = [data]
    else:
        items = []
    for item in items:
        children = (item.get("data") or {}).get("children") or []
        for child in children:
            d = (child.get("data") or {}) if isinstance(child, dict) else {}
            if d.get("title"):
                title = d["title"]
            text = d.get("selftext") or d.get("body")
            if text:
                parts.append(text)
    return "\n\n".join(p for p in parts if p), title


try:  # optional: brotli ("br") decompression; gzip/deflate always work
    import brotli  # type: ignore

    _ACCEPT_ENCODING = "gzip, deflate, br"
except ImportError:
    brotli = None  # type: ignore
    _ACCEPT_ENCODING = "gzip, deflate"

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _download_html(url: str, timeout: int) -> str:
    """Download a URL and return the decoded HTML text.

    trafilatura's own fetch_url returns the compressed body undecoded for
    some sites, so extraction fails with "page content could not be
    parsed".  Download ourselves with a browser UA and decompress
    gzip/br/deflate before handing the HTML to trafilatura.
    """
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _BROWSER_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": _ACCEPT_ENCODING,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    enc = (resp.headers.get("Content-Encoding") or "").lower()
    if "gzip" in enc:
        raw = gzip.decompress(raw)
    elif "br" in enc and brotli is not None:
        raw = brotli.decompress(raw)
    elif "deflate" in enc:
        raw = zlib.decompress(raw)
    return raw.decode("utf-8", errors="replace")


@traceable(run_type="retriever", name="Fetch URL")
def fetch_url(url: str, timeout: int = config.REQUEST_TIMEOUT) -> Optional[Tuple[str, str]]:
    if config.FETCH_CACHE_TTL > 0:
        with _fetch_cache_lock:
            hit = _fetch_cache.get(url)
            if hit and time.time() - hit[0] < config.FETCH_CACHE_TTL:
                return hit[1]

    # Reddit blocks scrapers; use the JSON endpoint instead.
    host = urllib.parse.urlparse(url).netloc.lower()
    if host.endswith("reddit.com"):
        try:
            sep = "&" if "?" in url else "?"
            json_url = f"{url}{sep}.json"
            with urllib.request.urlopen(json_url, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            text, title = _parse_reddit_json(raw, url)
            if not text:
                return ("error", "extraction failed (reddit JSON could not be parsed)")
            if len(text) > config.MAX_PAGE_CHARS:
                text = text[: config.MAX_PAGE_CHARS]
            return (title, text)
        except Exception as exc:
            return ("error", f"network error: {type(exc).__name__}: {exc}")

    try:
        html = _download_html(url, timeout)
    except Exception as exc:
        return ("error", f"network error: {type(exc).__name__}: {exc}")

    if not html or not html.strip():
        return ("error", "no content downloaded (empty response or blocked by server)")

    text = trafilatura.extract(
        html,
        output_format="markdown",
        favor_precision=False,
        include_comments=False,
        include_tables=False,
    )
    if not text:
        return ("error", "extraction failed (page content could not be parsed)")

    if len(text) > config.MAX_PAGE_CHARS:
        text = text[: config.MAX_PAGE_CHARS]
    metadata = trafilatura.extract_metadata(html)
    title = (metadata.title.strip() if metadata and metadata.title else None) or url
    return (title, text)


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
