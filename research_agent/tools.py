from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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

from . import config


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


def build_tools() -> ResearchTools:
    embedder = create_embedder()
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
    with DDGS() as search:
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
        downloaded = trafilatura.fetch_url(url, timeout=timeout)
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
