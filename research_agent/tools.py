from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
try:  # prefer new package name
    from ddgs import DDGS
except ImportError:  # fallback to legacy package
    from duckduckgo_search import DDGS
try:  # langchain 0.3+
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # fallback for older langchain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter
try:  # prefer modern packages
    from langchain_ollama import OllamaEmbeddings, ChatOllama
except ImportError:  # fallback to community if older
    from langchain_community.embeddings import OllamaEmbeddings  # type: ignore
    from langchain_ollama import ChatOllama
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_core.language_models.chat_models import BaseChatModel

from . import config


@dataclass
class ResearchTools:
    llm: BaseChatModel
    embedder: OllamaEmbeddings
    vectorstore: Chroma
    text_splitter: RecursiveCharacterTextSplitter


def create_llm(
    model: str = config.LLM_MODEL,
    temperature: float = config.LLM_TEMPERATURE,
    max_tokens: int = config.LLM_MAX_TOKENS,
) -> BaseChatModel:
    return ChatOllama(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        num_thread=4,
        verbose=False,
    )


def create_embedder(model: str = config.EMBED_MODEL) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model)


def create_vectorstore(embedder: OllamaEmbeddings) -> Chroma:
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


def run_ddg_search(query: str, max_results: int) -> List[Dict[str, str]]:
    with DDGS() as search:
        results = list(search.text(query, max_results=max_results))
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
            }
        )
    return cleaned


def fetch_url(url: str, timeout: int = config.REQUEST_TIMEOUT) -> Optional[Tuple[str, str]]:
    headers = {
        "User-Agent": "lite-research-agent/0.1 (+https://github.com/)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
    except Exception:
        return None
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = " ".join(text.split())
    if not text:
        return None
    if len(text) > config.MAX_PAGE_CHARS:
        text = text[: config.MAX_PAGE_CHARS]
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    return title, text


def timestamp() -> str:
    return datetime.utcnow().isoformat()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
