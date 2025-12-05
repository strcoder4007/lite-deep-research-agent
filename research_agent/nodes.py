from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from . import config
from .memory import add_to_memory, query_memory
from .state import ResearchState, append_error, append_message
from .tools import ResearchTools, cosine_similarity, fetch_url, run_ddg_search, timestamp

NO_THINK_FLAG = "/no_think"


def _with_no_think(text: str) -> str:
    return f"{text}\n{NO_THINK_FLAG}"


def _parse_plan(raw_text: str) -> Tuple[Dict[str, Any], List[str]]:
    notes: List[str] = []
    try:
        parsed = yaml.safe_load(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("Plan is not a mapping")
    except Exception as exc:
        notes.append(f"Plan parsing failed: {exc}")
        return {}, notes
    normalized = {
        "SEARCH_QUERIES": parsed.get("SEARCH_QUERIES") or parsed.get("search_queries"),
        "KEY_ASPECTS": parsed.get("KEY_ASPECTS") or parsed.get("key_aspects"),
        "GAPS_TO_ADDRESS": parsed.get("GAPS_TO_ADDRESS") or parsed.get("gaps_to_address"),
    }
    for key in list(normalized.keys()):
        val = normalized[key]
        if isinstance(val, list):
            normalized[key] = [str(v).strip() for v in val if v]
        else:
            normalized[key] = []
    return normalized, notes


def _parse_date_str(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    try:
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        return datetime.fromisoformat(cleaned)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, fmt)
        except Exception:
            continue
    return None


def plan_node(state: ResearchState, tools: ResearchTools) -> Dict[str, Any]:
    query = state["query"]
    today = datetime.utcnow().date().isoformat()
    messages = [
        SystemMessage(
            content=(
                "You are a research planner. Create a concise plan for a web research agent.\n"
                "Return ONLY structured YAML with keys SEARCH_QUERIES, KEY_ASPECTS, GAPS_TO_ADDRESS.\n"
                "Provide 4-5 search queries, key aspects to cover, and gaps/questions to verify.\n"
                f"Current date (UTC): {today}."
            )
        ),
        HumanMessage(content=(f"User query: {query}\nCurrent date (UTC): {today}")),
    ]
    raw = tools.llm.invoke(messages).content
    plan, notes = _parse_plan(raw)
    if not plan.get("SEARCH_QUERIES"):
        plan["SEARCH_QUERIES"] = [query]
        append_message(state, "Plan parsing failed; fell back to raw query.")
    append_message(state, f"Plan generated with {len(plan.get('SEARCH_QUERIES', []))} queries.")
    for note in notes:
        append_error(state, note)
    return {
        "research_plan": plan,
        "search_queries": plan.get("SEARCH_QUERIES", [query]),
        "plan_gaps": plan.get("GAPS_TO_ADDRESS", []),
        "messages": state.get("messages", []),
        "errors": state.get("errors", []),
    }


def _dedupe_results(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen_urls = set()
    seen_hosts = set()
    deduped = []
    for item in results:
        url = item.get("url")
        if not url or url in seen_urls:
            continue
        host = urlparse(url).hostname
        if config.SEARCH_RERANK_USE_HOST_DEDUP and host and host in seen_hosts:
            continue
        seen_urls.add(url)
        if host:
            seen_hosts.add(host)
        deduped.append(item)
    return deduped


def _rerank_results(
    tools: ResearchTools, query: str, candidates: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    try:
        q_emb = tools.embedder.embed_query(query)
        texts = [f"{c.get('title','')} {c.get('snippet','')}" for c in candidates]
        doc_embs = tools.embedder.embed_documents(texts)
        scored = []
        now = datetime.utcnow()
        for cand, emb in zip(candidates, doc_embs):
            score = cosine_similarity(q_emb, emb)
            cand_copy = dict(cand)
            recency_bonus = 0.0
            published_at = cand_copy.get("published_at")
            if published_at:
                dt = _parse_date_str(published_at)
                if dt:
                    age_days = max((now - dt).total_seconds() / 86400.0, 0.1)
                    recency_bonus = config.SEARCH_RECENCY_BOOST / (1.0 + age_days / 7.0)
            cand_copy["score"] = score + recency_bonus
            scored.append(cand_copy)
        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        return scored[: config.SEARCH_RERANK_TOP_N]
    except Exception as exc:
        # fallback: keep order
        return candidates[: config.SEARCH_RERANK_TOP_N]


def search_node(state: ResearchState, tools: ResearchTools) -> Dict[str, Any]:
    query = state["query"]
    all_results: List[Dict[str, str]] = []
    for sq in state.get("search_queries", []):
        results = run_ddg_search(
            sq,
            max_results=config.SEARCH_RESULTS_PER_QUERY,
            since_days=config.SEARCH_SINCE_DAYS,
            date_from=config.SEARCH_DATE_FROM,
            date_to=config.SEARCH_DATE_TO,
            time_limit=config.SEARCH_TIME_LIMIT,
        )
        for r in results:
            r["query"] = sq
        all_results.extend(results)
    deduped = _dedupe_results(all_results)
    reranked = _rerank_results(tools, query, deduped)
    append_message(state, f"Search fetched {len(deduped)} unique results, reranked to {len(reranked)}.")
    return {"search_results": reranked, "messages": state.get("messages", [])}


def fetch_node(state: ResearchState, tools: ResearchTools) -> Dict[str, Any]:
    fetched: List[Dict[str, Any]] = []
    to_visit = state.get("search_results", [])[: config.FETCH_LIMIT]
    for item in to_visit:
        url = item["url"]
        resp = fetch_url(url)
        if not resp:
            continue
        title, text = resp
        metadata = {
            "url": url,
            "title": title,
            "query": item.get("query") or state.get("query"),
            "timestamp": timestamp(),
        }
        add_to_memory(tools, text, metadata)
        fetched.append({"url": url, "title": title, "text": text, "metadata": metadata})
    append_message(state, f"Fetched {len(fetched)} pages.")
    return {"fetched_content": fetched, "messages": state.get("messages", [])}


def analyze_node(state: ResearchState, tools: ResearchTools) -> Dict[str, Any]:
    query = state["query"]
    facts: List[str] = []
    for doc in state.get("fetched_content", []):
        snippet = doc["text"][:1500]
        prompt = [
            SystemMessage(
                content=_with_no_think(
                    "Extract concise, factual statements relevant to the user query. "
                    "Prefix with source title in brackets. Keep it short."
                )
            ),
            HumanMessage(
                content=_with_no_think(
                    f"User query: {query}\n"
                    f"Source: {doc['title']}\n"
                    f"Content:\n{snippet}"
                )
            ),
        ]
        try:
            response = tools.llm.invoke(prompt).content
            lines = [line.strip(" -") for line in response.splitlines() if line.strip()]
            for line in lines:
                if line:
                    facts.append(line)
        except Exception as exc:
            append_error(state, f"Analysis failed for {doc.get('url')}: {exc}")
    append_message(state, f"Extracted {len(facts)} facts.")
    return {"extracted_facts": facts, "messages": state.get("messages", []), "errors": state.get("errors", [])}


def memory_node(state: ResearchState, tools: ResearchTools) -> Dict[str, Any]:
    aspects = state.get("plan_gaps") or state.get("research_plan", {}).get("KEY_ASPECTS", [])
    retrieved = query_memory(
        tools,
        query=state["query"],
        aspects=aspects,
        top_k=config.MEMORY_TOP_K,
        threshold=config.MEMORY_SIMILARITY_THRESHOLD,
    )
    append_message(state, f"Retrieved {len(retrieved)} memory chunks.")
    return {"relevant_memory": retrieved, "messages": state.get("messages", [])}


def synthesize_node(state: ResearchState, tools: ResearchTools) -> Dict[str, Any]:
    facts = state.get("extracted_facts", [])
    memory_bits = state.get("relevant_memory", [])
    sources = [item.get("url") for item in state.get("fetched_content", []) if item.get("url")]
    memory_sources = [m["metadata"].get("url") for m in memory_bits if m.get("metadata", {}).get("url")]
    combined_sources = list(dict.fromkeys([s for s in sources + memory_sources if s]))

    fact_block = "\n".join(f"- {f}" for f in facts)
    memory_block = "\n".join(
        f"- {m.get('metadata', {}).get('title','memory')} :: {m.get('content')}" for m in memory_bits
    )
    prompt = [
        SystemMessage(
            content=(
                "You are a research synthesis assistant. Write a grounded report using ONLY the provided facts "
                "and memory notes. Do not speculate. Include inline source markers like [Source 1].\n"
                "Structure: Executive Summary; Numbered Findings; Analysis; Conclusion; Notes.\n"
                "Temperature low. Keep concise."
            )
        ),
        HumanMessage(
            content=(
                f"User query: {state['query']}\n"
                f"Extracted facts:\n{fact_block}\n\n"
                f"Relevant memory:\n{memory_block}\n"
                f"Sources list:\n{combined_sources}"
            )
        ),
    ]
    synth_llm = tools.llm.bind()
    response = synth_llm.invoke(prompt).content
    return {
        "final_answer": response,
        "sources": combined_sources,
    }


def should_continue_node(state: ResearchState) -> Dict[str, Any]:
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", config.MAX_ITERATIONS)
    fetched = len(state.get("fetched_content", []))
    facts = len(state.get("extracted_facts", []))
    gaps = state.get("plan_gaps") or []
    next_step = "synthesize"
    search_queries = state.get("search_queries", [])
    if iteration >= max_iter:
        append_message(state, "Reached max iterations; moving to synthesis.")
    elif fetched < config.MIN_FETCHED_FOR_STOP or facts < config.MIN_FACTS_FOR_STOP or gaps:
        append_message(state, "Coverage insufficient; continuing search.")
        next_step = "search"
        base_query = state.get("query", "")
        new_queries = []
        for gap in gaps:
            q = f"{base_query} {gap}"
            new_queries.append(q)
        if not new_queries and base_query:
            new_queries.append(base_query)
        combined = list(dict.fromkeys(search_queries + new_queries))
        search_queries = combined
        iteration += 1
    else:
        append_message(state, "Facts sufficient; move to memory+synthesis.")
    return {
        "next_step": next_step,
        "search_queries": search_queries,
        "iteration": iteration,
        "messages": state.get("messages", []),
    }
