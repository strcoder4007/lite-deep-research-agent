from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from .agent import AdvancedResearchAgent


EXAMPLE_QUERIES = [
    "How are drought conditions evolving in California in 2024 and what mitigation steps are being taken?",
    "What are the latest techniques for reducing LLM hallucinations with grounding?",
    "State of small modular reactor deployments globally in 2024",
    "Tell me about the new toyota GR GT, epxected price, release date, specs"
]


def _choose_query() -> str:
    print("Select a query or type your own:")
    print("0) Custom query")
    for idx, q in enumerate(EXAMPLE_QUERIES, start=1):
        print(f"{idx}) {q}")
    choice = input("> ").strip()
    if choice and choice.isdigit() and 0 <= int(choice) <= len(EXAMPLE_QUERIES):
        idx = int(choice)
        if idx == 0:
            return input("Enter your query: ").strip()
        return EXAMPLE_QUERIES[idx - 1]
    return choice


def main() -> int:
    agent = AdvancedResearchAgent()
    try:
        query = _choose_query()
        if not query:
            print("No query provided. Exiting.")
            return 1
        result = agent.research(query=query, verbose=True)
    except KeyboardInterrupt:
        print("\nAborted.")
        return 1
    report = result.get("report", "")
    print("\n=== Research Report ===\n")
    print(report)
    print("\n=== Sources ===")
    for src in result.get("sources", []):
        print(f"- {src}")
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    path = Path(f"report_{digest}.txt")
    path.write_text(report)
    print(f"\nSaved report to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
