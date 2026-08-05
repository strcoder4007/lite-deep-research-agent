from __future__ import annotations

import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from . import logutil
from .agent import run
from .memory import ConversationMemory


EXAMPLE_QUERIES = [
    "Tell me all the good tv show episodes released in the last week.",
    "What are the new open source llms released this week?",
    "Tell me about the new Toyota GR GT, expected price, release date, specs",
]


def _choose_query() -> str:
    print(logutil.header("Select a query or type your own:"))
    print("0) Custom query")
    for idx, q in enumerate(EXAMPLE_QUERIES, start=1):
        print(f"  {idx}) {logutil.dim(q)}")
    choice = input(logutil.user("> ")).strip()
    if choice and choice.isdigit() and 0 <= int(choice) <= len(EXAMPLE_QUERIES):
        idx = int(choice)
        if idx == 0:
            return input(logutil.user("Enter your query: ")).strip()
        return EXAMPLE_QUERIES[idx - 1]
    return choice


def main() -> int:
    history = ConversationMemory()
    print(logutil.header("Lite Deep Research Agent"))
    print(logutil.dim("  conversational mode — type 'quit' to exit"))
    print(logutil.separator())
    turn = 0
    while True:
        try:
            if turn == 0:
                query = _choose_query()
            else:
                query = input(logutil.user("> ")).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            return 0
        if not query or query.lower() in ("quit", "exit", "q"):
            print(logutil.dim("Goodbye."))
            return 0
        turn += 1
        result = run(query=query, verbose=True, history=history)
        report = result.get("answer", "")
        print()
        print(logutil.agent(""))
        print(logutil.cyan(report))
        if result.get("errors"):
            print()
            print(logutil.error(f"Errors ({len(result['errors'])}):"))
            for err in result["errors"]:
                print(f"  {logutil.error(err)}")
        print(logutil.separator())


if __name__ == "__main__":
    sys.exit(main())