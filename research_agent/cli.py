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
    "I have a budget of $2,500 to build a local AI workstation for running 30B-70B quantized language models. Research the optimal CPU, GPU, RAM, motherboard, SSD, cooling, and power supply. Compare at least three complete builds, estimate real-world inference speeds, upgrade paths, energy consumption, and cost per token generated. Recommend the best value configuration for the next five years.",
    "What are the new open source llms released this week?",
    "Design a local-first AI coding assistant capable of serving 100 concurrent users on a single RTX 4090. Compare vLLM, SGLang, TensorRT-LLM, llama.cpp, and ExLlamaV2, covering KV cache management, continuous batching, speculative decoding, memory usage, latency, throughput, and deployment architecture. End with a production-ready recommendation and implementation roadmap.",
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
    print(logutil.header("Tiny Deep Researcher"))
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
        if not result.get("streamed"):
            # Streamed answers were already printed live during the run.
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