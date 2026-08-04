from __future__ import annotations

from typing import Optional

from . import tool

# Sentinel set by final_answer(); the agent loop reads and clears it.
FINAL_ANSWER: Optional[str] = None


@tool
def final_answer(answer: str) -> str:
    """Finish the task and return the final answer to the user."""
    global FINAL_ANSWER
    FINAL_ANSWER = answer
    return answer


def take_final_answer() -> Optional[str]:
    """Return and clear the sentinel set by final_answer()."""
    global FINAL_ANSWER
    answer, FINAL_ANSWER = FINAL_ANSWER, None
    return answer
