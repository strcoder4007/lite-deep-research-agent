from __future__ import annotations

import re
from typing import Optional

from . import config, logutil


def _parse_model_size(model_name: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", model_name)
    if match:
        return float(match.group(1))
    return None


def _parse_quantization(model_name: str) -> float:
    if "4bit" in model_name.lower() or "4-bit" in model_name.lower():
        return 0.5
    if "8bit" in model_name.lower() or "8-bit" in model_name.lower():
        return 1.0
    if "16bit" in model_name.lower() or "16-bit" in model_name.lower():
        return 2.0
    return 2.0


def _estimate_weight_gb(model_name: str) -> float:
    params_b = _parse_model_size(model_name)
    if params_b is None:
        return 0.0
    bytes_per_param = _parse_quantization(model_name)
    return params_b * bytes_per_param


def _estimate_kv_cache_mb(prompt_tokens: int) -> float:
    if prompt_tokens <= 0:
        return 0.0
    per_token_mb = 1.25
    return prompt_tokens * per_token_mb / 1000.0


def step_footer(
    step: int,
    llm_time: float,
    tool_time: float,
    fetch_time: float,
    total_time: float,
    tool_count: int,
    fetch_count: int,
    prompt_tokens: int,
    context_total: int,
    model_name: Optional[str] = None,
) -> str:
    """Boxed step summary: timing row + memory/context row."""
    if model_name is None:
        model_name = config.LLM_MODEL

    weight_gb = _estimate_weight_gb(model_name)
    kv_mb = _estimate_kv_cache_mb(prompt_tokens)

    row = (
        logutil.cyan("LLM") + logutil.dim(":") + f" {llm_time:>6.1f}s"
        + "   " + logutil.yellow("tools") + logutil.dim(":") + f" {tool_time:>6.1f}s"
        + "   " + logutil.green("fetch") + logutil.dim(":") + f" {fetch_time:>6.1f}s"
        + "   " + logutil.bold("total") + logutil.dim(":") + f" {total_time:>6.1f}s"
        + logutil.dim(f"   · {tool_count} calls + {fetch_count} fetches")
    )
    mem = (
        logutil.dim("weights") + logutil.dim(":") + f" {weight_gb:.1f} GB"
        + "   " + logutil.dim("kv cache") + logutil.dim(":") + f" {kv_mb:.1f} MB"
        + "   " + logutil.dim("context") + logutil.dim(":") + " "
        + logutil.context_bar(prompt_tokens, context_total, width=12)
    )
    return logutil.box([row, mem], f" step {step} ")