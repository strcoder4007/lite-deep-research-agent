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


def print_memory_stats(
    step: int,
    prompt_tokens: int,
    context_total: int,
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = config.LLM_MODEL

    weight_gb = _estimate_weight_gb(model_name)
    kv_mb = _estimate_kv_cache_mb(prompt_tokens)
    ctx_gb = prompt_tokens * 1250 / (1024 ** 3)

    parts = [
        logutil.dim("  ── memory ──"),
        logutil.cyan(f"weights: {weight_gb:.1f} GB"),
        logutil.dim(" ·"),
        logutil.yellow(f"kv cache: {kv_mb:.1f} MB"),
        logutil.dim(" ·"),
        logutil.blue(f"context: {logutil.context_bar(prompt_tokens, context_total, width=10)}"),
        logutil.dim(f" ({ctx_gb:.2f} GB)"),
    ]
    logutil._print("".join(parts))