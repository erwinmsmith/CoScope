"""Normalize usage objects returned by OpenAI-compatible SDKs."""

from __future__ import annotations

from typing import Any


def usage_values(usage: Any) -> dict[str, int]:
    if usage is None:
        payload: dict[str, Any] = {}
    elif hasattr(usage, "model_dump"):
        payload = usage.model_dump()
    elif isinstance(usage, dict):
        payload = usage
    else:
        payload = vars(usage)

    completion_details = payload.get("completion_tokens_details") or {}
    prompt_details = payload.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": int(payload.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(payload.get("completion_tokens", 0) or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens", 0) or 0),
        "cached_tokens": int(prompt_details.get("cached_tokens", 0) or 0),
        "total_tokens": int(payload.get("total_tokens", 0) or 0),
    }
