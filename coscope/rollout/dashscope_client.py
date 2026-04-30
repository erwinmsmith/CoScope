"""
DashScope (Aliyun Qwen) LLM client — OpenAI-compatible mode.

Implements the LLMClient Protocol defined in `coscope.core.interfaces`.
Reads API key from env `DASHSCOPE_API_KEY` or from constructor.

Defaults to `qwen-plus`. Supported models include:
  - qwen-turbo  (cheapest, fastest)
  - qwen-plus   (recommended, balanced)
  - qwen-max    (highest quality)
"""

from __future__ import annotations

import logging
import os
import time
import unicodedata
from typing import List, Optional

from openai import OpenAI

from coscope.core.interfaces import LLMClient, LLMResponse


logger = logging.getLogger(__name__)


_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_ASCII_FALLBACK_MAP = {
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u00a0": " ",
    "鈥": "-",
    "揂": "A",
    "揘": "N",
    "揅": "C",
}


def _ascii_safe_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = unicodedata.normalize("NFKC", text)
    for src, dst in _ASCII_FALLBACK_MAP.items():
        normalized = normalized.replace(src, dst)
    # Keep newlines/tabs; drop other non-ascii characters only as a last resort.
    cleaned = normalized.encode("ascii", "ignore").decode("ascii")
    return cleaned


def _needs_ascii_fallback(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "ascii" in msg and "encode" in msg


class DashScopeClient(LLMClient):
    """Qwen models via DashScope OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: Optional[str] = None,
        base_url: str = _DASHSCOPE_BASE_URL,
        max_tokens: int = 32768,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError(
                "DASHSCOPE_API_KEY not set. Pass api_key= or export env var."
            )
        self._client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
        self.model = model
        self.name = f"dashscope:{model}"
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        stop: Optional[List[str]] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> LLMResponse:
        orig_system = system
        orig_prompt = prompt
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        used_ascii_fallback = False
        if stop:
            params["stop"] = stop
        if seed is not None:
            params["seed"] = int(seed)
        if max_new_tokens is not None:
            params["max_tokens"] = int(max_new_tokens)

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(**params)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                choice = resp.choices[0]
                text = choice.message.content or ""
                usage = resp.usage
                return LLMResponse(
                    text=text,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    finish_reason=choice.finish_reason or "",
                    model=self.model,
                    latency_ms=dt_ms,
                    raw={"id": resp.id},
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if not used_ascii_fallback and _needs_ascii_fallback(exc):
                    used_ascii_fallback = True
                    sanitized_system = _ascii_safe_text(orig_system)
                    sanitized_prompt = _ascii_safe_text(orig_prompt) or ""
                    logger.warning(
                        "DashScope request hit ascii encoding issue; retrying with sanitized ASCII-compatible text."
                    )
                    messages = []
                    if sanitized_system:
                        messages.append({"role": "system", "content": sanitized_system})
                    messages.append({"role": "user", "content": sanitized_prompt})
                    params["messages"] = messages
                wait = min(2 ** attempt, 8)
                logger.warning(
                    "DashScope generate attempt %d/%d failed: %s; retry in %ds",
                    attempt + 1, self.max_retries + 1, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"DashScope generate failed after retries: {last_err}")

    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: Optional[str] = None,
        stop: Optional[List[str]] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[LLMResponse]:
        # DashScope has no native batch API; fall back to sequential calls.
        # Parallelism can be added at the caller side via ThreadPoolExecutor.
        out: List[LLMResponse] = []
        for p in prompts:
            out.append(
                self.generate(
                    p,
                    system=system,
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    max_new_tokens=max_new_tokens,
                )
            )
        return out
