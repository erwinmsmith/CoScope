"""DeepSeek chat adapter using its OpenAI-compatible API."""

from __future__ import annotations

from typing import Any, cast

from openai import OpenAI

from coscope.adapters.llm.base import LLMOutput
from coscope.adapters.usage import usage_values
from coscope.config import LLMSettings
from coscope.core.usage import UsageLedger


class DeepSeekLLM:
    def __init__(
        self,
        settings: LLMSettings,
        *,
        client: Any | None = None,
        usage_ledger: UsageLedger | None = None,
    ):
        settings.validate()
        self.settings = settings
        self.model_version = settings.model
        self.usage_ledger = usage_ledger
        self.client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
        )
        self._closed = False

    def invoke(self, messages: list[dict[str, str]]) -> LLMOutput:
        request: dict[str, Any] = {
            "model": self.settings.model,
            "messages": cast(Any, messages),
            "temperature": self.settings.temperature,
        }
        if self.settings.max_tokens is not None:
            request["max_tokens"] = self.settings.max_tokens
        response = self.client.chat.completions.create(
            **request,
        )
        message = response.choices[0].message
        usage = usage_values(response.usage)
        if self.usage_ledger is not None:
            self.usage_ledger.record("llm", self.model_version, usage)
        return LLMOutput(
            text=message.content or "",
            usage=usage,
        )

    def close(self) -> None:
        """Release the SDK HTTP transport after a task-local runtime ends."""
        if self._closed:
            return
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
        self._closed = True
