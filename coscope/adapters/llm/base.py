"""Provider-neutral LLM adapter contract for the future live runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LLMOutput:
    text: str
    artifacts: list[dict[str, object]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class LLMAdapter(Protocol):
    model_version: str

    def invoke(self, messages: list[dict[str, str]]) -> LLMOutput: ...
