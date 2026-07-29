"""Provider token-usage events collected across a runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class UsageEvent:
    category: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    metadata: Mapping[str, str] = field(default_factory=dict)


class UsageLedger:
    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    def record(
        self,
        category: str,
        model: str,
        usage: Mapping[str, int],
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> UsageEvent:
        event = UsageEvent(
            category=category,
            model=model,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            reasoning_tokens=int(usage.get("reasoning_tokens", 0)),
            cached_tokens=int(usage.get("cached_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            metadata=dict(metadata or {}),
        )
        self.events.append(event)
        return event

    def summary(self) -> dict[str, object]:
        categories: dict[str, dict[str, int]] = {}
        models: dict[str, dict[str, int]] = {}
        for event in self.events:
            for bucket, key in (
                (categories, event.category),
                (models, event.model),
            ):
                totals = bucket.setdefault(
                    key,
                    {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "reasoning_tokens": 0,
                        "cached_tokens": 0,
                        "total_tokens": 0,
                    },
                )
                totals["calls"] += 1
                totals["prompt_tokens"] += event.prompt_tokens
                totals["completion_tokens"] += event.completion_tokens
                totals["reasoning_tokens"] += event.reasoning_tokens
                totals["cached_tokens"] += event.cached_tokens
                totals["total_tokens"] += event.total_tokens
        return {
            "all": self._totals(self.events),
            "by_category": categories,
            "by_model": models,
            "events": [asdict(event) for event in self.events],
        }

    @staticmethod
    def _totals(events: list[UsageEvent]) -> dict[str, int]:
        return {
            "calls": len(events),
            "prompt_tokens": sum(event.prompt_tokens for event in events),
            "completion_tokens": sum(event.completion_tokens for event in events),
            "reasoning_tokens": sum(event.reasoning_tokens for event in events),
            "cached_tokens": sum(event.cached_tokens for event in events),
            "total_tokens": sum(event.total_tokens for event in events),
        }
