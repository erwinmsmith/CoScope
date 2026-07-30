"""In-memory trace recorder that rejects raw sensitive content."""

from __future__ import annotations

from typing import ClassVar

from coscope.core.events import RuntimeEvent


class TraceRecorder:
    _forbidden_keys: ClassVar[frozenset[str]] = frozenset(
        {"content", "full_query", "private_intent", "restricted_content"}
    )

    def __init__(self) -> None:
        self._events: list[RuntimeEvent] = []

    def record(self, event: RuntimeEvent) -> None:
        forbidden = self._forbidden_keys & set(event.attributes)
        if forbidden:
            raise ValueError(f"trace contains sensitive fields: {sorted(forbidden)}")
        self._events.append(event)

    def events(self, run_id: str | None = None) -> list[RuntimeEvent]:
        if run_id is None:
            return list(self._events)
        return [event for event in self._events if event.run_id == run_id]
