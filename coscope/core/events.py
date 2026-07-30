"""Redaction-friendly runtime events."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeEvent:
    event_type: str
    run_id: str
    actor_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, object] = field(default_factory=dict)
