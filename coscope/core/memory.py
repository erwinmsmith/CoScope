"""Runtime memory entity."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from coscope.core.artifact import ArtifactState
from coscope.scope.descriptor import ScopeDescriptor


@dataclass
class MemoryEntry:
    content: str
    scope: ScopeDescriptor
    source_type: str
    source_id: str
    state: ArtifactState = ArtifactState.COMMITTED
    memory_id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:16]}")
    created_at: float = field(default_factory=time.time)
    valid_until: float | None = None
    vector: tuple[float, ...] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return self.valid_until is not None and self.valid_until <= time.time()
