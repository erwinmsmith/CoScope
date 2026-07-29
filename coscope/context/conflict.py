"""Conservative conflict selection using explicit conflict keys."""

from __future__ import annotations

from coscope.core.artifact import ArtifactState
from coscope.core.memory import MemoryEntry


def resolve_conflicts(entries: list[MemoryEntry]) -> list[MemoryEntry]:
    keyed: dict[str, MemoryEntry] = {}
    unkeyed: list[MemoryEntry] = []
    for entry in entries:
        key = entry.metadata.get("conflict_key")
        if not key:
            unkeyed.append(entry)
            continue
        current = keyed.get(str(key))
        if current is None or _priority(entry) > _priority(current):
            keyed[str(key)] = entry
    return unkeyed + list(keyed.values())


def _priority(entry: MemoryEntry) -> tuple[int, float]:
    state_score = {
        ArtifactState.COMMITTED: 3,
        ArtifactState.VERIFIED: 2,
        ArtifactState.PROPOSED: 1,
    }.get(entry.state, 0)
    raw_confidence = entry.metadata.get("confidence", 0.0)
    confidence = (
        float(raw_confidence) if isinstance(raw_confidence, (int, float)) else 0.0
    )
    return state_score, confidence
