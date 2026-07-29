"""Context pollution guardrails."""

from __future__ import annotations

from coscope.core.artifact import ArtifactState
from coscope.core.memory import MemoryEntry


class PollutionGuard:
    def __init__(self, *, allow_provisional_owner_content: bool = True):
        self.allow_provisional_owner_content = allow_provisional_owner_content

    def allow(self, entry: MemoryEntry, agent_id: str) -> bool:
        if entry.expired:
            return False
        if entry.state in {
            ArtifactState.QUARANTINED,
            ArtifactState.EXPIRED,
            ArtifactState.REVOKED,
        }:
            return False
        if entry.state in {ArtifactState.DRAFT, ArtifactState.PROPOSED}:
            return (
                self.allow_provisional_owner_content
                and entry.scope.owner_id == agent_id
            )
        return True
