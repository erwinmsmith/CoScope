"""Explicit, policy-checked promotion into a wider scope."""

from __future__ import annotations

from dataclasses import replace

from coscope.core.agent import AgentInstance
from coscope.core.artifact import ArtifactState
from coscope.core.memory import MemoryEntry
from coscope.memory.store import MemoryStore
from coscope.scope.descriptor import ScopeDescriptor
from coscope.scope.permissions import Permission
from coscope.scope.policy import PolicyEngine


class PromotionService:
    def __init__(self, store: MemoryStore, policy: PolicyEngine):
        self.store = store
        self.policy = policy

    def promote(
        self,
        actor: AgentInstance,
        entry: MemoryEntry,
        target_scope: ScopeDescriptor,
    ) -> MemoryEntry:
        if entry.state != ArtifactState.VERIFIED:
            raise ValueError("only verified memory can be promoted")
        decision = self.policy.decide(actor, target_scope, Permission.PROMOTE)
        if not decision.allowed:
            raise PermissionError(f"promotion denied: {decision.reason}")
        promoted = replace(
            entry,
            scope=target_scope,
            state=ArtifactState.COMMITTED,
            metadata={**entry.metadata, "promoted_by": actor.agent_id},
        )
        self.store.replace(promoted)
        return promoted
