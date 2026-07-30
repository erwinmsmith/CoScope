"""Resolved corpus view for one agent at one reasoning node."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from coscope.core.agent import AgentInstance
from coscope.scope.descriptor import ScopeDescriptor
from coscope.scope.permissions import Permission
from coscope.scope.policy import PolicyEngine


@dataclass(frozen=True)
class EffectiveView:
    scope_ids: frozenset[str]
    public_scope_ids: frozenset[str]
    private_scope_ids: frozenset[str]
    tenant_id: str
    workspace_id: str
    policy_versions: frozenset[str]
    index_snapshots: frozenset[str]

    @classmethod
    def empty(cls, tenant_id: str, workspace_id: str) -> EffectiveView:
        return cls(
            frozenset(),
            frozenset(),
            frozenset(),
            tenant_id,
            workspace_id,
            frozenset(),
            frozenset(),
        )

    @classmethod
    def shared_intersection(cls, views: Iterable[EffectiveView]) -> EffectiveView:
        items = list(views)
        if not items:
            raise ValueError("at least one view is required")
        first = items[0]
        if any(
            item.tenant_id != first.tenant_id or item.workspace_id != first.workspace_id
            for item in items[1:]
        ):
            return cls.empty(first.tenant_id, first.workspace_id)
        public = set(first.public_scope_ids)
        for item in items[1:]:
            public.intersection_update(item.public_scope_ids)
        return cls(
            scope_ids=frozenset(public),
            public_scope_ids=frozenset(public),
            private_scope_ids=frozenset(),
            tenant_id=first.tenant_id,
            workspace_id=first.workspace_id,
            policy_versions=frozenset.intersection(
                *(item.policy_versions for item in items)
            ),
            index_snapshots=frozenset.intersection(
                *(item.index_snapshots for item in items)
            ),
        )

    def extra_from(self, shared: EffectiveView) -> EffectiveView:
        extra = self.scope_ids - shared.scope_ids
        return EffectiveView(
            scope_ids=frozenset(extra),
            public_scope_ids=frozenset(self.public_scope_ids - shared.scope_ids),
            private_scope_ids=frozenset(self.private_scope_ids - shared.scope_ids),
            tenant_id=self.tenant_id,
            workspace_id=self.workspace_id,
            policy_versions=self.policy_versions,
            index_snapshots=self.index_snapshots,
        )


class ScopeEngine:
    def __init__(self, policy_engine: PolicyEngine | None = None):
        self.policy_engine = policy_engine or PolicyEngine()

    def resolve(
        self,
        agent: AgentInstance,
        scopes: Iterable[ScopeDescriptor],
        *,
        runtime_region: str | None = None,
    ) -> EffectiveView:
        allowed: list[ScopeDescriptor] = []
        for scope in scopes:
            decision = self.policy_engine.decide(
                agent, scope, Permission.READ, runtime_region=runtime_region
            )
            if decision.allowed:
                allowed.append(scope)
        public = frozenset(
            scope.scope_id for scope in allowed if scope.is_public_execution_scope
        )
        private = frozenset(
            scope.scope_id for scope in allowed if not scope.is_public_execution_scope
        )
        return EffectiveView(
            scope_ids=public | private,
            public_scope_ids=public,
            private_scope_ids=private,
            tenant_id=agent.tenant_id,
            workspace_id=agent.workspace_id,
            policy_versions=frozenset(scope.policy_version for scope in allowed),
            index_snapshots=frozenset(scope.index_snapshot for scope in allowed),
        )
