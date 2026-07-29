"""Hard access-control decisions evaluated before retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from coscope.core.agent import AgentInstance
from coscope.scope.descriptor import ScopeDescriptor, Visibility
from coscope.scope.permissions import Permission


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str


class PolicyEngine:
    def decide(
        self,
        agent: AgentInstance,
        scope: ScopeDescriptor,
        permission: Permission = Permission.READ,
        *,
        runtime_region: str | None = None,
    ) -> PolicyDecision:
        if agent.tenant_id != scope.tenant_id:
            return PolicyDecision(False, "tenant_mismatch")
        if agent.workspace_id != scope.workspace_id:
            return PolicyDecision(False, "workspace_mismatch")
        if scope.knowledge_domains and agent.knowledge_permissions:
            if not scope.knowledge_domains <= agent.knowledge_permissions:
                return PolicyDecision(False, "knowledge_domain_denied")

        explicit = self._explicit_permission(agent, scope, permission)
        if explicit is not None:
            return PolicyDecision(explicit, "explicit_policy")

        if scope.visibility in {Visibility.SYSTEM, Visibility.TEAM_SHARED}:
            return PolicyDecision(permission == Permission.READ, "shared_default")
        if scope.visibility == Visibility.ROLE_SHARED:
            allowed = scope.owner_id in {None, agent.role}
            return PolicyDecision(allowed and permission == Permission.READ, "role_visibility")
        if scope.visibility == Visibility.AGENT_PRIVATE:
            return PolicyDecision(scope.owner_id == agent.agent_id, "agent_owner")
        if scope.visibility == Visibility.BRANCH_PRIVATE:
            return PolicyDecision(
                scope.owner_id == agent.agent_id and runtime_region == scope.runtime_region,
                "branch_owner",
            )
        if scope.visibility == Visibility.NODE_LOCAL:
            return PolicyDecision(
                scope.owner_id == agent.agent_id and runtime_region == scope.runtime_region,
                "node_owner",
            )
        return PolicyDecision(False, "restricted_requires_explicit_policy")

    @staticmethod
    def _explicit_permission(
        agent: AgentInstance,
        scope: ScopeDescriptor,
        permission: Permission,
    ) -> bool | None:
        principals = (agent.agent_id, f"role:{agent.role}", "team", "*")
        matched = False
        for principal in principals:
            permissions = scope.permissions.get(principal)
            if permissions is None:
                continue
            matched = True
            if permission in permissions:
                return True
        return False if matched else None
