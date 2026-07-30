"""Agent instances and application-level topology.

Agent topology describes who executes and communicates. It is deliberately
independent from the CoT/ToT/GoT reasoning mode used by an agent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from coscope.core.artifact import ArtifactType


@dataclass(frozen=True)
class AgentContextPolicy:
    """Class-level rules for context consumption and publication."""

    readable_artifact_types: frozenset[ArtifactType] = field(
        default_factory=frozenset
    )
    publishable_artifact_types: frozenset[ArtifactType] = field(
        default_factory=frozenset
    )
    allow_private_thinking: bool = True
    allow_public_thinking: bool = False
    default_context_budget: int = 2_000
    channel_budgets: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentClass:
    """Reusable role blueprint shared by instances of the same agent class."""

    class_id: str
    role: str
    knowledge_permissions: frozenset[str]
    context_policy: AgentContextPolicy = field(default_factory=AgentContextPolicy)
    capabilities: frozenset[str] = field(default_factory=frozenset)
    tool_permissions: frozenset[str] = field(default_factory=frozenset)

    def instantiate(
        self,
        agent_id: str,
        *,
        runtime_state: dict[str, object] | None = None,
        tenant_id: str = "default",
        workspace_id: str = "default",
    ) -> AgentInstance:
        return AgentInstance(
            agent_id=agent_id,
            role=self.role,
            capabilities=self.capabilities,
            tool_permissions=self.tool_permissions,
            knowledge_permissions=self.knowledge_permissions,
            runtime_state=dict(runtime_state or {}),
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            agent_class_id=self.class_id,
            context_policy=self.context_policy,
        )


@dataclass
class AgentInstance:
    agent_id: str
    role: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    tool_permissions: frozenset[str] = field(default_factory=frozenset)
    knowledge_permissions: frozenset[str] = field(default_factory=frozenset)
    runtime_state: dict[str, object] = field(default_factory=dict)
    tenant_id: str = "default"
    workspace_id: str = "default"
    agent_class_id: str = "ad_hoc"
    context_policy: AgentContextPolicy = field(default_factory=AgentContextPolicy)


@dataclass
class AgentTopology:
    """Directed communication/call graph between application agents."""

    agents: dict[str, AgentInstance] = field(default_factory=dict)
    edges: dict[str, set[str]] = field(default_factory=dict)

    def add_agent(self, agent: AgentInstance) -> None:
        if agent.agent_id in self.agents:
            raise ValueError(f"duplicate agent: {agent.agent_id}")
        self.agents[agent.agent_id] = agent
        self.edges.setdefault(agent.agent_id, set())

    def connect(self, source_id: str, target_id: str) -> None:
        if source_id not in self.agents or target_id not in self.agents:
            raise KeyError("both topology endpoints must be registered")
        if source_id == target_id:
            raise ValueError("self edges are not allowed")
        self.edges.setdefault(source_id, set()).add(target_id)

    def topological_order(self) -> list[str]:
        indegree = dict.fromkeys(self.agents, 0)
        for targets in self.edges.values():
            for target in targets:
                indegree[target] += 1
        ready = sorted(agent_id for agent_id, degree in indegree.items() if degree == 0)
        ordered: list[str] = []
        while ready:
            current = ready.pop(0)
            ordered.append(current)
            for target in sorted(self.edges.get(current, ())):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort()
        if len(ordered) != len(self.agents):
            raise ValueError("agent topology contains a cycle")
        return ordered
