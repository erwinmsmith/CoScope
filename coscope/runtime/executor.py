"""Runtime executor contract shared by simulated and live adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from coscope.context import ContextPacket
from coscope.core import AgentInstance, Artifact
from coscope.reasoning import ReasoningNode
from coscope.scope import ScopeDescriptor


@dataclass
class ExecutionOutput:
    text: str
    artifacts: list[Artifact] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class RuntimeExecutor(Protocol):
    def invoke(
        self,
        agent: AgentInstance,
        node: ReasoningNode,
        context: ContextPacket,
    ) -> ExecutionOutput: ...


@dataclass
class RuntimeInvocation:
    agent_id: str
    reasoning_node_id: str
    full_query: str
    public_intent: str
    private_intent: str | None = None
    required_facets: tuple[str, ...] = ()
    context_budget: int = 2_000
    retrieval_budget: int = 20
    memory_types: frozenset[str] = field(default_factory=frozenset)
    output_scope: ScopeDescriptor | None = None
