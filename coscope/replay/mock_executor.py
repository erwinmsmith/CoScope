"""Rule-driven executor for offline control-flow tests."""

from __future__ import annotations

from collections.abc import Callable

from coscope.context import ContextPacket
from coscope.core import AgentInstance, Artifact
from coscope.reasoning import ReasoningNode
from coscope.runtime import ExecutionOutput


class MockExecutor:
    def __init__(
        self,
        handler: Callable[
            [AgentInstance, ReasoningNode, ContextPacket],
            str | ExecutionOutput | list[Artifact],
        ]
        | None = None,
    ):
        self.handler = handler

    def invoke(
        self,
        agent: AgentInstance,
        node: ReasoningNode,
        context: ContextPacket,
    ) -> ExecutionOutput:
        if self.handler is None:
            return ExecutionOutput(
                text=f"{agent.role}:{node.node_id}",
                metadata={"context_items": len(context.all_memories)},
            )
        value = self.handler(agent, node, context)
        if isinstance(value, ExecutionOutput):
            return value
        if isinstance(value, list):
            return ExecutionOutput(text="", artifacts=value)
        return ExecutionOutput(text=str(value))
