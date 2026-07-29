"""Live runtime executor backed by an LLM adapter."""

from __future__ import annotations

from coscope.adapters.llm.base import LLMAdapter
from coscope.context.packet import ContextPacket
from coscope.core.agent import AgentInstance
from coscope.reasoning.node import ReasoningNode
from coscope.runtime.executor import ExecutionOutput


class LLMExecutor:
    """Generate text from an agent's already authorized ContextPacket."""

    def __init__(
        self,
        llm: LLMAdapter,
        *,
        system_instructions: tuple[str, ...] = (),
    ):
        self.llm = llm
        self.system_instructions = system_instructions

    def invoke(
        self,
        agent: AgentInstance,
        node: ReasoningNode,
        context: ContextPacket,
    ) -> ExecutionOutput:
        evidence = "\n".join(
            f"- [{entry.source_type}:{entry.source_id}] {entry.content}"
            for entry in context.all_memories
        )
        system_parts = [
            f"You are the {agent.role} agent.",
            "Use only the supplied authorized context. "
            "If evidence is insufficient, say so explicitly.",
            *self.system_instructions,
            *context.system_context,
        ]
        user = (
            f"Question:\n{context.full_query}\n\n"
            f"Authorized context:\n{evidence or '(none)'}"
        )
        output = self.llm.invoke(
            [
                {"role": "system", "content": "\n".join(system_parts)},
                {"role": "user", "content": user},
            ]
        )
        return ExecutionOutput(
            text=output.text,
            metadata={
                "model": self.llm.model_version,
                "usage": output.usage,
                "reasoning_node_id": node.node_id,
                "context_items": len(context.all_memories),
                "context_budget_usage": dict(context.budget_usage),
            },
        )
