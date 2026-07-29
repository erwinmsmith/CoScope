"""Agent-specific context packet."""

from __future__ import annotations

from dataclasses import dataclass, field

from coscope.core.memory import MemoryEntry


@dataclass
class ContextPacket:
    agent_id: str
    reasoning_node_id: str
    full_query: str = ""
    public_intent: str = ""
    private_intent: str | None = None
    system_context: list[str] = field(default_factory=list)
    task_shared_context: list[MemoryEntry] = field(default_factory=list)
    agent_private_context: list[MemoryEntry] = field(default_factory=list)
    branch_local_context: list[MemoryEntry] = field(default_factory=list)
    retrieved_evidence: list[MemoryEntry] = field(default_factory=list)
    tool_observations: list[MemoryEntry] = field(default_factory=list)
    provenance_map: dict[str, dict[str, object]] = field(default_factory=dict)
    budget_usage: dict[str, int] = field(default_factory=dict)

    @property
    def all_memories(self) -> list[MemoryEntry]:
        return (
            self.task_shared_context
            + self.agent_private_context
            + self.branch_local_context
            + self.retrieved_evidence
            + self.tool_observations
        )
