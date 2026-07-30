"""Context quality metrics."""

from dataclasses import dataclass

from coscope.context import ContextPacket


def duplicate_evidence_rate(packet: ContextPacket) -> float:
    contents = [entry.content.casefold().strip() for entry in packet.all_memories]
    if not contents:
        return 0.0
    return 1.0 - len(set(contents)) / len(contents)


def relevant_token_ratio(relevant_tokens: int, total_tokens: int) -> float:
    return relevant_tokens / total_tokens if total_tokens else 0.0


@dataclass(frozen=True)
class ContextPollutionReport:
    selected_items: int
    polluted_items: int
    private_thinking_exposure: int
    cross_role_knowledge_exposure: int

    @property
    def pollution_rate(self) -> float:
        return (
            self.polluted_items / self.selected_items
            if self.selected_items
            else 0.0
        )


def context_pollution(
    packet: ContextPacket,
    *,
    agent_id: str,
) -> ContextPollutionReport:
    """Measure intentionally irrelevant or private cross-agent context.

    Benchmark memories declare ``intended_agents`` and ``knowledge_class``.
    These annotations describe the ideal role view independently of the arm's
    access-control configuration.
    """
    polluted_ids: set[str] = set()
    private_ids: set[str] = set()
    cross_role_ids: set[str] = set()
    for entry in packet.all_memories:
        intended = entry.metadata.get("intended_agents")
        if isinstance(intended, (list, tuple, set, frozenset)):
            intended_agents = {str(item) for item in intended}
            if intended_agents and agent_id not in intended_agents:
                polluted_ids.add(entry.memory_id)
        if (
            entry.metadata.get("privacy") == "private"
            and entry.metadata.get("owner_agent_id") != agent_id
        ):
            private_ids.add(entry.memory_id)
            polluted_ids.add(entry.memory_id)
        knowledge_class = entry.metadata.get("knowledge_class")
        if knowledge_class and str(knowledge_class) != agent_id:
            cross_role_ids.add(entry.memory_id)
            polluted_ids.add(entry.memory_id)
    return ContextPollutionReport(
        selected_items=len(packet.all_memories),
        polluted_items=len(polluted_ids),
        private_thinking_exposure=len(private_ids),
        cross_role_knowledge_exposure=len(cross_role_ids),
    )
