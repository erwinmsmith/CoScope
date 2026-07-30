"""Assemble independent context packets after retrieval."""

from __future__ import annotations

from coscope.context.budget import estimate_tokens
from coscope.context.packet import ContextPacket
from coscope.context.selector import ContextSelector
from coscope.core.artifact import ArtifactState, ArtifactType
from coscope.memory.store import MemoryStore
from coscope.retrieval.request import RetrievalRequest, RetrievalResult
from coscope.scope.descriptor import Visibility


class ContextManager:
    def __init__(
        self,
        store: MemoryStore,
        selector: ContextSelector | None = None,
    ):
        self.store = store
        self.selector = selector or ContextSelector()

    def assemble(
        self,
        request: RetrievalRequest,
        result: RetrievalResult,
        *,
        system_context: list[str] | None = None,
    ) -> ContextPacket:
        visible_state = [
            entry
            for entry in self.store.entries_in_view(
                request.effective_view,
                states=frozenset(ArtifactState),
            )
            if not entry.metadata.get("retrieval_only")
            and self._accepted_by_agent_class(request, entry)
        ]
        retrieved = [
            candidate.memory
            for candidate in result.candidates
            if self._accepted_by_agent_class(request, candidate.memory)
        ]
        selected, used = self.selector.select(
            retrieved + visible_state,
            agent_id=request.agent_id,
            token_budget=request.context_budget,
        )
        retrieved_ids = {entry.memory_id for entry in retrieved}
        selected = self._apply_channel_budgets(
            selected,
            retrieved_ids=retrieved_ids,
            channel_budgets=request.channel_budgets,
        )
        used = sum(estimate_tokens(entry.content) for entry in selected)
        packet = ContextPacket(
            agent_id=request.agent_id,
            reasoning_node_id=request.reasoning_node_id,
            full_query=request.full_query,
            public_intent=request.public_intent,
            private_intent=request.private_intent,
            system_context=list(system_context or []),
            budget_usage={
                "tokens": used,
                "limit": request.context_budget,
                "selected_items": len(selected),
            },
        )
        for entry in selected:
            channel = self._channel(entry, retrieved_ids)
            if channel == "retrieved_evidence":
                packet.retrieved_evidence.append(entry)
            elif channel == "tool_observations":
                packet.tool_observations.append(entry)
            elif channel == "agent_private":
                packet.agent_private_context.append(entry)
            elif channel == "branch_local":
                packet.branch_local_context.append(entry)
            else:
                packet.task_shared_context.append(entry)
            packet.provenance_map[entry.memory_id] = {
                "source_type": entry.source_type,
                "source_id": entry.source_id,
                "scope_id": entry.scope.scope_id,
                "state": entry.state.value,
            }
        return packet

    @classmethod
    def _apply_channel_budgets(
        cls,
        entries,
        *,
        retrieved_ids: set[str],
        channel_budgets: dict[str, int],
    ):
        if not channel_budgets:
            return entries
        selected = []
        used: dict[str, int] = {}
        for entry in entries:
            channel = cls._channel(entry, retrieved_ids)
            limit = channel_budgets.get(channel)
            cost = estimate_tokens(entry.content)
            if limit is not None and used.get(channel, 0) + cost > limit:
                continue
            selected.append(entry)
            used[channel] = used.get(channel, 0) + cost
        return selected

    @staticmethod
    def _channel(entry, retrieved_ids: set[str]) -> str:
        if entry.memory_id in retrieved_ids:
            return "retrieved_evidence"
        if entry.metadata.get("artifact_type") == "tool_observation":
            return "tool_observations"
        if entry.scope.visibility == Visibility.AGENT_PRIVATE:
            return "agent_private"
        if entry.scope.visibility in {
            Visibility.BRANCH_PRIVATE,
            Visibility.NODE_LOCAL,
        }:
            return "branch_local"
        return "task_shared"

    @staticmethod
    def _accepted_by_agent_class(
        request: RetrievalRequest,
        entry,
    ) -> bool:
        raw_type = entry.metadata.get("artifact_type")
        if not raw_type or not request.readable_artifact_types:
            return True
        try:
            artifact_type = ArtifactType(str(raw_type))
        except ValueError:
            return False
        return artifact_type in request.readable_artifact_types
