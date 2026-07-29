"""Linear chain-of-thought state runtime."""

from __future__ import annotations

from coscope.reasoning.base import ReasoningConfig
from coscope.reasoning.modes import ReasoningMode
from coscope.reasoning.node import NodeStatus, ReasoningNode


class CoTRuntime:
    def __init__(self, owner_agent_id: str, config: ReasoningConfig | None = None):
        self.config = config or ReasoningConfig(ReasoningMode.COT)
        if self.config.mode != ReasoningMode.COT:
            raise ValueError("CoTRuntime requires mode=cot")
        self.owner_agent_id = owner_agent_id
        self._nodes: list[ReasoningNode] = []

    def create_root(self) -> ReasoningNode:
        if self._nodes:
            raise ValueError("root already exists")
        return self._append(parent=None)

    def advance(self) -> ReasoningNode:
        if not self._nodes:
            return self.create_root()
        if len(self._nodes) >= self.config.max_depth:
            raise RuntimeError("maximum CoT depth reached")
        parent = self._nodes[-1]
        parent.status = NodeStatus.COMPLETED
        return self._append(parent=parent)

    def _append(self, parent: ReasoningNode | None) -> ReasoningNode:
        index = len(self._nodes)
        node = ReasoningNode(
            node_id=f"cot_{index}",
            reasoning_mode=ReasoningMode.COT,
            owner_agent_id=self.owner_agent_id,
            parent_ids=[parent.node_id] if parent else [],
            runtime_region=f"cot/chain/step_{index}",
            status=NodeStatus.READY,
        )
        if parent:
            parent.child_ids.append(node.node_id)
        self._nodes.append(node)
        return node

    def nodes(self) -> list[ReasoningNode]:
        return list(self._nodes)
