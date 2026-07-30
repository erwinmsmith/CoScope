"""Graph-of-thought DAG runtime with multi-parent readiness."""

from __future__ import annotations

from coscope.reasoning.base import ReasoningConfig
from coscope.reasoning.modes import ReasoningMode
from coscope.reasoning.node import NodeStatus, ReasoningNode


class GoTRuntime:
    def __init__(self, owner_agent_id: str, config: ReasoningConfig | None = None):
        self.config = config or ReasoningConfig(ReasoningMode.GOT)
        if self.config.mode != ReasoningMode.GOT:
            raise ValueError("GoTRuntime requires mode=got")
        self.owner_agent_id = owner_agent_id
        self._nodes: dict[str, ReasoningNode] = {}

    def add_node(
        self,
        node_id: str,
        *,
        parent_ids: list[str] | None = None,
        runtime_region: str | None = None,
    ) -> ReasoningNode:
        if node_id in self._nodes:
            raise ValueError(f"duplicate reasoning node: {node_id}")
        parents = list(parent_ids or [])
        if any(parent not in self._nodes for parent in parents):
            raise KeyError("all GoT parents must exist before a child")
        node = ReasoningNode(
            node_id=node_id,
            reasoning_mode=ReasoningMode.GOT,
            owner_agent_id=self.owner_agent_id,
            parent_ids=parents,
            runtime_region=runtime_region or f"got/node/{node_id}",
            status=NodeStatus.READY if not parents else NodeStatus.PENDING,
        )
        self._nodes[node_id] = node
        for parent in parents:
            self._nodes[parent].child_ids.append(node_id)
        self.refresh_readiness()
        return node

    def complete(self, node_id: str) -> None:
        self._nodes[node_id].status = NodeStatus.COMPLETED
        self.refresh_readiness()

    def refresh_readiness(self) -> None:
        for node in self._nodes.values():
            if node.status != NodeStatus.PENDING:
                continue
            if all(self._nodes[parent].status == NodeStatus.COMPLETED for parent in node.parent_ids):
                node.status = NodeStatus.READY

    def visible_ancestor_ids(self, node_id: str) -> set[str]:
        visible: set[str] = set()
        stack = list(self._nodes[node_id].parent_ids)
        while stack:
            parent = stack.pop()
            if parent in visible:
                continue
            visible.add(parent)
            stack.extend(self._nodes[parent].parent_ids)
        return visible

    def nodes(self) -> list[ReasoningNode]:
        return list(self._nodes.values())
