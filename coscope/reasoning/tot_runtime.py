"""Tree-of-thought branching, scoring, pruning, and explicit merge."""

from __future__ import annotations

from coscope.reasoning.base import ReasoningConfig
from coscope.reasoning.modes import ReasoningMode
from coscope.reasoning.node import NodeStatus, ReasoningNode


class ToTRuntime:
    def __init__(self, owner_agent_id: str, config: ReasoningConfig | None = None):
        self.config = config or ReasoningConfig(ReasoningMode.TOT)
        if self.config.mode != ReasoningMode.TOT:
            raise ValueError("ToTRuntime requires mode=tot")
        self.owner_agent_id = owner_agent_id
        self._nodes: dict[str, ReasoningNode] = {}
        self._counter = 0

    def create_root(self) -> ReasoningNode:
        if self._nodes:
            raise ValueError("root already exists")
        return self._new([], "tot/root")

    def branch(self, parent_id: str, count: int | None = None) -> list[ReasoningNode]:
        parent = self._nodes[parent_id]
        branch_count = count or self.config.branching_factor
        if branch_count > self.config.branching_factor:
            raise ValueError("branching factor exceeded")
        raw_depth = parent.local_state.get("depth", 0)
        depth = (raw_depth if isinstance(raw_depth, int) else 0) + 1
        if depth >= self.config.max_depth:
            raise RuntimeError("maximum ToT depth reached")
        branches = []
        for index in range(branch_count):
            node = self._new(
                [parent_id],
                f"{parent.runtime_region}/branch_{index}_{self._counter}",
            )
            node.local_state["depth"] = depth
            parent.child_ids.append(node.node_id)
            branches.append(node)
        parent.status = NodeStatus.COMPLETED
        return branches

    def score(self, node_id: str, value: float) -> None:
        self._nodes[node_id].local_state["score"] = float(value)

    def prune(self, keep: int) -> list[str]:
        active = [
            node
            for node in self._nodes.values()
            if node.status in {NodeStatus.READY, NodeStatus.COMPLETED}
            and node.parent_ids
        ]
        def score_of(node: ReasoningNode) -> float:
            value = node.local_state.get("score")
            return float(value) if isinstance(value, (int, float)) else float("-inf")

        ranked = sorted(active, key=score_of, reverse=True)
        pruned = ranked[keep:]
        for node in pruned:
            node.status = NodeStatus.PRUNED
        return [node.node_id for node in pruned]

    def merge(self, parent_ids: list[str]) -> ReasoningNode:
        if len(parent_ids) < 2:
            raise ValueError("ToT merge requires at least two branches")
        if any(self._nodes[node_id].status == NodeStatus.PRUNED for node_id in parent_ids):
            raise ValueError("pruned branches cannot be merged")
        node = self._new(parent_ids, f"tot/merged/{self._counter}")
        for parent_id in parent_ids:
            self._nodes[parent_id].child_ids.append(node.node_id)
        return node

    def _new(self, parents: list[str], region: str) -> ReasoningNode:
        node = ReasoningNode(
            node_id=f"tot_{self._counter}",
            reasoning_mode=ReasoningMode.TOT,
            owner_agent_id=self.owner_agent_id,
            parent_ids=list(parents),
            runtime_region=region,
            status=NodeStatus.READY,
            local_state={"depth": 0},
        )
        self._counter += 1
        self._nodes[node.node_id] = node
        return node

    def nodes(self) -> list[ReasoningNode]:
        return list(self._nodes.values())
