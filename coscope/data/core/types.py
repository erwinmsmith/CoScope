"""
Core types for coscope.data (dataset construction subpackage).

These types are data-construction-specific and do NOT duplicate anything in
`coscope.core.types`. For Agent / MemoryEntry / RetrievalRequest /
PolicyConstraints / Provenance, we reuse the existing framework types.

See `data/docs/GoT_Dataset_Requirements_v2.md` section 3.6.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from coscope.core.types import Agent, MemoryEntry, RetrievalRequest


# ============================================================
# Enums
# ============================================================


class ReasoningPathType(str, Enum):
    """Reasoning path type. Stage 1 implements GoT only; CoT / ToT are reserved."""

    GOT = "GoT"
    COT = "CoT"   # Stage 2, not yet implemented
    TOT = "ToT"   # Stage 3, not yet implemented


class GraphType(str, Enum):
    """GoT graph structural type. Drives subset (S1/S2/S3) expectations."""

    LINEAR = "LINEAR"
    FORK = "FORK"
    MERGE = "MERGE"
    FORK_MERGE = "FORK_MERGE"
    INDEPENDENT = "INDEPENDENT"
    POLICY_ISOLATED = "POLICY_ISOLATED"


class SubsetLabel(str, Enum):
    """
    Experimental subset label.

    S1 / S2 / S3 are mutually exclusive and determined by rho.
    S4 is orthogonal (policy conflict) and is represented separately via
    `Episode.s4_eligible`; S4 does NOT overwrite `rho_subset`.
    """

    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"


class NodeType(str, Enum):
    """GoT node type."""

    PLANNER = "PLANNER"
    SOLVER = "SOLVER"
    VERIFIER = "VERIFIER"


class EdgeType(str, Enum):
    """GoT edge type."""

    DEPENDS_ON = "DEPENDS_ON"
    AGGREGATES = "AGGREGATES"


# ============================================================
# Exceptions
# ============================================================


class InvalidGraphError(ValueError):
    """Raised when a constructed GoTGraph fails DAG / structural validation."""


# ============================================================
# GoT graph dataclasses
# ============================================================


@dataclass
class GoTNode:
    """A node in a Graph-of-Thought."""

    node_id: str
    node_type: str          # "PLANNER" | "SOLVER" | "VERIFIER"
    agent_role: str         # "planner" | "solver" | "verifier"
    hop_index: Optional[int] = None
    parent_node_ids: List[str] = field(default_factory=list)
    child_node_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "agent_role": self.agent_role,
            "hop_index": self.hop_index,
            "parent_node_ids": list(self.parent_node_ids),
            "child_node_ids": list(self.child_node_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoTNode:
        return cls(
            node_id=data["node_id"],
            node_type=data["node_type"],
            agent_role=data.get("agent_role", data["node_type"].lower()),
            hop_index=data.get("hop_index"),
            parent_node_ids=list(data.get("parent_node_ids", [])),
            child_node_ids=list(data.get("child_node_ids", [])),
        )


@dataclass
class GoTEdge:
    """A directed edge in a Graph-of-Thought."""

    from_node: str
    to_node: str
    edge_type: str = EdgeType.DEPENDS_ON.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.from_node,
            "to": self.to_node,
            "edge_type": self.edge_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoTEdge:
        return cls(
            from_node=data.get("from") or data["from_node"],
            to_node=data.get("to") or data["to_node"],
            edge_type=data.get("edge_type", EdgeType.DEPENDS_ON.value),
        )


@dataclass
class GoTGraph:
    """
    A directed acyclic Graph-of-Thought describing agent reasoning dependencies
    within a single episode.
    """

    nodes: List[GoTNode] = field(default_factory=list)
    edges: List[GoTEdge] = field(default_factory=list)
    graph_type: GraphType = GraphType.LINEAR

    # --- lookups --------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[GoTNode]:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def reasoning_nodes(self) -> List[GoTNode]:
        """All nodes participating in the reasoning DAG (excludes VERIFIER)."""
        return [n for n in self.nodes if n.node_type != NodeType.VERIFIER.value]

    def solver_nodes(self) -> List[GoTNode]:
        return [n for n in self.nodes if n.node_type == NodeType.SOLVER.value]

    def planner_node(self) -> Optional[GoTNode]:
        for n in self.nodes:
            if n.node_type == NodeType.PLANNER.value:
                return n
        return None

    def verifier_node(self) -> Optional[GoTNode]:
        for n in self.nodes:
            if n.node_type == NodeType.VERIFIER.value:
                return n
        return None

    # --- graph algorithms ----------------------------------------------

    def _parent_map(self) -> Dict[str, List[str]]:
        parents: Dict[str, List[str]] = {n.node_id: list(n.parent_node_ids) for n in self.nodes}
        # Also incorporate edges that may not be reflected in node.parent_node_ids
        for e in self.edges:
            if e.to_node in parents and e.from_node not in parents[e.to_node]:
                parents[e.to_node].append(e.from_node)
        return parents

    def _child_map(self) -> Dict[str, List[str]]:
        children: Dict[str, List[str]] = {n.node_id: list(n.child_node_ids) for n in self.nodes}
        for e in self.edges:
            if e.from_node in children and e.to_node not in children[e.from_node]:
                children[e.from_node].append(e.to_node)
        return children

    def get_all_ancestors(self, node_id: str) -> List[str]:
        """
        Return all ancestor node ids (BFS). The result does NOT include `node_id` itself.
        """
        parent_map = self._parent_map()
        if node_id not in parent_map:
            return []
        visited: Set[str] = set()
        queue: deque = deque(parent_map[node_id])
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            for p in parent_map.get(cur, []):
                if p not in visited:
                    queue.append(p)
        return list(visited)

    def is_dag(self) -> bool:
        """Validate acyclicity via Kahn's algorithm (topological sort)."""
        parent_map = self._parent_map()
        child_map = self._child_map()
        indegree: Dict[str, int] = {nid: len(parent_map.get(nid, [])) for nid in parent_map}
        queue: deque = deque([nid for nid, d in indegree.items() if d == 0])
        visited = 0
        while queue:
            cur = queue.popleft()
            visited += 1
            for ch in child_map.get(cur, []):
                indegree[ch] -= 1
                if indegree[ch] == 0:
                    queue.append(ch)
        return visited == len(indegree)

    def validate(self) -> None:
        """
        Perform structural validation. Raises `InvalidGraphError` on failure.

        Checks:
        - DAG (no cycles).
        - No orphan non-Planner / non-Verifier nodes (each Solver has >=1 parent).
        - Verifier, if present, is isolated (no in/out edges).
        """
        if not self.is_dag():
            raise InvalidGraphError(f"GoTGraph is not a DAG (graph_type={self.graph_type})")

        parent_map = self._parent_map()
        child_map = self._child_map()
        for n in self.nodes:
            if n.node_type == NodeType.SOLVER.value and not parent_map.get(n.node_id):
                raise InvalidGraphError(
                    f"Solver node '{n.node_id}' has no parents (orphan)"
                )
            if n.node_type == NodeType.VERIFIER.value:
                if parent_map.get(n.node_id) or child_map.get(n.node_id):
                    raise InvalidGraphError(
                        f"Verifier node '{n.node_id}' must be isolated "
                        f"(no in/out edges), got parents="
                        f"{parent_map.get(n.node_id)} children={child_map.get(n.node_id)}"
                    )

    # --- serialization --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "graph_type": self.graph_type.value if isinstance(self.graph_type, GraphType) else str(self.graph_type),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoTGraph:
        gt_raw = data.get("graph_type", GraphType.LINEAR.value)
        try:
            gt = GraphType(gt_raw)
        except ValueError:
            gt = GraphType.LINEAR
        return cls(
            nodes=[GoTNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[GoTEdge.from_dict(e) for e in data.get("edges", [])],
            graph_type=gt,
        )


# ============================================================
# Ground truth evidence
# ============================================================


@dataclass
class GroundTruthEvidence:
    """
    A ground-truth memory item required by one or more agents to reach the
    correct answer. Used by evaluation to check recall.
    """

    memory_id: str
    scope_layer: str                           # "workspace_semantic" | "task_shared_episodic" | ...
    hop_index: Optional[int]
    required_by_agent_ids: List[str]
    evidence_type: str                          # "shared_required" | "private_required"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "scope_layer": self.scope_layer,
            "hop_index": self.hop_index,
            "required_by_agent_ids": list(self.required_by_agent_ids),
            "evidence_type": self.evidence_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroundTruthEvidence:
        return cls(
            memory_id=data["memory_id"],
            scope_layer=data["scope_layer"],
            hop_index=data.get("hop_index"),
            required_by_agent_ids=list(data.get("required_by_agent_ids", [])),
            evidence_type=data.get("evidence_type", "shared_required"),
        )


# ============================================================
# Episode
# ============================================================


@dataclass
class Episode:
    """
    A single multi-agent experiment unit.

    The fields `agents`, `memory_entries`, and `retrieval_requests` hold
    `coscope.core.types` objects directly, so an Episode can be fed into the
    CoScope runtime via `to_runtime_objects()` without format conversion.
    """

    episode_id: str
    dataset: str                         # "musique" | "2wikimhqa" | "hotpotqa" | "gsm8k" | "math"
    split: str                           # "train" | "dev" | "test"
    original_id: str
    hop_count: int
    graph_type: GraphType
    reasoning_path_type: ReasoningPathType

    rho: float
    rho_subset: SubsetLabel
    policy_conflict: bool
    s4_eligible: bool

    question: str
    answer: str

    got_graph: GoTGraph
    agents: List[Agent] = field(default_factory=list)
    memory_entries: List[MemoryEntry] = field(default_factory=list)
    retrieval_requests: List[RetrievalRequest] = field(default_factory=list)
    ground_truth: List[GroundTruthEvidence] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_runtime_objects(
        self,
    ) -> Tuple[List[Agent], List[MemoryEntry], List[RetrievalRequest]]:
        """Return (agents, memory_entries, retrieval_requests) for runtime use."""
        return self.agents, self.memory_entries, self.retrieval_requests
