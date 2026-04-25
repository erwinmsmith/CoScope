"""
GoT graph templates.

Each template is a function `(hop_count: int) -> GoTGraph` that returns a fresh
graph instance for the requested parameters. Templates intentionally construct
both the node list and edge list, so the resulting GoTGraph is ready to be
validated by `GoTGraph.validate()`.

Design decisions (see `coscope/data/docs/GoT_Dataset_Requirements_v2.md` and the
Cascade session notes for context):

- `POLICY_ISOLATED` is the only graph type that adds a Verifier node.
  For all other graph types we construct a Planner + Solvers DAG only.
- `policy_conflict` / `s4_eligible` are driven by graph_type == POLICY_ISOLATED,
  not by structural detection. This resolves the contradiction between
  section 6.1 ("every episode has a Verifier") and section 10.1
  (`_has_policy_conflict` would then always be True).
- `len(solver_nodes)` may exceed `hop_count` for FORK / FORK_MERGE / INDEPENDENT
  layouts that fan out multiple solvers per hop. The "agent_count == hop_count+2"
  constraint is relaxed to "agent_count >= hop_count + (1 if no verifier else 2)".
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from core.types import (
    EdgeType,
    GoTEdge,
    GoTGraph,
    GoTNode,
    GraphType,
    NodeType,
)


# ============================================================
# Node / edge helpers
# ============================================================


def _planner() -> GoTNode:
    return GoTNode(
        node_id="planner",
        node_type=NodeType.PLANNER.value,
        agent_role="planner",
        hop_index=None,
    )


def _solver(node_id: str, hop_index: int, parents: List[str]) -> GoTNode:
    return GoTNode(
        node_id=node_id,
        node_type=NodeType.SOLVER.value,
        agent_role="solver",
        hop_index=hop_index,
        parent_node_ids=list(parents),
    )


def _verifier() -> GoTNode:
    return GoTNode(
        node_id="verifier",
        node_type=NodeType.VERIFIER.value,
        agent_role="verifier",
        hop_index=None,
    )


def _edge(frm: str, to: str, edge_type: str = EdgeType.DEPENDS_ON.value) -> GoTEdge:
    return GoTEdge(from_node=frm, to_node=to, edge_type=edge_type)


def _wire_children(nodes: List[GoTNode]) -> None:
    """Populate `child_node_ids` from `parent_node_ids` in-place."""
    idx = {n.node_id: n for n in nodes}
    for child in nodes:
        for parent_id in child.parent_node_ids:
            parent = idx.get(parent_id)
            if parent is not None and child.node_id not in parent.child_node_ids:
                parent.child_node_ids.append(child.node_id)


def _finalize(
    nodes: List[GoTNode], edges: List[GoTEdge], graph_type: GraphType
) -> GoTGraph:
    _wire_children(nodes)
    graph = GoTGraph(nodes=nodes, edges=edges, graph_type=graph_type)
    graph.validate()
    return graph


# ============================================================
# Pattern builders (structural only - no Verifier)
# ============================================================


def _build_linear(hop_count: int) -> Tuple[List[GoTNode], List[GoTEdge]]:
    hop_count = max(hop_count, 2)
    nodes: List[GoTNode] = [_planner()]
    edges: List[GoTEdge] = []
    prev_id = "planner"
    for k in range(1, hop_count + 1):
        node_id = f"solver_{k}"
        nodes.append(_solver(node_id, hop_index=k, parents=[prev_id]))
        edges.append(_edge(prev_id, node_id))
        prev_id = node_id
    return nodes, edges


def _build_fork(hop_count: int) -> Tuple[List[GoTNode], List[GoTEdge]]:
    """
    FORK: planner -> solver_1 -> (solver_2a, solver_2b) [no merge]
    For hop_count >= 3, the fork happens at hop 2, then continues linearly
    along the `a` branch to finish remaining hops.
    """
    hop_count = max(hop_count, 2)
    nodes: List[GoTNode] = [_planner()]
    edges: List[GoTEdge] = []
    if hop_count == 2:
        nodes.append(_solver("solver_1a", hop_index=1, parents=["planner"]))
        nodes.append(_solver("solver_1b", hop_index=1, parents=["planner"]))
        edges.append(_edge("planner", "solver_1a"))
        edges.append(_edge("planner", "solver_1b"))
        return nodes, edges
    # hop_count >= 3
    nodes.append(_solver("solver_1", hop_index=1, parents=["planner"]))
    edges.append(_edge("planner", "solver_1"))
    nodes.append(_solver("solver_2a", hop_index=2, parents=["solver_1"]))
    nodes.append(_solver("solver_2b", hop_index=2, parents=["solver_1"]))
    edges.append(_edge("solver_1", "solver_2a"))
    edges.append(_edge("solver_1", "solver_2b"))
    prev_id = "solver_2a"
    for k in range(3, hop_count + 1):
        node_id = f"solver_{k}"
        nodes.append(_solver(node_id, hop_index=k, parents=[prev_id]))
        edges.append(_edge(prev_id, node_id))
        prev_id = node_id
    return nodes, edges


def _build_merge(hop_count: int) -> Tuple[List[GoTNode], List[GoTEdge]]:
    """
    MERGE: planner -> (solver_1, solver_2) -> solver_3 -> ... -> solver_h
    Works for hop_count >= 3.
    """
    hop_count = max(hop_count, 3)
    nodes: List[GoTNode] = [_planner()]
    edges: List[GoTEdge] = []
    nodes.append(_solver("solver_1", hop_index=1, parents=["planner"]))
    nodes.append(_solver("solver_2", hop_index=2, parents=["planner"]))
    edges.append(_edge("planner", "solver_1"))
    edges.append(_edge("planner", "solver_2"))
    # Merge hop
    merge_id = "solver_3"
    nodes.append(
        _solver(merge_id, hop_index=3, parents=["solver_1", "solver_2"])
    )
    edges.append(_edge("solver_1", merge_id, EdgeType.AGGREGATES.value))
    edges.append(_edge("solver_2", merge_id, EdgeType.AGGREGATES.value))
    prev_id = merge_id
    for k in range(4, hop_count + 1):
        node_id = f"solver_{k}"
        nodes.append(_solver(node_id, hop_index=k, parents=[prev_id]))
        edges.append(_edge(prev_id, node_id))
        prev_id = node_id
    return nodes, edges


def _build_fork_merge(hop_count: int) -> Tuple[List[GoTNode], List[GoTEdge]]:
    """
    FORK_MERGE: planner -> solver_1 -> (solver_2a, solver_2b) -> solver_3
    If hop_count > 3, the tail continues linearly from solver_3.
    If hop_count == 2, degenerate to:
        planner -> (solver_1a, solver_1b) -> solver_2 (merge).
    """
    hop_count = max(hop_count, 2)
    nodes: List[GoTNode] = [_planner()]
    edges: List[GoTEdge] = []
    if hop_count == 2:
        nodes.append(_solver("solver_1a", hop_index=1, parents=["planner"]))
        nodes.append(_solver("solver_1b", hop_index=1, parents=["planner"]))
        nodes.append(
            _solver("solver_2", hop_index=2, parents=["solver_1a", "solver_1b"])
        )
        edges.append(_edge("planner", "solver_1a"))
        edges.append(_edge("planner", "solver_1b"))
        edges.append(_edge("solver_1a", "solver_2", EdgeType.AGGREGATES.value))
        edges.append(_edge("solver_1b", "solver_2", EdgeType.AGGREGATES.value))
        return nodes, edges
    # hop_count >= 3
    nodes.append(_solver("solver_1", hop_index=1, parents=["planner"]))
    nodes.append(_solver("solver_2a", hop_index=2, parents=["solver_1"]))
    nodes.append(_solver("solver_2b", hop_index=2, parents=["solver_1"]))
    nodes.append(
        _solver("solver_3", hop_index=3, parents=["solver_2a", "solver_2b"])
    )
    edges.append(_edge("planner", "solver_1"))
    edges.append(_edge("solver_1", "solver_2a"))
    edges.append(_edge("solver_1", "solver_2b"))
    edges.append(_edge("solver_2a", "solver_3", EdgeType.AGGREGATES.value))
    edges.append(_edge("solver_2b", "solver_3", EdgeType.AGGREGATES.value))
    prev_id = "solver_3"
    for k in range(4, hop_count + 1):
        node_id = f"solver_{k}"
        nodes.append(_solver(node_id, hop_index=k, parents=[prev_id]))
        edges.append(_edge(prev_id, node_id))
        prev_id = node_id
    return nodes, edges


def _build_independent(hop_count: int) -> Tuple[List[GoTNode], List[GoTEdge]]:
    """
    INDEPENDENT: planner -> solver_1a, planner -> solver_1b, with two
    disjoint chains continuing for hop_count hops each. Produces low rho.
    """
    hop_count = max(hop_count, 2)
    nodes: List[GoTNode] = [_planner()]
    edges: List[GoTEdge] = []
    for branch in ("a", "b"):
        prev_id = "planner"
        for k in range(1, hop_count + 1):
            node_id = f"solver_{k}{branch}"
            nodes.append(_solver(node_id, hop_index=k, parents=[prev_id]))
            edges.append(_edge(prev_id, node_id))
            prev_id = node_id
    return nodes, edges


# ============================================================
# Public builder
# ============================================================


# (dataset, graph_type) -> structural builder
_BUILDERS: Dict[Tuple[str, GraphType], Callable[[int], Tuple[List[GoTNode], List[GoTEdge]]]] = {}


def _register(dataset: str, graph_type: GraphType, builder: Callable[[int], Tuple[List[GoTNode], List[GoTEdge]]]) -> None:
    _BUILDERS[(dataset, graph_type)] = builder


# The same structural builders apply across datasets; specializations can be
# introduced later by re-registering a (dataset, graph_type) pair.
_REGISTERED_DATASETS = (
    "musique",
    "2wikimhqa",
    "2wikimultihopqa",  # alias matching raw dir + config key
    "hotpotqa",
    "gsm8k",
    "math",
)

for _ds in _REGISTERED_DATASETS:
    _register(_ds, GraphType.LINEAR, _build_linear)
    _register(_ds, GraphType.FORK, _build_fork)
    _register(_ds, GraphType.INDEPENDENT, _build_independent)

# MERGE / FORK_MERGE are primarily used by datasets that benefit from multi-hop
# aggregation; keep available everywhere but expect callers to pick per spec.
for _ds in _REGISTERED_DATASETS:
    _register(_ds, GraphType.MERGE, _build_merge)
    _register(_ds, GraphType.FORK_MERGE, _build_fork_merge)


def build_graph(dataset: str, hop_count: int, graph_type: GraphType) -> GoTGraph:
    """
    Produce a fresh, validated GoTGraph for the given parameters.

    POLICY_ISOLATED requires a base structure; we default to LINEAR.
    """
    if graph_type == GraphType.POLICY_ISOLATED:
        nodes, edges = _build_linear(hop_count)
        nodes.append(_verifier())
        return _finalize(nodes, edges, GraphType.POLICY_ISOLATED)

    key = (dataset, graph_type)
    builder = _BUILDERS.get(key)
    if builder is None:
        raise KeyError(
            f"No GoT template registered for dataset={dataset!r} graph_type={graph_type!r}"
        )
    nodes, edges = builder(hop_count)
    return _finalize(nodes, edges, graph_type)


def available_graph_types(dataset: str) -> List[GraphType]:
    """Return the graph types registered for a given dataset, plus POLICY_ISOLATED."""
    types = [gt for (ds, gt) in _BUILDERS.keys() if ds == dataset]
    types.append(GraphType.POLICY_ISOLATED)
    # Preserve declaration order.
    seen = []
    for t in types:
        if t not in seen:
            seen.append(t)
    return seen
