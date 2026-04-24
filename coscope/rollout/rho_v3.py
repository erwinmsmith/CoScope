"""
rho v3 — structural Jaccard over artifact trace.

Definition (see design doc §6 / flow doc §Step 4):

  For each solver-pair (i, j):
    A_i = {plan memory_id} ∪ {conclusion.memory_id | produced by ancestor_or_self(solver_i)}
  rho = mean pairwise IoU over all solver pairs.

Workspace (corpus) entries are excluded; they are equally accessible to all
solvers through the flat RAG layer and would dominate the IoU denominator.

Private artifacts (QUERY_INTENT, SCRATCH) are excluded because they are
per-agent unique and would uniformly reduce IoU without reflecting structural
overlap.

This module operates on an ArtifactTrace rather than a raw memory_entries
list, so that the producer_index / consumer_index already maintained by the
rollout engine can be reused.
"""

from __future__ import annotations

from itertools import combinations
from typing import Set

from coscope.core.types import GoTGraph
from coscope.core.artifact_types import ArtifactSlot, ArtifactTrace


def compute_rho(trace: ArtifactTrace, got_graph: GoTGraph) -> float:
    """Compute rho over a finished rollout trace."""
    # Build slot index: slot -> {producer_node_id -> memory_id}
    by_slot: dict = {}
    for e in trace.entries:
        meta = e.metadata or {}
        by_slot.setdefault(meta.get("slot", ""), {})[meta.get("source_node_id", "")] = e.memory_id

    plan_ids: Set[str] = set(by_slot.get(ArtifactSlot.PLAN.value, {}).values())
    concl_by_node = by_slot.get(ArtifactSlot.CONCLUSION.value, {})

    solver_sets: list = []
    for node in got_graph.solver_nodes():
        ancestors_or_self = set(got_graph.get_all_ancestors(node.node_id))
        ancestors_or_self.add(node.node_id)
        accessible = set(plan_ids)
        for nid in ancestors_or_self:
            cid = concl_by_node.get(nid)
            if cid:
                accessible.add(cid)
        solver_sets.append(accessible)

    if len(solver_sets) == 0:
        return 0.0
    if len(solver_sets) == 1:
        return 1.0 if solver_sets[0] else 0.0

    ious = []
    for a, b in combinations(solver_sets, 2):
        union = a | b
        if not union:
            ious.append(0.0)
        else:
            ious.append(len(a & b) / len(union))
    if not ious:
        return 0.0
    return round(sum(ious) / len(ious), 4)
