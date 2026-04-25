"""
Trace DAG validator.

Enforces the invariants listed in §5 of the design doc. Pure function over
an ArtifactTrace; no IO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from core.types import GoTGraph, GraphType
from core.artifact_types import ArtifactSlot, ArtifactTrace


@dataclass
class TraceValidationResult:
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_trace(trace: ArtifactTrace, got_graph: GoTGraph) -> TraceValidationResult:
    """Check invariants per design doc §5."""
    errors: List[str] = []
    warnings: List[str] = []

    valid_slots = {s.value for s in ArtifactSlot}

    # 1. Unique memory_ids
    seen_ids = set()
    for e in trace.entries:
        if e.memory_id in seen_ids:
            errors.append(f"duplicate memory_id: {e.memory_id}")
        seen_ids.add(e.memory_id)

    # 2. Each entry has a recognized slot
    id_to_entry = {e.memory_id: e for e in trace.entries}
    for e in trace.entries:
        slot = (e.metadata or {}).get("slot")
        if slot not in valid_slots:
            errors.append(f"entry {e.memory_id} has invalid slot {slot!r}")

    # 3. Stream ordering matches topo_index
    for idx, (topo, mid) in enumerate(trace.stream):
        if mid not in id_to_entry:
            errors.append(f"stream references unknown memory_id {mid}")
    prev_topo = -1
    for topo, _ in trace.stream:
        if topo < prev_topo:
            errors.append("stream is not monotonically ordered by topo_index")
            break
        prev_topo = topo

    # 4. task_shared artifacts never carry OWNER visibility
    from core.types import VisibilityLevel
    for e in trace.entries:
        layer = (e.metadata or {}).get("scope_layer", "")
        if layer.startswith("task_shared") and VisibilityLevel.OWNER in e.visibility:
            errors.append(f"task_shared entry {e.memory_id} has OWNER visibility")

    # 5. Parent artifacts must exist
    for e in trace.entries:
        parents = (e.metadata or {}).get("parent_artifact_ids", []) or []
        for p in parents:
            if p not in id_to_entry:
                errors.append(
                    f"entry {e.memory_id} references missing parent {p}"
                )

    # 6. For each Solver, conclusion's parents should include each graph-parent's conclusion
    solver_to_conclusion: dict = {}
    for e in trace.entries:
        meta = e.metadata or {}
        if meta.get("slot") == ArtifactSlot.CONCLUSION.value:
            solver_to_conclusion[meta.get("source_node_id")] = e.memory_id
    for node in got_graph.solver_nodes():
        concl_id = solver_to_conclusion.get(node.node_id)
        if concl_id is None:
            errors.append(f"solver node {node.node_id} missing conclusion")
            continue
        parents_meta = id_to_entry[concl_id].metadata.get("parent_artifact_ids", []) or []
        for p_node in node.parent_node_ids:
            if p_node == "planner":
                continue  # plan is a separate artifact; dependency via plan is fine
            p_concl = solver_to_conclusion.get(p_node)
            if p_concl and p_concl not in parents_meta:
                warnings.append(
                    f"conclusion of {node.node_id} does not reference parent {p_node}'s conclusion"
                )

    # 7. AUDIT_REPORT iff POLICY_ISOLATED
    has_audit = any(
        (e.metadata or {}).get("slot") == ArtifactSlot.AUDIT_REPORT.value
        for e in trace.entries
    )
    if got_graph.graph_type == GraphType.POLICY_ISOLATED and not has_audit:
        errors.append("POLICY_ISOLATED graph missing AUDIT_REPORT")
    if has_audit and got_graph.graph_type != GraphType.POLICY_ISOLATED:
        warnings.append("AUDIT_REPORT present on non-POLICY_ISOLATED graph")

    return TraceValidationResult(
        passed=(len(errors) == 0), errors=errors, warnings=warnings
    )
