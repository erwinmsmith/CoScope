"""
Runtime access controller.

Applies the double-layer access rule (scope + clearance) plus the Solver-only
ancestor-range filter on task-shared entries. See §13.1 / §13.2 of the spec.

All filters fail silently (by omitting the entry) - we never leak the reason
for rejection to avoid policy-oracle attacks.
"""

from __future__ import annotations

from typing import List, Set

from core.types import Agent, AgentRole, MemoryEntry
from core.types import GoTGraph


class AccessController:
    """Stateless filter that enforces scope + clearance + ancestor rules."""

    def filter_accessible_entries(
        self,
        agent: Agent,
        memory_entries: List[MemoryEntry],
        got_graph: GoTGraph,
    ) -> List[MemoryEntry]:
        allowed_scopes: Set[str] = set(agent.config.allowed_scopes)
        agent_clearance = int(agent.config.policy.max_clearance or 0)
        excluded_zones: Set[str] = set(agent.config.policy.excluded_zones or [])
        ancestor_ids = self._ancestor_set(agent, got_graph)

        result: List[MemoryEntry] = []
        for entry in memory_entries:
            if entry.scope_id not in allowed_scopes:
                continue
            meta = entry.metadata or {}
            layer = meta.get("scope_layer", "")

            # Solver-only ancestor-range filter on task-shared entries.
            if (
                agent.config.role == AgentRole.SOLVER
                and layer == "task_shared_episodic"
            ):
                if meta.get("source_node_id") not in ancestor_ids:
                    continue

            # Clearance check.
            required_clearance = int(meta.get("required_clearance", 0) or 0)
            if required_clearance > agent_clearance:
                continue

            # Quarantine / excluded zones (entry-level flag or explicit tag).
            if meta.get("quarantined") is True:
                continue
            if set(entry.tags) & excluded_zones:
                continue

            result.append(entry)
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _ancestor_set(agent: Agent, got_graph: GoTGraph) -> Set[str]:
        meta = agent.config.metadata or {}
        ids = meta.get("accessible_ancestor_node_ids")
        if isinstance(ids, (list, tuple, set)):
            return set(ids)
        node_id = meta.get("node_id")
        if not node_id:
            return set()
        return set(got_graph.get_all_ancestors(node_id)) | {node_id}
