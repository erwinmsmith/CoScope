"""
Agent-private layer builder.

Each Planner / Solver / Verifier node gets a single placeholder MemoryEntry in
its own private scope. The runtime will eventually populate real content; the
construction pipeline only seeds a structural placeholder so that subsequent
writers have a scope to target.
"""

from __future__ import annotations

from typing import List

from coscope.core.types import (
    MemoryEntry,
    MemoryType,
    Provenance,
    VisibilityLevel,
)
from coscope.core.types import GoTGraph
from coscope.core.scope_ids import agent_private


class PrivateBuilder:
    """Builds agent-private placeholders for every node in the graph."""

    def build(self, got_graph: GoTGraph, episode_id: str, dataset: str) -> List[MemoryEntry]:
        entries: List[MemoryEntry] = []
        for node in got_graph.nodes:
            agent_id = f"{episode_id}_{node.node_id}"
            scope_id = agent_private(episode_id, node.node_id)
            entries.append(
                MemoryEntry(
                    scope_id=scope_id,
                    memory_type=MemoryType.EPISODIC,
                    content=f"[placeholder: {agent_id}_private_reasoning]",
                    visibility=[VisibilityLevel.OWNER],
                    confidence=0.0,
                    provenance=Provenance(source="placeholder", agent_id=agent_id),
                    tags=["agent_private", "placeholder", dataset],
                    metadata={
                        "scope_layer": "agent_private",
                        "is_gold_evidence": False,
                        "required_clearance": 0,
                        "source_node_id": node.node_id,
                        "source_paragraph_id": None,
                        "owner_agent_id": agent_id,
                        "dataset": dataset,
                        "episode_id": episode_id,
                    },
                )
            )
        return entries
