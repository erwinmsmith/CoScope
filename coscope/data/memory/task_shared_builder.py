"""
Task-shared / episodic layer builder.

For each Solver node in the graph we write (at most) one episodic entry that
mirrors the gold content of the corresponding hop / step. QA datasets use an
"oracle realtime" simulation (what Solver-k would eventually write); Math
datasets pre-fill the gold step conclusion.
"""

from __future__ import annotations

from typing import Any, Dict, List

from coscope.core.types import (
    MemoryEntry,
    MemoryType,
    Provenance,
    VisibilityLevel,
)
from coscope.data.core.types import GoTGraph, NodeType
from coscope.data.memory.scope_ids import task_shared


class TaskSharedBuilder:
    """Builds `task/{episode_id}/shared` MemoryEntry list."""

    def build(
        self,
        raw_item: Dict[str, Any],
        got_graph: GoTGraph,
        episode_id: str,
        dataset: str,
    ) -> List[MemoryEntry]:
        scope_id = task_shared(episode_id)
        # Index oracle task_shared_items by hop_index for quick lookup.
        items_by_hop: Dict[int, Dict[str, Any]] = {}
        for item in raw_item.get("task_shared_items", []) or []:
            hop = item.get("hop_index")
            if isinstance(hop, int):
                items_by_hop.setdefault(hop, item)

        entries: List[MemoryEntry] = []
        for node in got_graph.solver_nodes():
            hop = node.hop_index
            if not isinstance(hop, int):
                continue
            item = items_by_hop.get(hop)
            if item is None:
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            source_paragraph_id = str(item.get("source_paragraph_id", f"hop_{hop}"))
            writer_agent_id = f"{episode_id}_{node.node_id}"
            entries.append(
                MemoryEntry(
                    scope_id=scope_id,
                    memory_type=MemoryType.EPISODIC,
                    content=text,
                    visibility=[VisibilityLevel.TEAM],
                    confidence=1.0,
                    provenance=Provenance(
                        source=source_paragraph_id,
                        agent_id=writer_agent_id,
                    ),
                    tags=["task_shared", "episodic", dataset, f"hop_{hop}"],
                    metadata={
                        "scope_layer": "task_shared_episodic",
                        "is_gold_evidence": True,
                        "required_clearance": 0,
                        "source_node_id": node.node_id,
                        "source_paragraph_id": source_paragraph_id,
                        "hop_index": hop,
                        "dataset": dataset,
                        "episode_id": episode_id,
                    },
                )
            )
        return entries
