"""
Helper: turn (slot, node, content, parents) into a fully populated MemoryEntry.

Centralizes the slot -> scope_id / memory_type / visibility / metadata mapping
so that `rollout_engine.py` never constructs a MemoryEntry by hand.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from core.types import MemoryEntry, Provenance, VisibilityLevel
from core.types import GoTNode
from core.scope_ids import (
    agent_private,
    task_restricted,
    task_shared,
)
from core.artifact_types import (
    ArtifactSlot,
    MEMORY_TYPE_BY_SLOT,
    SCOPE_LAYER_BY_SLOT,
    VISIBILITY_BY_SLOT,
)


# Deterministic epoch so that artifact timestamps are reproducible across runs.
_DETERMINISTIC_EPOCH = datetime(2025, 1, 1, tzinfo=timezone.utc)


def scope_id_for(slot: ArtifactSlot, *, episode_id: str, node_id: str) -> str:
    """Return the canonical scope_id for a slot."""
    layer = SCOPE_LAYER_BY_SLOT[slot]
    if layer in {"agent_private", "agent_private_intent"}:
        return agent_private(episode_id, node_id)
    if layer in {"task_shared_plan", "task_shared_artifact"}:
        return task_shared(episode_id)
    if layer == "restricted_audit":
        return task_restricted(episode_id)
    raise ValueError(f"Unknown scope layer for slot {slot}: {layer!r}")


def write_artifact(
    *,
    slot: ArtifactSlot,
    node: GoTNode,
    content: str,
    dataset: str,
    episode_id: str,
    topo_index: int,
    parent_artifact_ids: Optional[List[str]] = None,
    is_gold_evidence: bool = False,
    required_clearance: int = 0,
    llm_name: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> MemoryEntry:
    """
    Construct a MemoryEntry for one artifact emission.

    Does not perform IO; caller appends the returned entry into its local list
    and records producer/consumer relations.
    """
    parents = list(parent_artifact_ids or [])
    agent_id = f"{episode_id}_{node.node_id}"
    source_tag = f"llm:{llm_name}" if llm_name else "template:v1"

    provenance = Provenance(
        source=source_tag,
        parent_event=parents[0] if parents else None,
        agent_id=agent_id,
        timestamp=_DETERMINISTIC_EPOCH + timedelta(seconds=topo_index),
        version=1,
        metadata={"all_parent_artifact_ids": parents},
    )

    metadata = {
        "scope_layer": SCOPE_LAYER_BY_SLOT[slot],
        "slot": slot.value,
        "source_node_id": node.node_id,
        "hop_index": node.hop_index,
        "topo_index": int(topo_index),
        "parent_artifact_ids": parents,
        "is_gold_evidence": bool(is_gold_evidence),
        "required_clearance": int(required_clearance),
        "dataset": dataset,
        "episode_id": episode_id,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return MemoryEntry(
        scope_id=scope_id_for(slot, episode_id=episode_id, node_id=node.node_id),
        memory_type=MEMORY_TYPE_BY_SLOT[slot],
        content=content,
        visibility=list(VISIBILITY_BY_SLOT[slot]),
        confidence=1.0 if is_gold_evidence else 0.8,
        provenance=provenance,
        tags=["rollout", slot.value, dataset],
        metadata=metadata,
    )
