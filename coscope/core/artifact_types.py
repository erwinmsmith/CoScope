"""
Artifact schema conventions on top of coscope.core.types.MemoryEntry.

Defines ArtifactSlot enum and the three canonical mapping tables:
  slot -> scope_layer, memory_type, visibility.

These tables are the single source of truth; all builders and the rollout
engine read from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

from coscope.core.types import MemoryEntry, MemoryType, VisibilityLevel


class ArtifactSlot(str, Enum):
    """Canonical artifact slot names produced by a single GoT node."""

    PLAN = "plan"
    QUERY_INTENT = "query_intent"
    SCRATCH = "scratch"
    CONCLUSION = "conclusion"
    AUDIT_REPORT = "audit_report"


SCOPE_LAYER_BY_SLOT: Dict[ArtifactSlot, str] = {
    ArtifactSlot.PLAN:         "task_shared_plan",
    ArtifactSlot.QUERY_INTENT: "agent_private_intent",
    ArtifactSlot.SCRATCH:      "agent_private",
    ArtifactSlot.CONCLUSION:   "task_shared_artifact",
    ArtifactSlot.AUDIT_REPORT: "restricted_audit",
}

MEMORY_TYPE_BY_SLOT: Dict[ArtifactSlot, MemoryType] = {
    ArtifactSlot.PLAN:         MemoryType.ARTIFACT,
    ArtifactSlot.QUERY_INTENT: MemoryType.EPISODIC,
    ArtifactSlot.SCRATCH:      MemoryType.EPISODIC,
    ArtifactSlot.CONCLUSION:   MemoryType.ARTIFACT,
    ArtifactSlot.AUDIT_REPORT: MemoryType.ARTIFACT,
}

VISIBILITY_BY_SLOT: Dict[ArtifactSlot, Tuple[VisibilityLevel, ...]] = {
    ArtifactSlot.PLAN:         (VisibilityLevel.TEAM,),
    ArtifactSlot.QUERY_INTENT: (VisibilityLevel.OWNER,),
    ArtifactSlot.SCRATCH:      (VisibilityLevel.OWNER,),
    ArtifactSlot.CONCLUSION:   (VisibilityLevel.TEAM,),
    ArtifactSlot.AUDIT_REPORT: (VisibilityLevel.RESTRICTED,),
}


@dataclass
class ArtifactTrace:
    """
    Aggregate result of one rollout.

    Carried inside the construction pipeline only. Downstream consumers see
    List[MemoryEntry] via Episode.memory_entries.
    """

    episode_id: str
    entries: List[MemoryEntry] = field(default_factory=list)
    producer_index: Dict[str, List[str]] = field(default_factory=dict)
    consumer_index: Dict[str, List[str]] = field(default_factory=dict)
    stream: List[Tuple[int, str]] = field(default_factory=list)
    rollout_meta: Dict[str, object] = field(default_factory=dict)
