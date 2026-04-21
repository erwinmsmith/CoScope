"""
Workspace / semantic layer builder (v2.1: two-tier).

Per v2.1 §7.1 / §7.2 we now write workspace entries at two scope levels:

- Global workspace (`workspace/{dataset}/semantic`)
  - Accessed by Planner and Verifier.
  - Contains all supporting + distractor paragraphs (QA) or all known
    quantities (Math). scope_layer = "workspace_semantic_global".

- Hop-level workspace (`workspace/{dataset}/semantic/hop_{k}`)
  - Accessed exclusively by Solver-k (hop_index == k).
  - Contains only the supporting paragraphs / formulas for hop k.
  - scope_layer = "workspace_semantic_hop", metadata["hop_index"] == k.

For each gold paragraph that has a `hop_index`, *two* MemoryEntry rows are
emitted (one in the global scope, one in the hop scope). Distractor paragraphs
only exist at the global scope. The two-tier design is what lets rho differ
meaningfully across GoT graph types.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from coscope.core.types import (
    MemoryEntry,
    MemoryType,
    Provenance,
    VisibilityLevel,
)
from coscope.data.memory.scope_ids import workspace_semantic, workspace_semantic_hop


class WorkspaceBuilder:
    """Constructs global + hop-level workspace MemoryEntry lists for an episode."""

    def build(
        self, raw_item: Dict[str, Any], dataset: str, episode_id: str
    ) -> List[MemoryEntry]:
        entries: List[MemoryEntry] = []
        global_scope = workspace_semantic(dataset)

        # Supporting paragraphs: global + hop (when hop_index is known).
        for para in raw_item.get("supporting_paragraphs", []) or []:
            hop_index = _coerce_hop_index(para.get("hop_index"))
            # (1) Global entry
            entries.append(
                self._make_entry(
                    para,
                    dataset=dataset,
                    episode_id=episode_id,
                    scope_id=global_scope,
                    scope_layer="workspace_semantic_global",
                    hop_index=hop_index,
                    is_gold=True,
                )
            )
            # (2) Hop-level entry (only when hop_index is known)
            if hop_index is not None:
                entries.append(
                    self._make_entry(
                        para,
                        dataset=dataset,
                        episode_id=episode_id,
                        scope_id=workspace_semantic_hop(dataset, hop_index),
                        scope_layer="workspace_semantic_hop",
                        hop_index=hop_index,
                        is_gold=True,
                    )
                )

        # Distractors: global only.
        for para in raw_item.get("distractor_paragraphs", []) or []:
            entries.append(
                self._make_entry(
                    para,
                    dataset=dataset,
                    episode_id=episode_id,
                    scope_id=global_scope,
                    scope_layer="workspace_semantic_global",
                    hop_index=None,
                    is_gold=False,
                )
            )
        return entries

    # ------------------------------------------------------------------

    @staticmethod
    def _make_entry(
        para: Dict[str, Any],
        *,
        dataset: str,
        episode_id: str,
        scope_id: str,
        scope_layer: str,
        hop_index: Optional[int],
        is_gold: bool,
    ) -> MemoryEntry:
        paragraph_id = str(para.get("paragraph_id", f"para_{abs(hash(para.get('text', '')))%100000:05d}"))
        text = str(para.get("text", "") or "")
        return MemoryEntry(
            scope_id=scope_id,
            memory_type=MemoryType.SEMANTIC,
            content=text,
            visibility=[VisibilityLevel.PUBLIC],
            confidence=1.0 if is_gold else 0.5,
            provenance=Provenance(source=paragraph_id, agent_id=None),
            tags=["workspace", "semantic", dataset],
            metadata={
                "scope_layer": scope_layer,
                "hop_index": hop_index,
                "is_gold_evidence": bool(is_gold),
                "required_clearance": 0,
                "source_node_id": None,
                "source_paragraph_id": paragraph_id,
                "title": para.get("title", ""),
                "dataset": dataset,
                "episode_id": episode_id,
            },
        )


def _coerce_hop_index(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
