"""
Workspace / semantic layer builder (v3: flat).

A single flat corpus scope per dataset:

    workspace/{dataset}/semantic        scope_layer = "workspace_semantic"

Contains all supporting + distractor paragraphs (QA) or all known quantities
(Math). Accessible to every agent on every episode of this dataset; the
per-hop partitioning from v2.1 has been removed (decision D1).

Hop information is still preserved in `metadata["hop_index"]` for gold
paragraphs so that downstream recall / coverage metrics can bucket by hop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.types import (
    MemoryEntry,
    MemoryType,
    Provenance,
    VisibilityLevel,
)
from core.scope_ids import workspace_semantic


class WorkspaceBuilder:
    """Constructs the flat workspace MemoryEntry list for an episode."""

    def build(
        self, raw_item: Dict[str, Any], dataset: str, episode_id: str
    ) -> List[MemoryEntry]:
        entries: List[MemoryEntry] = []
        flat_scope = workspace_semantic(dataset)

        for para in raw_item.get("supporting_paragraphs", []) or []:
            hop_index = _coerce_hop_index(para.get("hop_index"))
            entries.append(
                self._make_entry(
                    para,
                    dataset=dataset,
                    episode_id=episode_id,
                    scope_id=flat_scope,
                    scope_layer="workspace_semantic",
                    hop_index=hop_index,
                    is_gold=True,
                )
            )

        for para in raw_item.get("distractor_paragraphs", []) or []:
            entries.append(
                self._make_entry(
                    para,
                    dataset=dataset,
                    episode_id=episode_id,
                    scope_id=flat_scope,
                    scope_layer="workspace_semantic",
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
