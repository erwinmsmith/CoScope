"""
Restricted layer builder.

Restricted entries are generated offline (see `scripts/generate_restricted.py`)
and stored at `data/interim/restricted/{dataset}_restricted.jsonl`. The main
pipeline only *loads* them here; it never produces restricted content itself.

The offline file is a JSONL where each line is:
    {
        "original_id": str,
        "entries": [  # each dict hydrates into a MemoryEntry
            {
                "content": str,
                "confidence": float,
                "source": str,
                "metadata": { ... extra fields ... }
            },
            ...
        ]
    }

If the file is missing or there is no matching entry, an empty list is returned.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.types import (
    MemoryEntry,
    MemoryType,
    Provenance,
    VisibilityLevel,
)
from core.scope_ids import task_restricted


class RestrictedBuilder:
    """Loads pre-generated restricted-layer entries for an episode."""

    def __init__(self, interim_dir: Union[str, Path] = "data/interim/restricted"):
        self.interim_dir = Path(interim_dir)
        # dataset -> {original_id -> List[raw restricted record]}
        self._cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    # ------------------------------------------------------------------

    def load_from_interim(
        self, dataset: str, original_id: str, episode_id: str
    ) -> List[MemoryEntry]:
        records = self._lookup(dataset, original_id)
        if not records:
            return []
        scope_id = task_restricted(episode_id)
        entries: List[MemoryEntry] = []
        for record in records:
            entries.append(self._from_record(record, dataset, scope_id, episode_id))
        return entries

    # ------------------------------------------------------------------

    def _lookup(self, dataset: str, original_id: str) -> List[Dict[str, Any]]:
        index = self._cache.get(dataset)
        if index is None:
            index = self._load_index(dataset)
            self._cache[dataset] = index
        return index.get(str(original_id), [])

    def _load_index(self, dataset: str) -> Dict[str, List[Dict[str, Any]]]:
        path = self.interim_dir / f"{dataset}_restricted.jsonl"
        if not path.exists():
            return {}
        index: Dict[str, List[Dict[str, Any]]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(row.get("original_id", ""))
                if not key:
                    continue
                index.setdefault(key, []).extend(row.get("entries", []))
        return index

    # ------------------------------------------------------------------

    @staticmethod
    def _from_record(
        record: Dict[str, Any], dataset: str, scope_id: str, episode_id: str
    ) -> MemoryEntry:
        meta_extra = dict(record.get("metadata", {}) or {})
        meta = {
            "scope_layer": "restricted",
            "required_clearance": int(meta_extra.pop("required_clearance", 2)),
            "is_gold_evidence": False,
            "source_node_id": meta_extra.pop("source_node_id", "verifier"),
            "source_paragraph_id": meta_extra.pop("source_paragraph_id", record.get("source")),
            "dataset": dataset,
            "episode_id": episode_id,
        }
        meta.update(meta_extra)
        # Deterministic memory_id: restricted placeholders are episode-scoped
        # (different verifier runs for different episodes) so include episode_id.
        # Use source_paragraph_id when available; otherwise fall back to the
        # content hash so two distinct entries within one episode don't collide.
        content_str = str(record.get("content", "") or "")
        rs_basis = (
            meta.get("source_paragraph_id")
            or record.get("source")
            or hashlib.md5(content_str.encode()).hexdigest()[:8]
        )
        rs_key = f"rs|{episode_id}|{rs_basis}"
        memory_id = f"mem_rs_{hashlib.md5(rs_key.encode()).hexdigest()[:12]}"
        return MemoryEntry(
            memory_id=memory_id,
            scope_id=scope_id,
            memory_type=MemoryType.EPISODIC,
            content=content_str,
            visibility=[VisibilityLevel.RESTRICTED],
            confidence=float(record.get("confidence", 0.0)),
            provenance=Provenance(
                source=str(record.get("source", "")),
                agent_id=record.get("writer_agent_id") or "verifier",
            ),
            tags=["restricted", dataset],
            metadata=meta,
        )
