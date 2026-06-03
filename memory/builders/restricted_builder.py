"""
Restricted layer builder.

Restricted entries are generated offline (see `scripts/generate_restricted.py`)
and stored at `data/interim/restricted/{dataset}_restricted.jsonl`.

For smoke tests and early CoT handoff runs, this builder can also derive a
minimal restricted layer directly from the `raw_item` when no interim record is
available. That keeps POLICY_ISOLATED / S4 episodes structurally complete
before the offline restricted-generation pass has been run.

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

    # Process-wide cache keyed by (interim_dir, dataset) -> {original_id -> records}.
    # Hoisted from instance to class scope so workers that build many episodes
    # (e.g. ProcessPoolExecutor in DatasetPipeline, which constructs a fresh
    # EpisodeBuilder + RestrictedBuilder per episode) reuse a single in-memory
    # index instead of re-parsing the JSONL for every episode. The index is
    # purely a function of the file contents, so sharing across instances is
    # safe; the file is treated as read-only by this builder.
    _cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    def __init__(self, interim_dir: Union[str, Path] = "data/interim/restricted"):
        self.interim_dir = Path(interim_dir)

    # ------------------------------------------------------------------

    def load_from_interim(
        self, dataset: str, original_id: str, episode_id: str
    ) -> List[MemoryEntry]:
        records = self._lookup(dataset, original_id)
        if not records:
            return []
        return self._records_to_entries(records, dataset, episode_id)

    def build_from_raw_item(
        self, raw_item: Dict[str, Any], dataset: str, episode_id: str
    ) -> List[MemoryEntry]:
        """Derive a minimal restricted layer from the raw item itself.

        Priority:
        1. `raw_item["restricted_items"]` if explicitly provided.
        2. Deterministic QA / math fallback matching `generate_restricted.py`.
        """
        records = list(raw_item.get("restricted_items", []) or [])
        if not records:
            if raw_item.get("dataset_type") == "math":
                records = self._math_records(raw_item)
            else:
                records = self._qa_records(raw_item)
        if not records:
            return []
        return self._records_to_entries(records, dataset, episode_id)

    # ------------------------------------------------------------------

    def _records_to_entries(
        self, records: List[Dict[str, Any]], dataset: str, episode_id: str
    ) -> List[MemoryEntry]:
        scope_id = task_restricted(episode_id)
        entries: List[MemoryEntry] = []
        for record in records:
            entries.append(self._from_record(record, dataset, scope_id, episode_id))
        return entries

    # ------------------------------------------------------------------

    @staticmethod
    def _qa_records(raw_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for para in raw_item.get("supporting_paragraphs", []) or []:
            paragraph_id = str(para.get("paragraph_id", ""))
            if not paragraph_id:
                continue
            records.append(
                {
                    "content": f"{paragraph_id} credibility: 1.0 (gold evidence)",
                    "confidence": 1.0,
                    "source": paragraph_id,
                    "metadata": {
                        "source_paragraph_id": paragraph_id,
                        "kind": "gold_fact",
                    },
                }
            )
        for para in raw_item.get("distractor_paragraphs", []) or []:
            paragraph_id = str(para.get("paragraph_id", ""))
            if not paragraph_id:
                continue
            records.append(
                {
                    "content": f"{paragraph_id} credibility: 0.0 (distractor)",
                    "confidence": 0.0,
                    "source": paragraph_id,
                    "metadata": {
                        "source_paragraph_id": paragraph_id,
                        "kind": "distractor",
                    },
                }
            )
        return records

    @staticmethod
    def _math_records(raw_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for idx, step in enumerate(raw_item.get("solution_steps", []) or [], start=1):
            records.append(
                {
                    "content": f"step_{idx} numerical check: pass, value=<oracle>",
                    "confidence": 1.0,
                    "source": f"step_{idx}",
                    "metadata": {
                        "step_index": idx,
                        "kind": "numerical_check",
                        "step_text": str(step),
                    },
                }
            )
        return records

    # ------------------------------------------------------------------

    def _lookup(self, dataset: str, original_id: str) -> List[Dict[str, Any]]:
        # Include interim_dir in the cache key so multiple builders pointing at
        # different directories do not collide on the class-level cache.
        cache_key = f"{self.interim_dir}::{dataset}"
        index = self._cache.get(cache_key)
        if index is None:
            index = self._load_index(dataset)
            self._cache[cache_key] = index
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
        # Within one episode, a single source_paragraph_id can back multiple
        # distinct records (different 'kind' values: gold_fact, audit_note,
        # clearance_note, ...), so the paragraph id alone is NOT unique. We
        # therefore key on (episode_id, source_paragraph_id, content_hash) to
        # guarantee uniqueness inside an episode while remaining stable when
        # the same raw_item is rebuilt.
        content_str = str(record.get("content", "") or "")
        content_hash = hashlib.md5(content_str.encode()).hexdigest()[:8]
        rs_basis = (
            meta.get("source_paragraph_id")
            or record.get("source")
            or "no_source"
        )
        rs_key = f"rs|{episode_id}|{rs_basis}|{content_hash}"
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
