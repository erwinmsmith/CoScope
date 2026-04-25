"""
Rho calculator (v2.1).

rho is the Jaccard similarity across **Solver**-accessible MemoryEntry sets,
computed statically from the GoTGraph ancestor relation and the precomputed
memory store. See `docs/GoT_Dataset_Requirements_v2.md` §9.1.

Accessibility model (v2.1-aligned):

- Only SOLVER nodes participate in rho. Planner and Verifier are excluded:
  Planner always sees everything (does not reflect structural differences) and
  Verifier's policy isolation is handled via `s4_eligible`.
- For Solver-k:
    * `workspace_semantic_hop` entries: accessible iff `hop_index == k`.
    * `task_shared_episodic` entries: accessible iff `source_node_id` is in the
      Solver's ancestor-or-self set.
    * `workspace_semantic_global` entries: NOT counted (exclusive to Planner /
      Verifier; including them would swamp the intersection as in v2.0).
    * `agent_private` / `restricted`: not counted.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

from core.types import MemoryEntry
from core.types import GoTGraph


class RhoCalculator:
    """Static rho computation with optional JSON cache."""

    def __init__(self, cache_dir: Union[str, Path, None] = "data/interim/rho_cache"):
        self.cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None
        self._cache_mem: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------

    def compute(
        self,
        got_graph: GoTGraph,
        memory_entries: Iterable[MemoryEntry],
        dataset: str,
        episode_id: str,
        use_cache: bool = True,
    ) -> float:
        if use_cache:
            cached = self._get_cached(dataset, episode_id)
            if cached is not None:
                return float(cached)

        rho = self._compute_rho(got_graph, list(memory_entries), dataset)

        if use_cache:
            self._set_cached(dataset, episode_id, rho, got_graph.graph_type)
        return rho

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rho(
        got_graph: GoTGraph, memory_entries: List[MemoryEntry], dataset: str
    ) -> float:
        accessible_sets: List[Set[str]] = []
        for node in got_graph.solver_nodes():
            hop_k = node.hop_index
            ancestors = set(got_graph.get_all_ancestors(node.node_id))
            ancestors.add(node.node_id)
            accessible = _accessible_memory_ids_for_solver(
                memory_entries, ancestors, hop_k
            )
            accessible_sets.append(accessible)

        if not accessible_sets:
            return 0.0
        if len(accessible_sets) == 1:
            # Single Solver: rho is trivially 1 (only one set). Return 1.0
            # if the set is non-empty to flag the degenerate graph.
            return 1.0 if accessible_sets[0] else 0.0

        # Pairwise mean IoU (matches coscope/rollout/rho_calculator.py). Strict
        # full-set Jaccard collapses to 0 whenever any sibling-solver pair has
        # disjoint ancestors (FORK / FORK_MERGE / INDEPENDENT), producing a
        # bimodal {0, 0.5} distribution that cannot span three rho buckets.
        # Averaging over pairs lets partial overlaps (e.g. a merge node that
        # consumes from two branches) raise rho smoothly.
        ious: List[float] = []
        for a, b in combinations(accessible_sets, 2):
            union = a | b
            if not union:
                continue
            ious.append(len(a & b) / len(union))
        if not ious:
            return 0.0
        return round(sum(ious) / len(ious), 4)

    # --- caching --------------------------------------------------------

    def _cache_path(self, dataset: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{dataset}_rho_cache.json"

    def _load_cache(self, dataset: str) -> Dict[str, Any]:
        if dataset in self._cache_mem:
            return self._cache_mem[dataset]
        path = self._cache_path(dataset)
        if path is None or not path.exists():
            data = {"version": "2.0", "dataset": dataset, "entries": {}}
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {"version": "2.0", "dataset": dataset, "entries": {}}
        self._cache_mem[dataset] = data
        return data

    def _get_cached(self, dataset: str, episode_id: str) -> Optional[float]:
        cache = self._load_cache(dataset)
        entry = cache.get("entries", {}).get(episode_id)
        if entry is None:
            return None
        return entry.get("rho")

    def _set_cached(self, dataset: str, episode_id: str, rho: float, graph_type) -> None:
        cache = self._load_cache(dataset)
        cache.setdefault("entries", {})[episode_id] = {
            "rho": rho,
            "graph_type": graph_type.value if hasattr(graph_type, "value") else str(graph_type),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    def flush(self) -> None:
        """Persist any in-memory cache updates to disk."""
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for dataset, data in self._cache_mem.items():
            path = self._cache_path(dataset)
            if path is None:
                continue
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# Accessibility helper
# ============================================================


def _accessible_memory_ids_for_solver(
    memory_entries: List[MemoryEntry],
    ancestor_and_self_ids: Set[str],
    hop_k: Optional[int],
) -> Set[str]:
    """
    Return the set of memory_ids accessible to Solver-k per v2.1 rules.

    Only two categories are counted:
      1. `workspace_semantic_hop` entries whose metadata hop_index == hop_k.
      2. `task_shared_episodic` entries whose source_node_id is an ancestor or
         the Solver itself.
    """
    accessible: Set[str] = set()
    for entry in memory_entries:
        meta = entry.metadata or {}
        layer = meta.get("scope_layer", "")
        if layer == "workspace_semantic_hop":
            if meta.get("hop_index") == hop_k:
                accessible.add(entry.memory_id)
        elif layer == "task_shared_episodic":
            if meta.get("source_node_id") in ancestor_and_self_ids:
                accessible.add(entry.memory_id)
        # workspace_semantic_global / agent_private / restricted: excluded.
    return accessible
