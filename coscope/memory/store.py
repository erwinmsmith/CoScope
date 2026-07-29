"""In-process runtime memory with scope-first vector search."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from coscope.core.artifact import ArtifactState
from coscope.core.memory import MemoryEntry
from coscope.scope.effective_view import EffectiveView


def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b):
        raise ValueError("vector dimensions differ")
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if not norm_a or not norm_b:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)


@dataclass(frozen=True)
class SearchHit:
    memory: MemoryEntry
    score: float


class RuntimeMemoryStore:
    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._scope_index: dict[str, set[str]] = defaultdict(set)
        self._revision = 0

    def add(self, entry: MemoryEntry) -> None:
        if entry.memory_id in self._entries:
            raise ValueError(f"duplicate memory: {entry.memory_id}")
        self._entries[entry.memory_id] = entry
        self._scope_index[entry.scope.scope_id].add(entry.memory_id)
        self._revision += 1

    def replace(self, entry: MemoryEntry) -> None:
        previous = self._entries.get(entry.memory_id)
        if previous is not None:
            self._scope_index[previous.scope.scope_id].discard(entry.memory_id)
        self._entries[entry.memory_id] = entry
        self._scope_index[entry.scope.scope_id].add(entry.memory_id)
        self._revision += 1

    def get(self, memory_id: str) -> MemoryEntry | None:
        return self._entries.get(memory_id)

    def scopes(self) -> list:
        by_id = {entry.scope.scope_id: entry.scope for entry in self._entries.values()}
        return list(by_id.values())

    def entries_in_view(
        self,
        view: EffectiveView,
        *,
        states: frozenset[ArtifactState] | None = None,
        memory_types: frozenset[str] | None = None,
    ) -> list[MemoryEntry]:
        allowed_states = states or frozenset(
            {ArtifactState.VERIFIED, ArtifactState.COMMITTED}
        )
        ids: set[str] = set()
        for scope_id in view.scope_ids:
            ids.update(self._scope_index.get(scope_id, ()))
        entries: list[MemoryEntry] = []
        for memory_id in ids:
            entry = self._entries[memory_id]
            if entry.expired or entry.state not in allowed_states:
                continue
            if memory_types and not (entry.scope.memory_types & memory_types):
                continue
            entries.append(entry)
        return entries

    def search(
        self,
        vector: tuple[float, ...],
        view: EffectiveView,
        *,
        top_k: int,
        memory_types: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []
        hits = [
            SearchHit(entry, cosine_similarity(vector, entry.vector))
            for entry in self.entries_in_view(view, memory_types=memory_types)
            if entry.vector is not None
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.memory.memory_id))
        return hits[:top_k]

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def revision(self) -> int:
        return self._revision

    def __iter__(self) -> Iterable[MemoryEntry]:
        return iter(self._entries.values())
