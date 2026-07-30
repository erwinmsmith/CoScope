"""Write artifacts to their initial runtime scope."""

from __future__ import annotations

from coscope.adapters.embedding import EmbeddingAdapter
from coscope.core.artifact import Artifact
from coscope.core.memory import MemoryEntry
from coscope.memory.store import MemoryStore
from coscope.scope.descriptor import ScopeDescriptor


class CommitService:
    def __init__(self, store: MemoryStore, embedder: EmbeddingAdapter):
        self.store = store
        self.embedder = embedder

    def write(
        self,
        artifact: Artifact,
        scope: ScopeDescriptor,
        *,
        searchable: bool = True,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            content=artifact.content,
            scope=scope,
            source_type=artifact.source_type,
            source_id=artifact.artifact_id,
            state=artifact.state,
            vector=(
                self.embedder.embed(artifact.content) if searchable else None
            ),
            metadata={
                **artifact.metadata,
                "artifact_type": artifact.artifact_type.value,
                "searchable": searchable,
            },
        )
        self.store.add(entry)
        return entry
