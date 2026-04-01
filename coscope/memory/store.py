"""
Memory Storage for CoScope.

Provides memory storage backends and the memory management system.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np

from coscope.core.scope import PolicyConstraints, ScopeSpec
from coscope.core.types import (
    EmbeddingProvider,
    MemoryEntry,
    MemoryStore,
    MemoryType,
    PolicyConstraints,
    Provenance,
    RetrievedCandidate,
    ScopeSpec,
    VisibilityLevel,
)

if TYPE_CHECKING:
    from coscope.config.settings import CoScopeConfig

logger = logging.getLogger(__name__)


# ============================================================
# In-Memory Store Implementation
# ============================================================


class InMemoryMemoryStore:
    """
    Simple in-memory implementation of MemoryStore.

    Suitable for testing, development, or small-scale deployments.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        default_clearance: int = 3,
    ):
        self._entries: Dict[str, MemoryEntry] = {}
        self._scope_index: Dict[str, Set[str]] = {}  # scope_id -> memory_ids
        self._type_index: Dict[MemoryType, Set[str]] = {}  # memory_type -> memory_ids
        self._embedding_provider = embedding_provider
        self.default_clearance = default_clearance

    def add(self, memory: MemoryEntry) -> None:
        """Add a memory entry to the store."""
        self._entries[memory.memory_id] = memory

        # Update scope index
        if memory.scope_id not in self._scope_index:
            self._scope_index[memory.scope_id] = set()
        self._scope_index[memory.scope_id].add(memory.memory_id)

        # Update type index
        if memory.memory_type not in self._type_index:
            self._type_index[memory.memory_type] = set()
        self._type_index[memory.memory_type].add(memory.memory_id)

        logger.debug(f"Added memory {memory.memory_id} to scope {memory.scope_id}")

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        return self._entries.get(memory_id)

    def search(
        self,
        query_embedding: np.ndarray,
        scope_filter: Optional[List[str]] = None,
        memory_type_filter: Optional[List[MemoryType]] = None,
        policy_filter: Optional[PolicyConstraints] = None,
        top_k: int = 50,
    ) -> List[RetrievedCandidate]:
        """
        Search for relevant memories.

        Note: This is a simplified implementation that filters by scope/type
        and uses embedding similarity. A real implementation would use a
        proper vector index for efficiency.
        """
        # Determine candidate IDs
        candidate_ids: Set[str] = set()

        if scope_filter:
            for scope_id in scope_filter:
                if scope_id in self._scope_index:
                    candidate_ids.update(self._scope_index[scope_id])
        else:
            candidate_ids.update(self._entries.keys())

        if memory_type_filter:
            type_ids: Set[str] = set()
            for mem_type in memory_type_filter:
                if mem_type in self._type_index:
                    type_ids.update(self._type_index[mem_type])
            candidate_ids &= type_ids

        # Get candidates
        candidates = []
        for memory_id in candidate_ids:
            memory = self._entries.get(memory_id)
            if not memory:
                continue

            # Check policy
            if policy_filter and not self._check_policy(memory, policy_filter):
                continue

            # Compute similarity
            if memory.embedding is not None and query_embedding is not None:
                similarity = self._cosine_similarity(query_embedding, memory.embedding)
            else:
                similarity = 0.0

            candidates.append(
                RetrievedCandidate(
                    memory=memory,
                    score=float(similarity),
                    source="search",
                )
            )

        # Sort by score
        candidates.sort(key=lambda c: c.score, reverse=True)

        return candidates[:top_k]

    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        if memory_id not in self._entries:
            return False

        memory = self._entries[memory_id]

        # Remove from indices
        if memory.scope_id in self._scope_index:
            self._scope_index[memory.scope_id].discard(memory_id)

        if memory.memory_type in self._type_index:
            self._type_index[memory.memory_type].discard(memory_id)

        del self._entries[memory_id]
        return True

    def list_scopes(self) -> List[str]:
        """List all available scope IDs."""
        return list(self._scope_index.keys())

    def _check_policy(
        self, memory: MemoryEntry, policy: PolicyConstraints
    ) -> bool:
        """Check if memory satisfies policy constraints."""
        # Check visibility
        if policy.visibility:
            if not any(v in [m.value for m in memory.visibility] for v in policy.visibility):
                return False

        # Check excluded zones (simplified)
        # In real implementation, would check quarantine status

        return True

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the store."""
        return {
            "total_entries": len(self._entries),
            "scopes": len(self._scope_index),
            "memory_types": len(self._type_index),
        }

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._scope_index.clear()
        self._type_index.clear()


# ============================================================
# Memory Manager
# ============================================================


class MemoryManager:
    """
    High-level memory management interface.

    Provides a simplified API for memory operations with:
    - Automatic embedding generation
    - Scope management
    - Policy enforcement
    - TTL management
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_provider: Optional[EmbeddingProvider] = None,
        auto_embed: bool = True,
    ):
        self.store = memory_store
        self.embedding_provider = embedding_provider
        self.auto_embed = auto_embed

    def store_memory(
        self,
        content: str,
        scope_id: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        visibility: Optional[List[VisibilityLevel]] = None,
        confidence: float = 0.5,
        tags: Optional[List[str]] = None,
        ttl: Optional[int] = None,
        provenance: Optional[Provenance] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """
        Store a new memory entry.

        Args:
            content: The memory content
            scope_id: The scope to store in
            memory_type: Type of memory
            visibility: Visibility levels
            confidence: Confidence score
            tags: Optional tags
            ttl: Time-to-live in seconds
            provenance: Provenance information
            metadata: Additional metadata

        Returns:
            The created MemoryEntry
        """
        # Generate embedding if provider is available and auto_embed is True
        embedding = None
        embedding_dim = 0
        if self.auto_embed and self.embedding_provider:
            try:
                embedding = self.embedding_provider.embed_query(content)
                embedding_dim = len(embedding)
                # Skip storing zero embeddings (fallback case)
                if embedding is not None and np.all(embedding == 0):
                    embedding = None
            except (ValueError, TypeError):
                # Embedding generation failed, skip
                pass

        entry = MemoryEntry(
            memory_id=f"mem_{len(self.store._entries)}_{hash(content) % 100000:05d}",
            scope_id=scope_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            embedding_dim=embedding_dim,
            visibility=visibility or [VisibilityLevel.TEAM],
            confidence=confidence,
            salience=metadata.get("salience", 0.5) if metadata else 0.5,
            ttl=ttl,
            provenance=provenance or Provenance(),
            tags=tags or [],
            metadata=metadata or {},
        )

        self.store.add(entry)
        return entry

    def retrieve(
        self,
        query: str,
        scope_filter: Optional[List[str]] = None,
        memory_type_filter: Optional[List[MemoryType]] = None,
        top_k: int = 20,
    ) -> List[RetrievedCandidate]:
        """
        Retrieve memories based on a query.

        Args:
            query: Query text
            scope_filter: Scopes to search
            memory_type_filter: Memory types to search
            top_k: Number of results

        Returns:
            List of retrieved candidates
        """
        # Generate query embedding
        query_embedding = np.zeros(512)
        if self.embedding_provider:
            query_embedding = self.embedding_provider.embed_query(query)

        return self.store.search(
            query_embedding=query_embedding,
            scope_filter=scope_filter,
            memory_type_filter=memory_type_filter,
            top_k=top_k,
        )

    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        return self.store.delete(memory_id)

    def cleanup_expired(self) -> int:
        """
        Remove expired memory entries.

        Returns:
            Number of entries removed
        """
        expired_ids = [
            mid for mid, entry in self.store._entries.items()
            if entry.is_expired()
        ]

        for mid in expired_ids:
            self.store.delete(mid)

        return len(expired_ids)

    def get_by_scope(self, scope_id: str) -> List[MemoryEntry]:
        """Get all memories in a scope."""
        candidates = self.store.search(
            query_embedding=np.zeros(512),
            scope_filter=[scope_id],
            top_k=1000,
        )
        return [c.memory for c in candidates]


# ============================================================
# Factory Functions
# ============================================================


def create_memory_store(
    backend: Literal["inmemory"] = "inmemory",
    embedding_provider: Optional[EmbeddingProvider] = None,
    **kwargs,
) -> MemoryStore:
    """
    Factory function to create a memory store.

    Currently supports:
    - "inmemory": In-memory store
    - "chromadb": ChromaDB (requires chromadb package)
    - "faiss": FAISS (requires faiss package)
    """

    if backend == "inmemory":
        return InMemoryMemoryStore(
            embedding_provider=embedding_provider,
            default_clearance=kwargs.get("default_clearance", 3),
        )

    elif backend == "chromadb":
        try:
            import chromadb
            from coscope.memory.chromadb_store import ChromaDBMemoryStore

            client = chromadb.PersistentClient(
                path=kwargs.get("persist_dir", "./coscope_data/chromadb")
            )
            return ChromaDBMemoryStore(
                client=client,
                embedding_provider=embedding_provider,
                collection_name=kwargs.get("collection_name", "coscope_memories"),
            )
        except ImportError:
            logger.warning("ChromaDB not installed, falling back to in-memory store")
            return InMemoryMemoryStore(embedding_provider=embedding_provider)

    elif backend == "faiss":
        try:
            import faiss
            from coscope.memory.faiss_store import FAISSMemoryStore

            return FAISSMemoryStore(
                embedding_provider=embedding_provider,
                dimension=kwargs.get("dimension", 1024),
                index_type=kwargs.get("index_type", "IDMap2,SQ8"),
            )
        except ImportError:
            logger.warning("FAISS not installed, falling back to in-memory store")
            return InMemoryMemoryStore(embedding_provider=embedding_provider)

    else:
        logger.warning(f"Unknown backend '{backend}', using in-memory store")
        return InMemoryMemoryStore(embedding_provider=embedding_provider)
