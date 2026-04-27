"""
Memory CRUD (Create, Read, Update, Delete) Manager.

Provides a high-level interface for memory operations.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import numpy as np

from core.types import (
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

from .types import MemoryFilter, MemoryQuery

logger = logging.getLogger(__name__)


class MemoryCRUD:
    """
    High-level CRUD interface for memory operations.

    Example usage:
    ```python
    crud = MemoryCRUD(memory_store, embedding_provider)

    # Create
    memory = crud.create(
        content="Important fact",
        scope_id="task/shared",
        memory_type=MemoryType.SEMANTIC,
    )

    # Read
    memory = crud.get(memory_id)
    memories = crud.list(scope_id="task/shared")

    # Update
    crud.update(memory_id, content="Updated content")

    # Delete
    crud.delete(memory_id)

    # Query
    results = crud.query(MemoryQuery(text="Important", limit=10))
    ```
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

    def create(
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
        memory_id: Optional[str] = None,
    ) -> MemoryEntry:
        """
        Create a new memory entry.

        Args:
            content: Memory content text
            scope_id: Scope to store in
            memory_type: Type of memory
            visibility: Visibility levels
            confidence: Confidence score (0-1)
            tags: Optional tags
            ttl: Time-to-live in seconds
            provenance: Provenance info
            metadata: Additional metadata
            memory_id: Optional custom ID

        Returns:
            Created MemoryEntry
        """
        # Generate embedding if enabled
        embedding = None
        embedding_dim = 0
        if self.auto_embed and self.embedding_provider:
            try:
                embedding = self.embedding_provider.embed_query(content)
                embedding_dim = len(embedding)
                if embedding is not None and np.all(embedding == 0):
                    embedding = None
            except (ValueError, TypeError):
                pass

        entry = MemoryEntry(
            memory_id=memory_id or f"mem_{len(self.store._entries)}_{int(hashlib.md5(content.encode()).hexdigest()[:8], 16) % 100000:05d}",
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
        logger.debug(f"Created memory {entry.memory_id} in scope {scope_id}")

        return entry

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        return self.store.get(memory_id)

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        visibility: Optional[List[VisibilityLevel]] = None,
        confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[MemoryEntry]:
        """
        Update an existing memory entry.

        Args:
            memory_id: ID of memory to update
            content: New content (optional)
            visibility: New visibility (optional)
            confidence: New confidence (optional)
            tags: New tags (optional)
            metadata: New metadata (optional)

        Returns:
            Updated MemoryEntry or None if not found
        """
        entry = self.store.get(memory_id)
        if not entry:
            logger.warning(f"Memory not found: {memory_id}")
            return None

        # Update fields
        if content is not None:
            entry.content = content
            # Regenerate embedding
            if self.auto_embed and self.embedding_provider:
                try:
                    entry.embedding = self.embedding_provider.embed_query(content)
                    entry.embedding_dim = len(entry.embedding)
                except (ValueError, TypeError):
                    pass

        if visibility is not None:
            entry.visibility = visibility

        if confidence is not None:
            entry.confidence = confidence

        if tags is not None:
            entry.tags = tags

        if metadata is not None:
            entry.metadata.update(metadata)

        entry.updated_at = datetime.utcnow()

        # Re-add to store
        self.store.add(entry)
        logger.debug(f"Updated memory {memory_id}")

        return entry

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if deleted, False if not found
        """
        result = self.store.delete(memory_id)
        if result:
            logger.debug(f"Deleted memory {memory_id}")
        return result

    def list(
        self,
        scope_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
    ) -> List[MemoryEntry]:
        """
        List memory entries with optional filters.

        Args:
            scope_id: Filter by scope
            memory_type: Filter by type
            limit: Maximum results

        Returns:
            List of MemoryEntry
        """
        candidates = self.store.search(
            query_embedding=np.zeros(512),
            scope_filter=[scope_id] if scope_id else None,
            memory_type_filter=[memory_type] if memory_type else None,
            top_k=limit,
        )

        return [c.memory for c in candidates]

    def query(
        self,
        query: MemoryQuery,
    ) -> List[RetrievedCandidate]:
        """
        Query memories with filters and text search.

        Args:
            query: Query specification

        Returns:
            List of retrieved candidates
        """
        # Get embedding for text search
        embedding = np.zeros(512)
        if query.text and self.embedding_provider:
            try:
                embedding = self.embedding_provider.embed_query(query.text)
            except (ValueError, TypeError):
                pass

        # Search
        candidates = self.store.search(
            query_embedding=embedding,
            scope_filter=query.filter.scope_ids if query.filter else None,
            memory_type_filter=query.filter.memory_types if query.filter else None,
            top_k=query.limit + query.offset,
        )

        # Apply additional filters
        if query.filter:
            filtered = [c for c in candidates if query.filter.matches(c.memory)]
        else:
            filtered = candidates

        # Apply offset and limit
        filtered = filtered[query.offset : query.offset + query.limit]

        # Sort
        if query.sort_by == "created_at":
            filtered.sort(
                key=lambda c: c.memory.created_at,
                reverse=(query.sort_order == "desc"),
            )
        elif query.sort_by == "confidence":
            filtered.sort(
                key=lambda c: c.memory.confidence,
                reverse=(query.sort_order == "desc"),
            )

        return filtered

    def bulk_create(
        self,
        entries: List[Dict[str, Any]],
    ) -> List[MemoryEntry]:
        """
        Create multiple memory entries.

        Args:
            entries: List of memory entry dicts

        Returns:
            List of created MemoryEntry
        """
        results = []
        for entry_data in entries:
            entry = self.create(**entry_data)
            results.append(entry)
        return results

    def bulk_delete(
        self,
        memory_ids: List[str],
    ) -> int:
        """
        Delete multiple memory entries.

        Args:
            memory_ids: List of IDs to delete

        Returns:
            Number of entries deleted
        """
        count = 0
        for mid in memory_ids:
            if self.delete(mid):
                count += 1
        return count

    def cleanup_expired(self) -> int:
        """Remove expired memory entries."""
        expired = [
            mid
            for mid, entry in self.store._entries.items()
            if entry.is_expired()
        ]
        for mid in expired:
            self.store.delete(mid)
        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        entries = list(self.store._entries.values())

        if not entries:
            return {"total": 0}

        return {
            "total": len(entries),
            "by_scope": self._count_by(entries, "scope_id"),
            "by_type": self._count_by(entries, "memory_type"),
            "avg_confidence": sum(e.confidence for e in entries) / len(entries),
        }

    def _count_by(self, entries: List[MemoryEntry], attr: str) -> Dict[str, int]:
        """Count entries by an attribute."""
        counts: Dict[str, int] = {}
        for entry in entries:
            value = getattr(entry, attr)
            key = value.value if hasattr(value, "value") else str(value)
            counts[key] = counts.get(key, 0) + 1
        return counts
