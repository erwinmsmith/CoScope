"""
Base retriever interface for candidate retrieval.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from core.types import (
        EmbeddingProvider,
        MemoryStore,
    )

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Context for retrieval operations."""

    scope_id: str
    memory_types: List[Any]  # List[MemoryType]
    policy: Any  # PolicyConstraints
    scope_spec: Any  # ScopeSpec


@dataclass
class CandidatePool:
    """A pool of retrieved candidates."""

    pool_id: str
    candidates: List[Any]  # List[MemoryEntry]
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.candidates)


class CandidateRetriever(ABC):
    """
    Base class for candidate retrievers.

    Retrievers fetch relevant candidates from memory stores
    using query representations.
    """

    def __init__(
        self,
        memory_store: "MemoryStore",
        embedding_provider: Optional["EmbeddingProvider"] = None,
    ):
        self.memory_store = memory_store
        self.embedding_provider = embedding_provider

    @abstractmethod
    def retrieve(
        self,
        query_representation: Any,
        context: RetrievalContext,
        top_k: int = 50,
    ) -> CandidatePool:
        """
        Retrieve candidates using a query representation.

        Args:
            query_representation: Query vector or another retriever-specific representation
            context: Retrieval context with scope and policy info
            top_k: Number of candidates to retrieve

        Returns:
            CandidatePool with retrieved candidates
        """
        ...

    def batch_retrieve(
        self,
        query_representations: List[Any],
        contexts: List[RetrievalContext],
        top_k: int = 50,
    ) -> List[CandidatePool]:
        """Retrieve for multiple queries."""
        if len(query_representations) != len(contexts):
            raise ValueError("Mismatched query representations and contexts")
        return [
            self.retrieve(pr, ctx, top_k)
            for pr, ctx in zip(query_representations, contexts)
        ]
