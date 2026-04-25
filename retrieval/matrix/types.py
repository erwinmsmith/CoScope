"""
Query Matrix type definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

if TYPE_CHECKING:
    from core.types import ScopeBucketKey

# Re-export QueryMatrix from core types
from core.types import QueryMatrix as CoreQueryMatrix


@dataclass
class QueryMatrix:
    """
    Represents a query matrix Q for a shared retrieval bucket.

    Q ∈ R^(k × n) where:
    - k = query embedding dimension
    - n = number of queries in the bucket
    - Column i = query vector for request i
    """

    request_ids: List[str] = field(default_factory=list)
    agent_ids: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=np.array)
    scope_bucket_key: Optional[Any] = None  # ScopeBucketKey

    @property
    def num_queries(self) -> int:
        return len(self.request_ids)

    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[0] if self.embeddings.size else 0

    @property
    def shape(self) -> tuple:
        return self.embeddings.shape

    def get_query_vector(self, index: int) -> np.ndarray:
        """Get the query vector at the given index."""
        if 0 <= index < self.num_queries:
            return self.embeddings[:, index]
        raise IndexError(f"Index {index} out of range for {self.num_queries} queries")

    def transpose(self) -> np.ndarray:
        """Get Q^T for projection."""
        return self.embeddings.T

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "num_queries": self.num_queries,
            "embedding_dim": self.embedding_dim,
            "request_ids": self.request_ids,
            "scope_bucket_key": str(self.scope_bucket_key) if self.scope_bucket_key else None,
        }
