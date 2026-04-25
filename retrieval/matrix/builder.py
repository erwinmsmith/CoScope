"""
Query Matrix Builder.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from core.scope import ScopeBucketKey
from core.types import EmbeddingProvider, RetrievalRequest

from .types import QueryMatrix

logger = logging.getLogger(__name__)


class EncodingStrategy:
    """Base class for encoding strategies."""

    def encode(
        self,
        queries: List[str],
        provider: EmbeddingProvider,
    ) -> np.ndarray:
        """Encode queries into a matrix."""
        raise NotImplementedError


class BatchEncodingStrategy(EncodingStrategy):
    """Batch encoding that processes all queries at once."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def encode(
        self,
        queries: List[str],
        provider: EmbeddingProvider,
    ) -> np.ndarray:
        """Encode queries in batch."""
        embeddings = provider.embed_texts(queries)
        Q = np.column_stack(embeddings)

        if self.normalize:
            norms = np.linalg.norm(Q, axis=0, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            Q = Q / norms

        return Q


class QueryMatrixBuilder:
    """
    Builds QueryMatrix objects from RetrievalBuckets.

    This module constructs Q ∈ R^(k×n) matrices per bucket where:
    - k = embedding dimension
    - n = number of queries in bucket
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        encoding_strategy: Optional[EncodingStrategy] = None,
        query_embedding_dim: Optional[int] = None,
    ):
        self.embedding_provider = embedding_provider
        self.encoding_strategy = encoding_strategy or BatchEncodingStrategy()
        self.query_embedding_dim = query_embedding_dim

    def build(
        self,
        requests: List[RetrievalRequest],
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> QueryMatrix:
        """
        Build a query matrix from a list of requests.

        Args:
            requests: List of retrieval requests
            embedding_provider: Optional provider override

        Returns:
            QueryMatrix for the requests
        """
        if not requests:
            return QueryMatrix()

        effective_provider = embedding_provider or self.embedding_provider

        if effective_provider is None:
            raise ValueError("Embedding provider required for encoding")

        request_ids = [r.request_id for r in requests]
        agent_ids = [r.agent_id for r in requests]
        queries = [r.query for r in requests]

        # Encode queries
        try:
            embeddings = self.encoding_strategy.encode(queries, effective_provider)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise

        # Determine scope bucket key
        primary_scope = requests[0].scope.primary_scope
        key = ScopeBucketKey(
            primary_scope=primary_scope,
            memory_types=tuple(sorted(t.value for t in requests[0].memory_types)),
            policy_hash=hash(str(requests[0].policy)),
        )

        return QueryMatrix(
            request_ids=request_ids,
            agent_ids=agent_ids,
            queries=queries,
            embeddings=embeddings,
            scope_bucket_key=key,
        )

    def build_batch(
        self,
        batch_requests: List[List[RetrievalRequest]],
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> List[QueryMatrix]:
        """Build query matrices for multiple batches."""
        return [self.build(reqs, embedding_provider) for reqs in batch_requests]
