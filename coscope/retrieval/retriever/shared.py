"""
Shared Candidate Retriever - Retrieves candidates from shared memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from coscope.core.scope import ScopeSpec
from coscope.core.types import EmbeddingProvider, MemoryEntry, MemoryStore
from coscope.retrieval.projection import ProjectionResult

from .base import CandidatePool, CandidateRetriever, RetrievalContext
from .similarity import CosineSimilarity, SimilarityStrategy

logger = logging.getLogger(__name__)


class SharedCandidateRetriever(CandidateRetriever):
    """
    Retrieves candidates from shared memory using projected representations.

    Workflow:
    1. Fetch candidate keys from memory store
    2. Project candidates to shared subspace
    3. Compute similarity matrix
    4. Select top-k candidates
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_provider: Optional[EmbeddingProvider] = None,
        similarity_strategy: Optional[SimilarityStrategy] = None,
        default_top_k: int = 50,
    ):
        super().__init__(memory_store, embedding_provider)
        self.similarity_strategy = similarity_strategy or CosineSimilarity()
        self.default_top_k = default_top_k

    def retrieve(
        self,
        projection_result: ProjectionResult,
        context: RetrievalContext,
        top_k: Optional[int] = None,
    ) -> CandidatePool:
        """Retrieve shared candidates."""
        k = top_k or self.default_top_k
        Z = projection_result.projected  # Shape: (n, r)

        if Z.size == 0:
            logger.warning("Empty projected matrix")
            return CandidatePool(
                pool_id=f"pool_{id(context)}",
                candidates=[],
                scores=[],
            )

        # Fetch candidates from memory
        entries, embeddings = self._fetch_candidates(context, k)

        if not entries:
            logger.info(f"No candidates found for scope {context.scope_id}")
            return CandidatePool(
                pool_id=f"pool_{id(context)}",
                candidates=[],
                scores=[],
            )

        # Project candidates
        W_final = projection_result.projection_matrix
        K = np.array(embeddings) @ W_final

        # Compute similarity
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            A = self.similarity_strategy.compute(Z, K)
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

        # Get top-k per query
        selected_scores = []
        selected_indices = np.argsort(-A, axis=1)[:, :k]

        for i, indices in enumerate(selected_indices):
            for idx, score in zip(indices, A[i, indices]):
                if entries[idx] not in [c for c, _ in zip(selected_scores, selected_scores)]:
                    selected_scores.append((entries[idx], float(score)))

        # Deduplicate
        seen = set()
        final_candidates = []
        final_scores = []
        for entry, score in selected_scores:
            if entry.memory_id not in seen:
                seen.add(entry.memory_id)
                final_candidates.append(entry)
                final_scores.append(score)

        return CandidatePool(
            pool_id=f"pool_{id(context)}",
            candidates=final_candidates[:k],
            scores=final_scores[:k],
            metadata={
                "num_queries": Z.shape[0],
                "num_total_candidates": len(entries),
                "num_selected": len(final_candidates),
            },
        )

    def _fetch_candidates(
        self,
        context: RetrievalContext,
        limit: int,
    ) -> tuple:
        """Fetch candidates from memory store."""
        entries = []
        embeddings = []

        for scope_id in context.scope_spec.all_scopes:
            for mem_type in context.memory_types:
                candidates = self.memory_store.search(
                    query_embedding=np.zeros(512),
                    scope_filter=[scope_id],
                    memory_type_filter=[mem_type],
                    policy_filter=context.policy,
                    top_k=limit * 2,
                )

                for candidate in candidates:
                    entries.append(candidate.memory)
                    if candidate.memory.embedding is not None:
                        embeddings.append(candidate.memory.embedding)
                    elif self.embedding_provider:
                        embeddings.append(
                            self.embedding_provider.embed_query(candidate.memory.content)
                        )
                    else:
                        embeddings.append(np.zeros(512))

        return entries, embeddings
