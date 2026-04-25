"""
Fallback trigger and retrieval strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from core.types import RetrievedCandidate, RetrievalRequest

from .base import FallbackResult, FallbackRetriever, FallbackTrigger

logger = logging.getLogger(__name__)


class CountBasedTrigger(FallbackTrigger):
    """Trigger fallback based on count of shared candidates."""

    def __init__(self, min_count: int = 5):
        self.min_count = min_count

    def should_trigger(
        self,
        shared_candidates: List["RetrievedCandidate"],
        request: "RetrievalRequest",
        threshold: int,
    ) -> bool:
        return len(shared_candidates) < (threshold or self.min_count)


class ScoreBasedTrigger(FallbackTrigger):
    """Trigger fallback based on quality of shared candidates."""

    def __init__(self, min_score: float = 0.5, top_k: int = 5):
        self.min_score = min_score
        self.top_k = top_k

    def should_trigger(
        self,
        shared_candidates: List["RetrievedCandidate"],
        request: "RetrievalRequest",
        threshold: int,
    ) -> bool:
        if not shared_candidates:
            return True

        top_k_candidates = shared_candidates[: self.top_k]
        avg_score = sum(c.score for c in top_k_candidates) / len(top_k_candidates)
        return avg_score < self.min_score


class CombinedTrigger(FallbackTrigger):
    """Trigger fallback based on both count and score."""

    def __init__(
        self,
        min_count: int = 5,
        min_score: float = 0.3,
        top_k: int = 5,
    ):
        self.count_trigger = CountBasedTrigger(min_count)
        self.score_trigger = ScoreBasedTrigger(min_score, top_k)

    def should_trigger(
        self,
        shared_candidates: List["RetrievedCandidate"],
        request: "RetrievalRequest",
        threshold: int,
    ) -> bool:
        return (
            self.count_trigger.should_trigger(shared_candidates, request, threshold)
            or self.score_trigger.should_trigger(shared_candidates, request, threshold)
        )


class PrivateFallbackRetriever(FallbackRetriever):
    """
    Retrieves additional candidates from private scopes.
    """

    def retrieve(
        self,
        request: "RetrievalRequest",
        shared_candidates: List["RetrievedCandidate"],
        trigger: FallbackTrigger,
        threshold: Optional[int] = None,
    ) -> FallbackResult:
        """Perform private fallback retrieval."""
        effective_threshold = threshold or self.default_threshold

        if not trigger.should_trigger(shared_candidates, request, effective_threshold):
            return FallbackResult(
                triggered=False,
                reason="Shared candidates sufficient",
            )

        logger.info(
            f"Fallback triggered for request {request.request_id}: "
            f"{len(shared_candidates)} < {effective_threshold}"
        )

        # Fetch from private scopes
        candidates = self._fetch_private_candidates(request)

        return FallbackResult(
            triggered=True,
            candidates=candidates,
            reason=f"Below threshold: {len(shared_candidates)} < {effective_threshold}",
            metadata={
                "private_scopes_accessed": request.scope.private_scopes,
                "trigger_type": trigger.name,
            },
        )

    def _fetch_private_candidates(
        self,
        request: "RetrievalRequest",
    ) -> List["RetrievedCandidate"]:
        """Fetch candidates from private scopes."""
        all_candidates = []
        query_embedding = (
            self.embedding_provider.embed_query(request.query)
            if self.embedding_provider
            else np.zeros(1, dtype="float32")
        )

        for scope_id in request.scope.private_scopes:
            for mem_type in request.memory_types:
                candidates = self.memory_store.search(
                    query_embedding=query_embedding,
                    scope_filter=[scope_id],
                    memory_type_filter=[mem_type],
                    policy_filter=request.policy,
                    top_k=self.max_candidates,
                )
                all_candidates.extend(candidates)

        # Deduplicate
        seen = set()
        unique = []
        for candidate in all_candidates:
            if candidate.memory.memory_id not in seen:
                seen.add(candidate.memory.memory_id)
                candidate.source = "private"
                unique.append(candidate)

        return unique[: self.max_candidates]
