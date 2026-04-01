"""
Fusion strategies for merging candidates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from coscope.core.types import RetrievedCandidate

from .base import EvidenceFusion, FusionResult

logger = logging.getLogger(__name__)


class ReciprocalRankFusion(EvidenceFusion):
    """
    Reciprocal Rank Fusion (RRF).

    RRF score = 1 / (k + rank)
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        shared_candidates: List["RetrievedCandidate"],
        private_candidates: List["RetrievedCandidate"],
        top_k: int = 20,
    ) -> FusionResult:
        """Fuse using reciprocal rank."""
        rrf_scores: Dict[str, tuple] = {}

        # Shared contribution
        for rank, candidate in enumerate(shared_candidates):
            memory_id = candidate.memory.memory_id
            rrf = 1.0 / (self.k + rank + 1)
            if memory_id in rrf_scores:
                rrf_scores[memory_id] = (
                    rrf_scores[memory_id][0] + rrf,
                    min(rrf_scores[memory_id][1], rank),
                )
            else:
                rrf_scores[memory_id] = (rrf, rank)

        # Private contribution
        for rank, candidate in enumerate(private_candidates):
            memory_id = candidate.memory.memory_id
            rrf = 1.0 / (self.k + rank + 1)
            if memory_id in rrf_scores:
                rrf_scores[memory_id] = (
                    rrf_scores[memory_id][0] + rrf,
                    min(rrf_scores[memory_id][1], rank),
                )
            else:
                rrf_scores[memory_id] = (rrf, rank)

        # Build candidate map
        candidate_map: Dict[str, RetrievedCandidate] = {}
        for candidate in shared_candidates:
            candidate_map[candidate.memory.memory_id] = RetrievedCandidate(
                memory=candidate.memory,
                score=0.0,
                source="shared",
                agent_id=candidate.agent_id,
                relevance_labels=candidate.relevance_labels,
            )

        for candidate in private_candidates:
            memory_id = candidate.memory.memory_id
            if memory_id in candidate_map:
                candidate_map[memory_id].source = "both"
            else:
                candidate_map[memory_id] = RetrievedCandidate(
                    memory=candidate.memory,
                    score=0.0,
                    source="private",
                    agent_id=candidate.agent_id,
                )

        # Set RRF scores and sort
        fused = []
        for memory_id, (rrf_score, _) in rrf_scores.items():
            candidate = candidate_map[memory_id]
            candidate.score = rrf_score
            fused.append(candidate)

        fused.sort(key=lambda c: c.score, reverse=True)

        # Assign ranks
        for rank, candidate in enumerate(fused[:top_k], 1):
            candidate.rank = rank

        shared, private, both = self._count_sources(fused[:top_k])

        return FusionResult(
            candidates=fused[:top_k],
            shared_count=shared,
            private_count=private,
            both_count=both,
            metadata={"strategy": "rrf", "k": self.k},
        )


class ScoreWeightedFusion(EvidenceFusion):
    """Weighted fusion based on candidate scores."""

    def __init__(self, private_weight: float = 0.7):
        self.private_weight = private_weight

    def fuse(
        self,
        shared_candidates: List["RetrievedCandidate"],
        private_candidates: List["RetrievedCandidate"],
        top_k: int = 20,
    ) -> FusionResult:
        """Fuse using score weighting."""
        fused = []
        seen = set()

        # Add shared first
        for candidate in shared_candidates:
            candidate.source = "shared"
            fused.append(candidate)
            seen.add(candidate.memory.memory_id)

        # Add private with adjusted scores
        for candidate in private_candidates:
            if candidate.memory.memory_id not in seen:
                candidate.source = "private"
                candidate.score *= self.private_weight
                fused.append(candidate)
                seen.add(candidate.memory.memory_id)

        # Sort by score
        fused.sort(key=lambda c: c.score, reverse=True)

        # Assign ranks
        for rank, candidate in enumerate(fused[:top_k], 1):
            candidate.rank = rank

        shared, private, both = self._count_sources(fused[:top_k])

        return FusionResult(
            candidates=fused[:top_k],
            shared_count=shared,
            private_count=private,
            both_count=both,
            metadata={"strategy": "score_weighted", "private_weight": self.private_weight},
        )
