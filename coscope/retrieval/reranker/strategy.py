"""
Reranking strategies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from coscope.core.types import AgentRole, AgentState

from .base import CandidateReranker, RerankedCandidate
from .feature import (
    AuthorityExtractor,
    FeatureExtractor,
    KeywordOverlapExtractor,
    ProvenanceExtractor,
    RecencyExtractor,
    SummaryScoreExtractor,
)

logger = logging.getLogger(__name__)


class WeightedReranker(CandidateReranker):
    """
    Weighted sum reranking based on feature extractors.
    """

    def __init__(
        self,
        extractors: List[FeatureExtractor],
        weights: Dict[str, float],
    ):
        self.extractors = extractors
        self.weights = weights

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        role: AgentRole,
        state: AgentState,
        top_k: int = 20,
    ) -> List[RerankedCandidate]:
        """Rerank using weighted feature sum."""
        if not candidates:
            return []

        scored = []
        for candidate in candidates:
            features = {}
            total_score = 0.0

            for extractor in self.extractors:
                score = extractor.extract(query, candidate, role, state)
                name = extractor.name
                features[name] = score

                weight = self.weights.get(name, 0.0)
                total_score += weight * score

            scored.append(
                RerankedCandidate(
                    candidate=candidate,
                    original_score=getattr(candidate, "score", 0.0),
                    reranked_score=total_score,
                    rank=0,
                    features=features,
                )
            )

        # Sort by reranked score
        scored.sort(key=lambda x: x.reranked_score, reverse=True)

        # Assign ranks
        for rank, item in enumerate(scored[:top_k], 1):
            item.rank = rank

        return scored[:top_k]


class RoleAwareReranker(CandidateReranker):
    """
    Role-aware reranker with predefined extractors and weights.
    """

    ROLE_CONFIGS = {
        AgentRole.PLANNER: {
            "extractors": [
                SummaryScoreExtractor(),
                KeywordOverlapExtractor(),
                RecencyExtractor(),
                AuthorityExtractor(),
            ],
            "weights": {
                "summary": 0.4,
                "keyword": 0.3,
                "recency": 0.2,
                "authority": 0.1,
            },
        },
        AgentRole.SOLVER: {
            "extractors": [
                SummaryScoreExtractor(),
                KeywordOverlapExtractor(),
                AuthorityExtractor(),
            ],
            "weights": {
                "summary": 0.5,
                "keyword": 0.3,
                "authority": 0.2,
            },
        },
        AgentRole.VERIFIER: {
            "extractors": [
                ProvenanceExtractor(),
                AuthorityExtractor(),
                RecencyExtractor(),
            ],
            "weights": {
                "provenance": 0.4,
                "authority": 0.3,
                "recency": 0.3,
            },
        },
    }

    def __init__(self):
        self.default_config = {
            "extractors": [
                SummaryScoreExtractor(),
                KeywordOverlapExtractor(),
                RecencyExtractor(),
            ],
            "weights": {
                "summary": 0.5,
                "keyword": 0.3,
                "recency": 0.2,
            },
        }

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        role: AgentRole,
        state: AgentState,
        top_k: int = 20,
    ) -> List[RerankedCandidate]:
        """Rerank using role-aware configuration."""
        config = self.ROLE_CONFIGS.get(role, self.default_config)

        reranker = WeightedReranker(
            extractors=config["extractors"],
            weights=config["weights"],
        )

        return reranker.rerank(query, candidates, role, state, top_k)
