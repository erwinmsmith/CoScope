"""
Base reranker interface for agent-specific ranking.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from core.types import (
        AgentRole,
        AgentState,
        RetrievedCandidate,
    )

logger = logging.getLogger(__name__)


@dataclass
class RerankedCandidate:
    """A candidate with reranking scores."""

    candidate: Any  # MemoryEntry
    original_score: float
    reranked_score: float
    rank: int
    features: Dict[str, float] = field(default_factory=dict)


class CandidateReranker(ABC):
    """
    Base class for candidate rerankers.

    Rerankers personalize shared candidates based on agent context.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Any],  # List[MemoryEntry]
        role: "AgentRole",
        state: "AgentState",
        top_k: int = 20,
    ) -> List[RerankedCandidate]:
        """
        Rerank candidates for a specific agent.

        Args:
            query: Original query string
            candidates: Candidates to rerank
            role: Agent role
            state: Agent state
            top_k: Number of candidates to return

        Returns:
            List of reranked candidates
        """
        ...

    def _create_default_weights(self, role: "AgentRole") -> Dict[str, float]:
        """Get default weights for a role."""
        defaults = {
            "planner": {"summary": 0.4, "constraint": 0.3, "recency": 0.2, "authority": 0.1},
            "solver": {"factual": 0.5, "summary": 0.3, "recency": 0.1, "authority": 0.1},
            "verifier": {"provenance": 0.4, "authority": 0.3, "recency": 0.2, "summary": 0.1},
            "critic": {"conflict": 0.4, "summary": 0.3, "recency": 0.2, "authority": 0.1},
        }
        return defaults.get(role.value, {"summary": 0.4, "factual": 0.3, "recency": 0.2, "authority": 0.1})
