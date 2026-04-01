"""
Base fusion interface for merging candidates.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from coscope.core.types import AgentRole, RetrievedCandidate

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Result of evidence fusion."""

    candidates: List[Any]  # List[RetrievedCandidate]
    shared_count: int
    private_count: int
    both_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvidenceFusion(ABC):
    """
    Base class for evidence fusion strategies.

    Fusion merges shared and private candidates into final results.
    """

    @abstractmethod
    def fuse(
        self,
        shared_candidates: List["RetrievedCandidate"],
        private_candidates: List["RetrievedCandidate"],
        top_k: int = 20,
    ) -> FusionResult:
        """
        Fuse shared and private candidates.

        Args:
            shared_candidates: Candidates from shared retrieval
            private_candidates: Candidates from private fallback
            top_k: Number of candidates to return

        Returns:
            FusionResult with merged candidates
        """
        ...

    def _count_sources(
        self, candidates: List["RetrievedCandidate"]
    ) -> tuple:
        """Count candidates by source."""
        shared = sum(1 for c in candidates if c.source in ("shared", "both"))
        private = sum(1 for c in candidates if c.source in ("private", "both"))
        both = sum(1 for c in candidates if c.source == "both")
        return shared, private, both
