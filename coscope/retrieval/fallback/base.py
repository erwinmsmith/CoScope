"""
Base fallback interface for private scope retrieval.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from coscope.core.types import RetrievedCandidate, RetrievalRequest

logger = logging.getLogger(__name__)


@dataclass
class FallbackResult:
    """Result of fallback retrieval."""

    triggered: bool
    candidates: List[Any] = field(default_factory=list)  # List[RetrievedCandidate]
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class FallbackTrigger(ABC):
    """
    Base class for fallback trigger strategies.

    Fallback retrieves from private scopes when shared candidates are insufficient.
    """

    @abstractmethod
    def should_trigger(
        self,
        shared_candidates: List["RetrievedCandidate"],
        request: "RetrievalRequest",
        threshold: int,
    ) -> bool:
        """
        Determine if private fallback should be triggered.

        Args:
            shared_candidates: Candidates from shared retrieval
            request: Original retrieval request
            threshold: Configured threshold

        Returns:
            True if fallback should be triggered
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Trigger", "").lower()


class FallbackRetriever(ABC):
    """
    Base class for fallback retrievers.

    Fallback retrievers search private scopes for additional candidates.
    """

    def __init__(
        self,
        memory_store: Any,
        embedding_provider: Optional[Any] = None,
        default_threshold: int = 5,
        max_candidates: int = 20,
    ):
        self.memory_store = memory_store
        self.embedding_provider = embedding_provider
        self.default_threshold = default_threshold
        self.max_candidates = max_candidates

    @abstractmethod
    def retrieve(
        self,
        request: "RetrievalRequest",
        shared_candidates: List["RetrievedCandidate"],
        trigger: FallbackTrigger,
        threshold: Optional[int] = None,
    ) -> FallbackResult:
        """
        Perform fallback retrieval if triggered.

        Args:
            request: Original retrieval request
            shared_candidates: Candidates from shared retrieval
            trigger: Fallback trigger strategy
            threshold: Override threshold

        Returns:
            FallbackResult with private candidates
        """
        ...
