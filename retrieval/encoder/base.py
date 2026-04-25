"""
Base encoder interface for retrieval requests.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from core.types import (
        AgentConfig,
        AgentRole,
        AgentState,
        EmbeddingProvider,
        MemoryType,
        PolicyConstraints,
        RetrievalRequest,
        ScopeSpec,
    )

logger = logging.getLogger(__name__)


@dataclass
class EncodingResult:
    """Result of encoding a retrieval request."""

    request: "RetrievalRequest"
    normalized_query: str
    embedding: Optional[Any] = None
    warnings: List[str] = field(default_factory=list)


class RequestEncoder(ABC):
    """
    Base class for retrieval request encoders.

    Encoders normalize and validate agent requests before routing.
    """

    @abstractmethod
    def encode(
        self,
        raw_query: str,
        agent_config: "AgentConfig",
        state: "AgentState",
        scope: Optional["ScopeSpec"] = None,
        memory_types: Optional[List["MemoryType"]] = None,
        policy: Optional["PolicyConstraints"] = None,
        priority: int = 0,
        metadata: Optional[dict] = None,
        normalize: bool = True,
        embed: bool = False,
    ) -> EncodingResult:
        """
        Encode a raw request into a standardized RetrievalRequest.

        Args:
            raw_query: The original query text from the agent
            agent_config: Configuration for the requesting agent
            state: Current state of the agent
            scope: Optional scope specification
            memory_types: Optional memory types
            policy: Optional policy constraints
            priority: Request priority for scheduling
            metadata: Additional metadata
            normalize: Whether to normalize the query
            embed: Whether to generate embeddings

        Returns:
            EncodingResult containing the encoded request and metadata
        """
        ...

    @abstractmethod
    def encode_batch(
        self,
        requests: List[dict],
        normalize: bool = True,
        embed: bool = False,
    ) -> List[EncodingResult]:
        """
        Encode a batch of raw requests.

        Args:
            requests: List of raw request dicts
            normalize: Whether to normalize queries
            embed: Whether to generate embeddings

        Returns:
            List of EncodingResults
        """
        ...

    def validate_request(self, result: EncodingResult) -> List[str]:
        """
        Validate an encoded request and return warnings.

        Args:
            result: The encoding result to validate

        Returns:
            List of warning messages
        """
        warnings = []

        # Check query is not empty
        if not result.normalized_query.strip():
            warnings.append("Query is empty after normalization")

        # Check scope has accessible areas
        if not result.request.scope.all_scopes:
            warnings.append("Scope has no accessible scopes")

        return warnings
