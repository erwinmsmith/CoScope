"""
Base scope router interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from coscope.core.scope import ScopeRegistry
    from coscope.core.types import RetrievalRequest, ScopeBucketKey

logger = logging.getLogger(__name__)


@dataclass
class RetrievalBucket:
    """
    A bucket of requests that can share first-stage retrieval.

    All requests in a bucket:
    - Access the same primary shared scope
    - Request compatible memory types
    - Have compatible policies
    """

    bucket_id: str
    scope_bucket_key: "ScopeBucketKey"
    requests: List["RetrievalRequest"] = field(default_factory=list)
    shared_scopes: List[str] = field(default_factory=list)
    memory_types: List[Any] = field(default_factory=list)  # List[MemoryType]
    merged_policy: Any = field(default_factory=list)  # PolicyConstraints
    is_shareable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_requests(self) -> int:
        return len(self.requests)

    @property
    def primary_scope(self) -> str:
        return self.scope_bucket_key.primary_scope


@dataclass
class RoutingResult:
    """
    Result of routing a batch of requests.
    """

    shareable_buckets: List[RetrievalBucket] = field(default_factory=list)
    independent_requests: List["RetrievalRequest"] = field(default_factory=list)
    routing_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_buckets(self) -> int:
        return len(self.shareable_buckets)

    @property
    def total_independent(self) -> int:
        return len(self.independent_requests)

    @property
    def total_requests(self) -> int:
        return sum(b.num_requests for b in self.shareable_buckets) + self.total_independent


class ScopeRouter(ABC):
    """
    Base class for scope-based request routers.

    Routers group retrieval requests by scope overlap and policy compatibility
    to enable collaborative first-stage retrieval.
    """

    def __init__(
        self,
        scope_registry: Optional["ScopeRegistry"] = None,
    ):
        from coscope.core.scope import ScopeRegistry

        self.scope_registry = scope_registry or ScopeRegistry()

    @abstractmethod
    def route(
        self,
        requests: List["RetrievalRequest"],
    ) -> RoutingResult:
        """
        Route requests to appropriate buckets.

        Args:
            requests: List of retrieval requests to route

        Returns:
            RoutingResult with shareable buckets and independent requests
        """
        ...

    def route_single(
        self,
        request: "RetrievalRequest",
    ) -> RoutingResult:
        """
        Route a single request.

        Args:
            request: The request to route

        Returns:
            RoutingResult with the request classified
        """
        return self.route([request])

    def get_bucket_stats(self, result: RoutingResult) -> Dict[str, Any]:
        """
        Get statistics about routing result.

        Args:
            result: The routing result

        Returns:
            Dict of statistics
        """
        bucket_sizes = [b.num_requests for b in result.shareable_buckets]

        return {
            "total_requests": result.total_requests,
            "total_buckets": result.total_buckets,
            "total_independent": result.total_independent,
            "avg_bucket_size": sum(bucket_sizes) / len(bucket_sizes) if bucket_sizes else 0,
            "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
            "min_bucket_size": min(bucket_sizes) if bucket_sizes else 0,
            "shareable_ratio": (
                result.total_requests - result.total_independent
            ) / result.total_requests if result.total_requests > 0 else 0,
        }
