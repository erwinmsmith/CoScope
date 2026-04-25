"""
Overlap-aware routing strategy using scope overlap detection.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Set

from core.scope import (
    PolicyCompatibilityChecker,
    ScopeOverlapDetector,
    ScopeRegistry,
    ScopeSpec,
)
from core.types import MemoryType, RetrievalRequest

from .base import RetrievalBucket, RoutingResult, ScopeRouter

logger = logging.getLogger(__name__)


class OverlapAwareRouter(ScopeRouter):
    """
    Advanced routing strategy that considers full scope overlap patterns.

    This strategy:
    1. Computes pairwise overlap scores
    2. Groups requests by overlap connectivity
    3. Validates policy compatibility within groups
    """

    def __init__(
        self,
        scope_registry: Any = None,
        min_overlap_score: float = 0.5,
    ):
        super().__init__(scope_registry)
        self.overlap_detector = ScopeOverlapDetector()
        self.policy_checker = PolicyCompatibilityChecker()
        self.min_overlap_score = min_overlap_score

    def route(
        self,
        requests: List["RetrievalRequest"],
    ) -> RoutingResult:
        """Route requests using overlap-aware strategy."""
        if not requests:
            return RoutingResult()

        if len(requests) == 1:
            return self._create_trivial_result(requests)

        # Build overlap graph
        overlap_graph = self._build_overlap_graph(requests)

        # Find connected components
        groups = self._find_connected_components(overlap_graph, len(requests))

        # Create buckets from groups
        shareable_buckets: List[RetrievalBucket] = []
        independent_requests: List[RetrievalRequest] = []

        for i, group_indices in enumerate(groups):
            group_requests = [requests[idx] for idx in group_indices]

            if len(group_requests) == 1:
                independent_requests.extend(group_requests)
            else:
                bucket = self._create_bucket_from_requests(group_requests, f"overlap_bucket_{i}")
                if bucket:
                    shareable_buckets.append(bucket)

        logger.info(
            f"Overlap-aware routing: {len(shareable_buckets)} buckets, "
            f"{len(independent_requests)} independent"
        )

        return RoutingResult(
            shareable_buckets=shareable_buckets,
            independent_requests=independent_requests,
            routing_metadata={
                "strategy": "overlap_aware",
                "min_overlap_score": self.min_overlap_score,
                "total_buckets": len(shareable_buckets),
                "total_independent": len(independent_requests),
            },
        )

    def _build_overlap_graph(
        self,
        requests: List["RetrievalRequest"],
    ) -> Dict[int, Set[int]]:
        """Build a graph of overlapping request indices."""
        graph: Dict[int, Set[int]] = {i: set() for i in range(len(requests))}

        for i in range(len(requests)):
            for j in range(i + 1, len(requests)):
                req1, req2 = requests[i], requests[j]

                # Check scope overlap
                overlap = self.overlap_detector.detect_overlap(req1.scope, req2.scope)
                if not overlap.can_share_retrieval:
                    continue

                # Check policy compatibility
                if not self.policy_checker.are_compatible(req1.policy, req2.policy):
                    continue

                # Check overlap score threshold
                if overlap.score < self.min_overlap_score:
                    continue

                graph[i].add(j)
                graph[j].add(i)

        return graph

    def _find_connected_components(
        self, graph: Dict[int, Set[int]], n: int
    ) -> List[List[int]]:
        """Find connected components using BFS."""
        visited = set()
        components: List[List[int]] = []

        for i in range(n):
            if i in visited:
                continue

            component = []
            queue = [i]

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue

                visited.add(node)
                component.append(node)
                queue.extend(graph[node] - visited)

            if component:
                components.append(component)

        return components

    def _create_trivial_result(
        self, requests: List["RetrievalRequest"]
    ) -> RoutingResult:
        """Create a trivial routing result for single requests."""
        if not requests:
            return RoutingResult()

        req = requests[0]
        independent_requests = [req]

        return RoutingResult(
            independent_requests=independent_requests,
            routing_metadata={"strategy": "trivial", "reason": "single_request"},
        )

    def _create_bucket_from_requests(
        self,
        requests: List["RetrievalRequest"],
        bucket_id: str,
    ) -> RetrievalBucket:
        """Create a bucket from a group of requests."""
        if not requests:
            return None

        # Determine primary scope (most common)
        scope_counts: Dict[str, int] = defaultdict(int)
        for req in requests:
            scope_counts[req.scope.primary_scope] += 1

        primary_scope = max(scope_counts, key=scope_counts.get)

        # Collect memory types
        all_types: Set[MemoryType] = set()
        for req in requests:
            all_types.update(req.memory_types)

        # Merge policies
        merged_policy = self.policy_checker.merge_policies([r.policy for r in requests])

        key = requests[0].get_scope_key()

        return RetrievalBucket(
            bucket_id=bucket_id,
            scope_bucket_key=key,
            requests=requests,
            shared_scopes=[primary_scope],
            memory_types=list(all_types),
            merged_policy=merged_policy,
            is_shareable=True,
        )
