"""
Hierarchical routing strategy for scope-based bucketing.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from coscope.core.scope import PolicyCompatibilityChecker, ScopeRegistry
from coscope.core.types import MemoryType, PolicyConstraints, RetrievalRequest

from .base import RetrievalBucket, RoutingResult, ScopeRouter

logger = logging.getLogger(__name__)


class HierarchicalRouter(ScopeRouter):
    """
    Hierarchical routing strategy.

    Processing order:
    1. Group by primary scope
    2. Split by memory type
    3. Check policy compatibility
    4. Split on policy conflict
    """

    def __init__(self, scope_registry: Any = None):
        super().__init__(scope_registry)
        self.policy_checker = PolicyCompatibilityChecker()

    def route(
        self,
        requests: List["RetrievalRequest"],
    ) -> RoutingResult:
        """Route requests using hierarchical strategy."""
        if not requests:
            return RoutingResult()

        # Sort by priority (higher priority first)
        sorted_requests = sorted(requests, key=lambda r: r.priority, reverse=True)

        # Phase 1: Group by primary scope
        scope_groups: Dict[str, List[RetrievalRequest]] = defaultdict(list)
        for req in sorted_requests:
            primary = req.scope.primary_scope
            scope_groups[primary].append(req)

        shareable_buckets: List[RetrievalBucket] = []
        independent_requests: List[RetrievalRequest] = []

        # Phase 2: Within each scope group, split by memory type
        for scope_id, scope_requests in scope_groups.items():
            buckets_by_type = self._group_by_memory_type(scope_id, scope_requests)

            for mem_types, type_requests in buckets_by_type.items():
                # Phase 3: Check policy compatibility
                compatible_groups = self._split_by_policy(
                    scope_id, mem_types, type_requests
                )

                for group in compatible_groups:
                    if len(group.requests) == 1:
                        independent_requests.extend(group.requests)
                    elif group.is_shareable:
                        shareable_buckets.append(group)
                    else:
                        independent_requests.extend(group.requests)

        logger.info(
            f"Routing complete: {len(shareable_buckets)} buckets, "
            f"{len(independent_requests)} independent"
        )

        return RoutingResult(
            shareable_buckets=shareable_buckets,
            independent_requests=independent_requests,
            routing_metadata={
                "strategy": "hierarchical",
                "total_buckets": len(shareable_buckets),
                "total_independent": len(independent_requests),
                "scope_groups": len(scope_groups),
            },
        )

    def _group_by_memory_type(
        self, scope_id: str, requests: List["RetrievalRequest"]
    ) -> Dict[Tuple[MemoryType, ...], List["RetrievalRequest"]]:
        """Group requests by memory type tuple."""
        groups: Dict[Tuple[MemoryType, ...], List["RetrievalRequest"]] = defaultdict(
            list
        )
        for req in requests:
            key = tuple(sorted(req.memory_types))
            groups[key].append(req)
        return groups

    def _split_by_policy(
        self,
        scope_id: str,
        memory_types: Tuple[MemoryType, ...],
        requests: List["RetrievalRequest"],
    ) -> List[RetrievalBucket]:
        """Split requests by policy compatibility."""
        if len(requests) <= 1:
            policy_hash = self._compute_policy_hash(requests[0].policy) if requests else 0
            key = requests[0].get_scope_key() if requests else None

            return [
                RetrievalBucket(
                    bucket_id=f"bucket_{hashlib.md5(str(policy_hash).encode()).hexdigest()[:8]}",
                    scope_bucket_key=key or requests[0].get_scope_key(),
                    requests=requests,
                    shared_scopes=[scope_id],
                    memory_types=list(memory_types),
                    merged_policy=requests[0].policy if requests else PolicyConstraints(),
                    is_shareable=True,
                )
            ]

        # Find compatible groups
        remaining = list(requests)
        buckets: List[RetrievalBucket] = []

        while remaining:
            current = remaining[0]
            remaining = remaining[1:]
            compatible = [current]

            for i, other in enumerate(remaining):
                if self.policy_checker.are_compatible(current.policy, other.policy):
                    compatible.append(other)

            # Remove compatible from remaining
            for i in reversed(range(len(remaining))):
                if self.policy_checker.are_compatible(current.policy, remaining[i].policy):
                    compatible.append(remaining[i])
                    remaining.pop(i)

            # Merge policies
            merged_policy = self.policy_checker.merge_policies([r.policy for r in compatible])

            is_shareable = all(
                self.policy_checker.are_compatible(c.policy, merged_policy) for c in compatible
            )

            policy_hash = self._compute_policy_hash(merged_policy)
            key = compatible[0].get_scope_key()

            buckets.append(
                RetrievalBucket(
                    bucket_id=f"bucket_{hashlib.md5(str(policy_hash).encode()).hexdigest()[:8]}",
                    scope_bucket_key=key,
                    requests=compatible,
                    shared_scopes=[scope_id],
                    memory_types=list(memory_types),
                    merged_policy=merged_policy,
                    is_shareable=is_shareable,
                )
            )

        return buckets

    def _compute_policy_hash(self, policy: PolicyConstraints) -> int:
        """Compute a hash for policy deduplication."""
        import hashlib

        key_parts = [
            ",".join(sorted(v.value for v in policy.visibility)),
            str(policy.max_clearance),
            ",".join(sorted(policy.excluded_zones)),
        ]
        return int(hashlib.md5("|".join(key_parts).encode()).hexdigest(), 16)
