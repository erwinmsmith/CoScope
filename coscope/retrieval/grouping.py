"""Scope-compatible complete-link clustering with medoid representatives."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from coscope.memory.store import cosine_similarity
from coscope.retrieval.request import RetrievalRequest
from coscope.scope.effective_view import EffectiveView


@dataclass
class RetrievalGroup:
    group_id: str
    requests: list[RetrievalRequest]
    representative: RetrievalRequest
    shared_view: EffectiveView
    similarity_stats: dict[str, float]


class RequestGrouper:
    def __init__(
        self,
        *,
        medoid_threshold: float = 0.90,
        minimum_pairwise_similarity: float = 0.85,
        max_group_size: int = 16,
    ):
        self.medoid_threshold = medoid_threshold
        self.minimum_pairwise_similarity = minimum_pairwise_similarity
        self.max_group_size = max_group_size

    def group(self, requests: list[RetrievalRequest]) -> list[RetrievalGroup]:
        buckets: dict[str, list[RetrievalRequest]] = {}
        for request in requests:
            buckets.setdefault(request.scope_signature, []).append(request)

        groups: list[RetrievalGroup] = []
        for signature in sorted(buckets):
            clusters: list[list[RetrievalRequest]] = []
            for request in sorted(buckets[signature], key=lambda item: item.request_id):
                placed = False
                for cluster in clusters:
                    if self._can_join(request, cluster):
                        cluster.append(request)
                        placed = True
                        break
                if not placed:
                    clusters.append([request])
            groups.extend(self._make_group(signature, cluster) for cluster in clusters)
        return groups

    def _can_join(
        self, request: RetrievalRequest, cluster: list[RetrievalRequest]
    ) -> bool:
        if len(cluster) >= self.max_group_size:
            return False
        medoid = self._medoid(cluster)
        if (
            cosine_similarity(request.public_vector, medoid.public_vector)
            < self.medoid_threshold
        ):
            return False
        return all(
            cosine_similarity(request.public_vector, member.public_vector)
            >= self.minimum_pairwise_similarity
            for member in cluster
        )

    def _make_group(
        self, signature: str, requests: list[RetrievalRequest]
    ) -> RetrievalGroup:
        representative = self._medoid(requests)
        similarities = [
            cosine_similarity(left.public_vector, right.public_vector)
            for index, left in enumerate(requests)
            for right in requests[index + 1 :]
        ]
        group_key = signature + "|" + "|".join(sorted(r.request_id for r in requests))
        return RetrievalGroup(
            group_id=f"grp_{hashlib.sha256(group_key.encode()).hexdigest()[:16]}",
            requests=requests,
            representative=representative,
            shared_view=EffectiveView.shared_intersection(
                request.effective_view for request in requests
            ),
            similarity_stats={
                "minimum": min(similarities, default=1.0),
                "mean": sum(similarities) / len(similarities) if similarities else 1.0,
            },
        )

    @staticmethod
    def _medoid(requests: list[RetrievalRequest]) -> RetrievalRequest:
        if not requests:
            raise ValueError("cannot choose a medoid from an empty group")
        return max(
            requests,
            key=lambda candidate: (
                sum(
                    cosine_similarity(candidate.public_vector, other.public_vector)
                    for other in requests
                ),
                candidate.request_id,
            ),
        )
