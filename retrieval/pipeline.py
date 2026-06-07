"""
Retrieval pipeline orchestration.

This module wires the retrieval components into the end-to-end CoScope flow:
route requests into safe scope buckets, retrieve a shared candidate pool with
mean bucket queries, personalize results, optionally fallback to private
scopes, and fuse the final evidence set per agent.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from core.scope import ScopeRegistry
from core.types import (
    EmbeddingProvider,
    MemoryEntry,
    MemoryStore,
    RetrievedCandidate,
    RetrievalRequest,
    RetrievalResult,
    ScopeBucketKey,
)
from retrieval.fallback import CombinedTrigger, PrivateFallbackRetriever
from retrieval.fusion import ReciprocalRankFusion
from retrieval.reranker import RoleAwareReranker
from retrieval.router import RetrievalBucket, ScopeRouter


@dataclass
class PipelineConfig:
    """Runtime configuration for the retrieval pipeline."""

    query_embedding_dim: int = 1024
    shared_top_k: int = 50
    rerank_top_k: int = 20
    enable_rerank: bool = True
    enable_private_fallback: bool = True
    fallback_threshold: int = 5
    variant: str = "a4"


class RetrievalPipeline:
    """End-to-end collaborative retrieval pipeline."""

    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_provider: EmbeddingProvider,
        scope_registry: Optional[ScopeRegistry] = None,
        config: Optional[PipelineConfig] = None,
        router: Optional[ScopeRouter] = None,
    ):
        self.memory_store = memory_store
        self.embedding_provider = embedding_provider
        self.scope_registry = scope_registry or ScopeRegistry()
        self.config = config or PipelineConfig()

        self.router = router
        self.reranker = RoleAwareReranker()
        self.fallback_retriever = PrivateFallbackRetriever(
            memory_store=self.memory_store,
            embedding_provider=self.embedding_provider,
            default_threshold=self.config.fallback_threshold,
            max_candidates=self.config.rerank_top_k,
        )
        self.fallback_trigger = CombinedTrigger(min_count=self.config.fallback_threshold)
        self.fusion = ReciprocalRankFusion()

        self._stats: Dict[str, Any] = {
            "retrieval_calls": 0,
            "requests_processed": 0,
            "shareable_buckets": 0,
            "independent_requests": 0,
            "fallback_triggers": 0,
            "first_stage_retrievals": 0,
        }

    def retrieve(
        self,
        requests: List[RetrievalRequest],
        fit_projection: bool = True,
    ) -> List[RetrievalResult]:
        """Retrieve evidence for a batch of agent requests."""
        started = time.perf_counter()
        self._stats["retrieval_calls"] += 1
        self._stats["requests_processed"] += len(requests)
        variant = self._normalize_variant(self.config.variant)

        if variant == "a1":
            results = {
                request.request_id: self._retrieve_independent(
                    request,
                    use_fallback=False,
                    metadata_mode="a1_independent",
                )
                for request in requests
            }
            self._stats["independent_requests"] += len(requests)
            return self._order_results(requests, results, started)

        if variant == "a2":
            # Force-merge: all requests into a single shared pool, no scope /
            # policy filtering, no rerank, no fallback. Intentionally unsafe;
            # used as the Section 14.4.1 upper bound on FMR for S4 evaluation.
            results = self._retrieve_force_merge(requests)
            self._stats["shareable_buckets"] += 1
            return self._order_results(requests, results, started)

        # A3/A4 use scope-only buckets and mean bucket queries.
        supported_bucket_variants = {"a3", "a4", "a4_nofb", "a4_norerank"}
        if variant not in supported_bucket_variants:
            raise ValueError(
                f"Unsupported retrieval variant '{variant}'. "
                "Supported variants are a1, a2, a3, a4, a4_nofb, and a4_norerank."
            )
        routing = self._route_scope_only(requests)
        self._stats["shareable_buckets"] += len(routing.shareable_buckets)
        self._stats["independent_requests"] += len(routing.independent_requests)

        results: Dict[str, RetrievalResult] = {}

        for bucket in routing.shareable_buckets:
            if variant == "a3":
                bucket_results = self._retrieve_mean_bucket(
                    bucket,
                    enable_rerank=False,
                    enable_fallback=False,
                    metadata_mode="a3_scope_mean",
                )
            elif variant == "a4":
                bucket_results = self._retrieve_mean_bucket(
                    bucket,
                    enable_rerank=self.config.enable_rerank,
                    enable_fallback=self.config.enable_private_fallback,
                    metadata_mode="a4_shared_mean",
                )
            elif variant == "a4_nofb":
                bucket_results = self._retrieve_mean_bucket(
                    bucket,
                    enable_rerank=self.config.enable_rerank,
                    enable_fallback=False,
                    metadata_mode="a4_nofb_shared_mean",
                )
            elif variant == "a4_norerank":
                bucket_results = self._retrieve_mean_bucket(
                    bucket,
                    enable_rerank=False,
                    enable_fallback=self.config.enable_private_fallback,
                    metadata_mode="a4_norerank_shared_mean",
                )
            else:
                raise ValueError(f"Unsupported variant in bucket dispatch: {variant}")
            results.update({result.request_id: result for result in bucket_results})

        for request in routing.independent_requests:
            if variant in {"a3", "a4_norerank"}:
                rerank_override: Optional[bool] = False
            else:
                rerank_override = None
            result = self._retrieve_independent(
                request,
                use_fallback=variant not in {"a3", "a4_nofb"},
                metadata_mode=f"{variant}_independent",
                enable_rerank=rerank_override,
            )
            results[result.request_id] = result

        return self._order_results(requests, results, started)

    def _order_results(
        self,
        requests: List[RetrievalRequest],
        results: Dict[str, RetrievalResult],
        started: float,
    ) -> List[RetrievalResult]:
        latency_ms = (time.perf_counter() - started) * 1000
        ordered = []
        for request in requests:
            result = results.get(request.request_id)
            if result is None:
                result = RetrievalResult(
                    request_id=request.request_id,
                    agent_id=request.agent_id,
                    role=request.role,
                    candidates=[],
                    latency_ms=latency_ms,
                    metadata={"error": "request not processed"},
                )
            else:
                result.latency_ms = latency_ms
            ordered.append(result)
        return ordered

    def _retrieve_mean_bucket(
        self,
        bucket: RetrievalBucket,
        enable_rerank: bool,
        enable_fallback: bool,
        metadata_mode: str,
    ) -> List[RetrievalResult]:
        embeddings = [self.embedding_provider.embed_query(r.query) for r in bucket.requests]
        query_embedding = np.mean(np.vstack(embeddings), axis=0)
        query_embedding = self._l2_normalize(query_embedding)

        accessible = self._bucket_accessible_scopes(bucket)
        candidates = self.memory_store.search(
            query_embedding=query_embedding,
            scope_filter=accessible,
            memory_type_filter=bucket.memory_types or None,
            policy_filter=bucket.merged_policy,
            top_k=self.config.shared_top_k,
        )
        self._stats["first_stage_retrievals"] += 1
        memories = [candidate.memory for candidate in candidates]
        score_by_id = {candidate.memory.memory_id: candidate.score for candidate in candidates}

        out: List[RetrievalResult] = []
        for request in bucket.requests:
            shared = self._personalize(
                request=request,
                memories=memories,
                score_by_id=score_by_id,
                source="shared",
                enable_rerank=enable_rerank,
            )
            fallback = self._fallback(request, shared) if enable_fallback else self._no_fallback(
                "Fallback disabled for this experiment variant"
            )
            if fallback.triggered:
                self._stats["fallback_triggers"] += 1
            fusion = self.fusion.fuse(shared, fallback.candidates, top_k=self.config.rerank_top_k)
            out.append(
                RetrievalResult(
                    request_id=request.request_id,
                    agent_id=request.agent_id,
                    role=request.role,
                    candidates=fusion.candidates,
                    shared_candidates=shared,
                    private_candidates=fallback.candidates,
                    fallback_triggered=fallback.triggered,
                    metadata={
                        "mode": metadata_mode,
                        "bucket_id": bucket.bucket_id,
                        "pool_size": len(memories),
                        "fusion": fusion.metadata,
                        "fallback_reason": fallback.reason,
                    },
                )
            )
        return out

    def _retrieve_independent(
        self,
        request: RetrievalRequest,
        use_fallback: bool = True,
        metadata_mode: str = "independent",
        enable_rerank: Optional[bool] = None,
    ) -> RetrievalResult:
        query_embedding = self.embedding_provider.embed_query(request.query)
        candidates = self.memory_store.search(
            query_embedding=query_embedding,
            scope_filter=request.scope.all_scopes,
            memory_type_filter=request.memory_types,
            policy_filter=request.policy,
            top_k=self.config.shared_top_k,
        )
        self._stats["first_stage_retrievals"] += 1
        score_by_id = {c.memory.memory_id: c.score for c in candidates}
        shared = self._personalize(
            request=request,
            memories=[c.memory for c in candidates],
            score_by_id=score_by_id,
            source="independent",
            enable_rerank=enable_rerank,
        )
        fallback = self._fallback(request, shared) if use_fallback else self._no_fallback(
            "Fallback disabled for this experiment variant"
        )
        if fallback.triggered:
            self._stats["fallback_triggers"] += 1
        fusion = self.fusion.fuse(shared, fallback.candidates, top_k=self.config.rerank_top_k)
        return RetrievalResult(
            request_id=request.request_id,
            agent_id=request.agent_id,
            role=request.role,
            candidates=fusion.candidates,
            shared_candidates=shared,
            private_candidates=fallback.candidates,
            fallback_triggered=fallback.triggered,
            metadata={
                "mode": metadata_mode,
                "fallback_reason": fallback.reason,
                "fusion": fusion.metadata,
            },
        )

    def _retrieve_force_merge(
        self,
        requests: List[RetrievalRequest],
    ) -> Dict[str, RetrievalResult]:
        """
        A2 variant: force all requests into a single shared pool with no
        scope / policy filtering, mean-pooled query, no rerank, no fallback.

        This is the deliberately permissive baseline used in 搂14.4.1. Every
        request receives the same top-k list, sampled from the global memory
        pool. The variant exists to establish the upper bound of False Merge
        Rate on S4: if the system ignored scope/policy, verifier-only content
        would leak into non-verifier results. It is not a recommended mode.
        """
        embeddings = [self.embedding_provider.embed_query(r.query) for r in requests]
        query_embedding = self._l2_normalize(np.mean(np.vstack(embeddings), axis=0))

        candidates = self.memory_store.search(
            query_embedding=query_embedding,
            scope_filter=None,
            memory_type_filter=None,
            policy_filter=None,
            top_k=self.config.shared_top_k,
        )
        self._stats["first_stage_retrievals"] += 1
        top_k = self.config.rerank_top_k

        results: Dict[str, RetrievalResult] = {}
        for request in requests:
            shared = [
                RetrievedCandidate(
                    memory=candidate.memory,
                    score=candidate.score,
                    rank=rank,
                    source="a2_force_merge",
                    agent_id=request.agent_id,
                )
                for rank, candidate in enumerate(candidates[:top_k], start=1)
            ]
            results[request.request_id] = RetrievalResult(
                request_id=request.request_id,
                agent_id=request.agent_id,
                role=request.role,
                candidates=list(shared),
                shared_candidates=list(shared),
                private_candidates=[],
                fallback_triggered=False,
                metadata={
                    "mode": "a2_force_merge",
                    "pool_size": len(candidates),
                    "merged_agents": len(requests),
                },
            )
        return results

    def _personalize(
        self,
        request: RetrievalRequest,
        memories: List[MemoryEntry],
        score_by_id: Dict[str, float],
        source: str,
        enable_rerank: Optional[bool] = None,
    ) -> List[RetrievedCandidate]:
        if not memories:
            return []

        should_rerank = self.config.enable_rerank if enable_rerank is None else enable_rerank

        if should_rerank:
            reranked = self.reranker.rerank(
                query=request.query,
                candidates=memories,
                role=request.role,
                state=request.state,
                top_k=self.config.rerank_top_k,
            )
            candidates = []
            for item in reranked:
                base_score = score_by_id.get(item.candidate.memory_id, item.original_score)
                score = 0.5 * float(base_score) + 0.5 * float(item.reranked_score)
                candidates.append(
                    RetrievedCandidate(
                        memory=item.candidate,
                        score=score,
                        rank=item.rank,
                        source=source,
                        agent_id=request.agent_id,
                        relevance_labels=item.features,
                    )
                )
        else:
            candidates = [
                RetrievedCandidate(
                    memory=memory,
                    score=score_by_id.get(memory.memory_id, 0.0),
                    rank=i + 1,
                    source=source,
                    agent_id=request.agent_id,
                )
                for i, memory in enumerate(memories[: self.config.rerank_top_k])
            ]

        candidates.sort(key=lambda c: c.score, reverse=True)
        for rank, candidate in enumerate(candidates, 1):
            candidate.rank = rank
        return candidates[: self.config.rerank_top_k]

    def _fallback(
        self,
        request: RetrievalRequest,
        shared: List[RetrievedCandidate],
    ):
        if not self.config.enable_private_fallback:
            from retrieval.fallback import FallbackResult

            return FallbackResult(triggered=False, reason="Private fallback disabled")
        return self.fallback_retriever.retrieve(
            request=request,
            shared_candidates=shared,
            trigger=self.fallback_trigger,
            threshold=self.config.fallback_threshold,
        )

    def _no_fallback(self, reason: str):
        from retrieval.fallback import FallbackResult

        return FallbackResult(triggered=False, reason=reason)

    def _bucket_accessible_scopes(self, bucket: RetrievalBucket) -> List[str]:
        """
        Scopes the shared pool for this bucket may read.

        Routers build buckets around ``shared_scopes = [primary_scope]`` --
        typically a per-task ``task/.../shared`` scope. That alone excludes
        the workspace / corpus that every agent in the bucket is already
        cleared to read, so the shared pool shrinks to a handful of
        task-shared entries and rerank / fallback become no-ops for A3/A4.
        Here we additionally union in the **intersection** of all bucket
        requests' ``workspace_scopes``: that set is, by construction, shared
        by every agent in the bucket and safe to pool into shared retrieval.
        """
        scopes = list(bucket.shared_scopes)
        if bucket.requests:
            workspace_sets = [set(r.scope.workspace_scopes) for r in bucket.requests]
            if workspace_sets:
                shared_ws = set.intersection(*workspace_sets)
                for ws in sorted(shared_ws):
                    if ws not in scopes:
                        scopes.append(ws)
        return scopes

    def _route_scope_only(self, requests: List[RetrievalRequest]):
        from collections import defaultdict

        from retrieval.router import RetrievalBucket, RoutingResult

        groups: Dict[str, List[RetrievalRequest]] = defaultdict(list)
        for request in sorted(requests, key=lambda r: r.priority, reverse=True):
            groups[request.scope.primary_scope].append(request)

        shareable = []
        independent = []
        for scope_id, grouped in groups.items():
            if len(grouped) == 1:
                independent.extend(grouped)
                continue

            memory_types = []
            seen_types = set()
            for request in grouped:
                for memory_type in request.memory_types:
                    if memory_type not in seen_types:
                        seen_types.add(memory_type)
                        memory_types.append(memory_type)

            key = ScopeBucketKey(
                primary_scope=scope_id,
                memory_types=tuple(sorted(t.value for t in memory_types)),
                policy_hash=0,
            )
            shareable.append(
                RetrievalBucket(
                    bucket_id=f"scope_only_{int(hashlib.md5(scope_id.encode()).hexdigest()[:8], 16) % 100000:05d}",
                    scope_bucket_key=key,
                    requests=grouped,
                    shared_scopes=[scope_id],
                    memory_types=memory_types,
                    merged_policy=None,
                    is_shareable=True,
                    metadata={"strategy": "scope_only"},
                )
            )

        return RoutingResult(
            shareable_buckets=shareable,
            independent_requests=independent,
            routing_metadata={"strategy": "scope_only"},
        )

    def _normalize_variant(self, variant: str) -> str:
        aliases = {
            "independent": "a1",
            "per_agent_independent": "a1",
            "force_merge": "a2",
            "naive_sharing": "a2",
            "scope_only": "a3",
            "scope_mean": "a3",
            "shared_mean": "a4",
            "shared_mean_nofb": "a4_nofb",
            "a4_no_fallback": "a4_nofb",
            "shared_mean_norerank": "a4_norerank",
        }
        value = (variant or "a4").lower()
        return aliases.get(value, value)

    def _l2_normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def get_stats(self) -> Dict[str, Any]:
        """Return pipeline counters and component stats."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset runtime counters."""
        for key in self._stats:
            self._stats[key] = 0


def create_pipeline(
    memory_store: MemoryStore,
    embedding_provider: EmbeddingProvider,
    scope_registry: Optional[ScopeRegistry] = None,
    config: Optional[PipelineConfig] = None,
    **kwargs: Any,
) -> RetrievalPipeline:
    """Factory function for a RetrievalPipeline."""
    return RetrievalPipeline(
        memory_store=memory_store,
        embedding_provider=embedding_provider,
        scope_registry=scope_registry,
        config=config,
        **kwargs,
    )
