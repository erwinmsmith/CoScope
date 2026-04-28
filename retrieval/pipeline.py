"""
Retrieval pipeline orchestration.

This module wires the retrieval components into the end-to-end CoScope flow:
route requests, build query matrices for shareable buckets, retrieve shared
candidates, personalize results, optionally fallback to private scopes, and
fuse the final evidence set per agent.
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
    PolicyConstraints,
    RetrievedCandidate,
    RetrievalRequest,
    RetrievalResult,
    ScopeBucketKey,
)
from retrieval.fallback import CombinedTrigger, PrivateFallbackRetriever
from retrieval.fusion import ReciprocalRankFusion
from retrieval.matrix import QueryMatrixBuilder
from retrieval.projection import ProjectionConfig, SharedProjectionModule
from retrieval.reranker import RoleAwareReranker
from retrieval.retriever import RetrievalContext, SharedCandidateRetriever
from retrieval.router import HierarchicalRouter, RetrievalBucket, ScopeRouter


@dataclass
class PipelineConfig:
    """Runtime configuration for the retrieval pipeline."""

    query_embedding_dim: int = 1024
    projection_dim: int = 256
    svd_rank: int = 64
    use_sparse_mask: bool = True
    shared_top_k: int = 50
    rerank_top_k: int = 20
    enable_rerank: bool = True
    enable_private_fallback: bool = True
    fallback_threshold: int = 5
    variant: str = "a5"
    # When set to a directory path, _retrieve_svd_bucket will dump per-bucket
    # artifacts (Q, full singular values S, Z, W_final, K_proj, plus a small
    # JSON of bucket metadata) into npz files for offline analysis. Disabled
    # by default to keep the hot path zero-overhead. The pipeline only writes
    # files; consumers (e.g. scripts/analyze_svd_dumps.py) handle aggregation.
    dump_svd_artifacts: Optional[str] = None


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

        self.router = router or HierarchicalRouter(self.scope_registry)
        self.matrix_builder = QueryMatrixBuilder(
            embedding_provider=self.embedding_provider,
            query_embedding_dim=self.config.query_embedding_dim,
        )
        self.projection = SharedProjectionModule(
            ProjectionConfig(
                input_dim=self.config.query_embedding_dim,
                intermediate_dim=self.config.projection_dim,
                svd_rank=self.config.svd_rank,
                use_mask=self.config.use_sparse_mask,
            )
        )
        self.shared_retriever = SharedCandidateRetriever(
            memory_store=self.memory_store,
            embedding_provider=self.embedding_provider,
            default_top_k=self.config.shared_top_k,
        )
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
            # A5 SVD path instrumentation: distinguish real truncated SVD from
            # the identity fallback taken when bucket size n <= svd_rank.
            "svd_real_projections": 0,
            "svd_identity_fallbacks": 0,
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

        # A7/A8 differ from A6 only in the query text supplied to retrieval:
        #   A7 = raw sub-question (no Step 2 LLM rewrite)
        #   A8 = LLM-generated query_intent (Step 2 enabled)
        # Both reuse the full A6 pipeline below. We mutate a shallow copy of
        # each request's query so downstream bucketing/encoding sees the new
        # string without changing the caller-visible object.
        if variant == "a8":
            rewritten = []
            for r in requests:
                qi = (r.metadata or {}).get("query_intent")
                if qi and isinstance(qi, str) and qi.strip():
                    new_req = self._clone_request_with_query(r, qi.strip())
                    rewritten.append(new_req)
                else:
                    rewritten.append(r)
            requests = rewritten
            variant = "a6"  # delegate to A6 pipeline below
        elif variant == "a7":
            variant = "a6"  # A7 = A6 on raw query (current default behavior)

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
            # used as the §14.4.1 upper bound on FMR for S4 evaluation.
            results = self._retrieve_force_merge(requests)
            self._stats["shareable_buckets"] += 1
            return self._order_results(requests, results, started)

        # Per paper §14.4.1: A5 uses scope-only routing (no block routing),
        # A6 adds hierarchical block routing on top of A5.
        # a4_norerank / a5_norerank: isolate SVD vs mean-query first-stage
        # ranking power by disabling the hybrid full-dim rerank.
        scope_only_routing = variant in {
            "a3", "a4", "a4_nofb", "a4_norerank", "a5", "a5_noproj", "a5_norerank"
        }
        use_identity_projection = variant == "a5_noproj"
        skip_fulldim_rerank = variant == "a5_norerank"
        is_svd_family = variant in {"a5", "a5_noproj", "a5_norerank", "a6"}

        routing = (
            self._route_scope_only(requests)
            if scope_only_routing
            else self.router.route(requests)
        )
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
            elif is_svd_family:
                bucket_results = self._retrieve_svd_bucket(
                    bucket,
                    metadata_mode=f"{variant}_query_matrix_svd",
                    use_identity_projection=use_identity_projection,
                    skip_fulldim_rerank=skip_fulldim_rerank,
                )
            else:
                raise ValueError(f"Unsupported variant in bucket dispatch: {variant}")
            results.update({result.request_id: result for result in bucket_results})

        for request in routing.independent_requests:
            result = self._retrieve_independent(
                request,
                use_fallback=variant not in {"a3", "a4_nofb"},
                metadata_mode=f"{variant}_independent",
                enable_rerank=(
                    False if variant in {"a4_norerank", "a5_norerank"} else None
                ),
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

    def _retrieve_svd_bucket(
        self,
        bucket: RetrievalBucket,
        metadata_mode: str,
        use_identity_projection: bool = False,
        skip_fulldim_rerank: bool = False,
    ) -> List[RetrievalResult]:
        query_matrix = self.matrix_builder.build(bucket.requests, self.embedding_provider)
        projection_result = self._svd_projection(
            query_matrix,
            use_identity_projection=use_identity_projection,
        )
        strategy = projection_result.metadata.get("strategy", "")
        if strategy == "identity_fallback_for_small_bucket":
            self._stats["svd_identity_fallbacks"] += 1
        else:
            self._stats["svd_real_projections"] += 1

        context = RetrievalContext(
            scope_id=bucket.primary_scope,
            memory_types=bucket.memory_types,
            policy=bucket.merged_policy,
            scope_spec=self._bucket_scope_spec(bucket),
        )
        pool = self.shared_retriever.retrieve(
            projection_result,
            context,
            top_k=self.config.shared_top_k,
        )
        self._stats["first_stage_retrievals"] += 1

        # Hybrid A5: SVD selects the shared pool in batch, but per-agent
        # ranking uses the original full-dim embedding space to avoid the
        # resolution loss incurred by scoring in a min(n, svd_rank)-dim
        # subspace. For the a5_norerank ablation we instead score each agent
        # in the SVD subspace directly (Z[i] vs K), which is the faithful
        # realization of "Q matrix + projection + no full-dim rerank" and
        # isolates the first-stage ranking power of SVD vs mean query.
        pool_embeddings = np.vstack(
            [self._embedding_for(memory) for memory in pool.candidates]
        ) if pool.candidates else np.zeros((0, 1), dtype="float32")

        # Optional per-bucket artifact dump for offline SVD analysis.
        if self.config.dump_svd_artifacts and strategy == "query_matrix_truncated_svd":
            self._dump_svd_bucket(
                bucket=bucket,
                query_matrix=query_matrix,
                projection_result=projection_result,
                pool=pool,
                pool_embeddings=pool_embeddings,
            )

        if skip_fulldim_rerank and pool.candidates:
            # Pre-compute K in the projected subspace so scoring costs O(n*r)
            # per bucket instead of O(n*k). Z is (n, r), K is (m, r).
            W_final = projection_result.projection_matrix
            Z_all = projection_result.projected
            K_proj = pool_embeddings @ W_final
            K_norms = np.linalg.norm(K_proj, axis=1, keepdims=True)
            K_norms[K_norms == 0] = 1.0
            K_proj_normed = K_proj / K_norms

        out: List[RetrievalResult] = []
        for idx, request in enumerate(bucket.requests):
            if pool.candidates:
                if skip_fulldim_rerank:
                    z_vec = Z_all[idx]
                    z_norm = np.linalg.norm(z_vec)
                    if z_norm == 0:
                        z_norm = 1.0
                    scores = (K_proj_normed @ z_vec) / z_norm
                else:
                    q_vec = query_matrix.embeddings[:, idx]
                    q_vec = self._l2_normalize(q_vec)
                    pool_norms = np.linalg.norm(pool_embeddings, axis=1, keepdims=True)
                    pool_norms[pool_norms == 0] = 1.0
                    scores = (pool_embeddings @ q_vec) / pool_norms.squeeze(-1)
                score_by_id = {
                    memory.memory_id: float(scores[i])
                    for i, memory in enumerate(pool.candidates)
                }
            else:
                score_by_id = {}

            # Re-sort pool by per-agent score so _personalize (which walks
            # memories in order when rerank is off) picks up the right
            # ranking. For the rerank path, order is overridden by reranker.
            if pool.candidates and skip_fulldim_rerank:
                sorted_pairs = sorted(
                    zip(pool.candidates, scores), key=lambda p: -float(p[1])
                )
                personalized_memories = [m for m, _ in sorted_pairs]
            else:
                personalized_memories = pool.candidates

            shared = self._personalize(
                request=request,
                memories=personalized_memories,
                score_by_id=score_by_id,
                source="shared",
                enable_rerank=False if skip_fulldim_rerank else None,
            )
            fallback = self._fallback(request, shared)
            if fallback.triggered:
                self._stats["fallback_triggers"] += 1
            fusion = self.fusion.fuse(
                shared,
                fallback.candidates,
                top_k=self.config.rerank_top_k,
            )
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
                        "pool_size": pool.size,
                        "fusion": fusion.metadata,
                        "fallback_reason": fallback.reason,
                        "projection": projection_result.metadata,
                    },
                )
            )
        return out

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
            scope_filter=accessible or None,
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
            scope_filter=request.scope.all_scopes or None,
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

        This is the deliberately permissive baseline used in §14.4.1. Every
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

    def _bucket_scope_spec(self, bucket: RetrievalBucket):
        from core.types import ScopeSpec

        shared_scopes = list(bucket.shared_scopes)
        workspace_scopes: List[str] = []
        if bucket.requests:
            ws_sets = [set(r.scope.workspace_scopes) for r in bucket.requests]
            if ws_sets:
                workspace_scopes = sorted(set.intersection(*ws_sets))
        return ScopeSpec(
            shared_scopes=shared_scopes,
            workspace_scopes=workspace_scopes,
        )

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

    def _svd_projection(self, query_matrix, use_identity_projection: bool = False):
        from retrieval.projection import ProjectionResult

        Q = query_matrix.embeddings
        n = query_matrix.num_queries

        # A5-noproj ablation: skip SVD entirely and use full-rank identity
        # projection. Equivalent to scoring each query independently in the
        # original embedding space but still fetching the shared pool in a
        # single batch. Quantifies the marginal benefit of SVD compression.
        if use_identity_projection:
            k_dim = Q.shape[0]
            W_final = np.eye(k_dim, dtype="float32")
            Z = Q.T
            return ProjectionResult(
                projected=Z,
                projection_matrix=W_final,
                svd_rank=k_dim,
                mask_applied=False,
                metadata={
                    "strategy": "identity_projection_ablation",
                    "num_queries": n,
                    "Z_shape": Z.shape,
                    "W_final_shape": W_final.shape,
                },
            )

        if n < 3:
            # Truncated SVD is only meaningful for n >= 3. For n in {1, 2} we
            # fall back to a full-rank identity projection so each query keeps
            # all k embedding dimensions. The previous implementation used
            # np.eye(k, r) which silently discarded (k - r) of the k dims
            # (e.g. 352/384 for sentence-transformers + svd_rank=32) and
            # collapsed A5 on small buckets to an incoherent low-rank match.
            k_dim = Q.shape[0]
            W_final = np.eye(k_dim, dtype="float32")  # (k, k), no reduction
            Z = Q.T  # (n, k)
            return ProjectionResult(
                projected=Z,
                projection_matrix=W_final,
                svd_rank=k_dim,
                mask_applied=False,
                metadata={
                    "strategy": "identity_fallback_for_small_bucket",
                    "num_queries": n,
                    "Z_shape": Z.shape,
                    "W_final_shape": W_final.shape,
                },
            )

        X = Q.T
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        r = min(self.config.svd_rank, U.shape[1], VT.shape[0])
        Z = U[:, :r] * S[:r]
        W_final = VT[:r, :].T
        return ProjectionResult(
            projected=Z,
            projection_matrix=W_final,
            svd_rank=r,
            mask_applied=False,
            metadata={
                "strategy": "query_matrix_truncated_svd",
                "num_queries": n,
                "singular_values_top5": S[:5].tolist() if len(S) >= 5 else S.tolist(),
                # Full spectrum is needed for offline rank-energy analysis when
                # --dump-svd-artifacts is enabled. Kept as a list (small: <=k).
                "singular_values_full": S.tolist(),
                "Z_shape": Z.shape,
                "W_final_shape": W_final.shape,
            },
        )

    def _dump_svd_bucket(
        self,
        bucket,
        query_matrix,
        projection_result,
        pool,
        pool_embeddings,
    ) -> None:
        """Persist per-bucket SVD artifacts to ``self.config.dump_svd_artifacts``.

        Layout::

            {dump_root}/{episode_id}/{bucket_id}.npz   (Q, S, Z, W_final, K_proj)
            {dump_root}/{episode_id}/{bucket_id}.json  (bucket metadata)

        ``episode_id`` is read from the first request's metadata; missing
        episode_id falls back to ``"_unknown"``. Failures are logged but do
        not raise (analysis hooks must never break the eval hot path).
        """
        import json as _json
        import os as _os
        import numpy as _np

        try:
            requests = list(bucket.requests)
            episode_id = "_unknown"
            if requests:
                ep = (requests[0].metadata or {}).get("episode_id")
                if ep:
                    episode_id = str(ep)
            bucket_id = getattr(bucket, "bucket_id", None) or getattr(bucket, "primary_scope", "bucket")
            safe_bucket = str(bucket_id).replace("/", "_").replace(":", "_")

            root = self.config.dump_svd_artifacts
            ep_dir = _os.path.join(root, episode_id)
            _os.makedirs(ep_dir, exist_ok=True)
            npz_path = _os.path.join(ep_dir, f"{safe_bucket}.npz")
            json_path = _os.path.join(ep_dir, f"{safe_bucket}.json")

            W_final = projection_result.projection_matrix
            Z = projection_result.projected
            K_proj = (
                pool_embeddings @ W_final
                if pool_embeddings.size and pool_embeddings.shape[1] == W_final.shape[0]
                else _np.zeros((0, W_final.shape[1]), dtype="float32")
            )
            S_full = _np.asarray(
                projection_result.metadata.get("singular_values_full", []),
                dtype="float32",
            )
            _np.savez_compressed(
                npz_path,
                Q=query_matrix.embeddings.astype("float32"),
                S=S_full,
                Z=Z.astype("float32"),
                W_final=W_final.astype("float32"),
                K_proj=K_proj.astype("float32"),
                pool_embeddings=pool_embeddings.astype("float32"),
            )

            meta = {
                "episode_id": episode_id,
                "bucket_id": str(bucket_id),
                "primary_scope": getattr(bucket, "primary_scope", None),
                "n_queries": int(Z.shape[0]),
                "k_embed_dim": int(W_final.shape[0]),
                "r_svd": int(W_final.shape[1]),
                "n_pool_candidates": int(pool_embeddings.shape[0]),
                "shared_top_k": int(self.config.shared_top_k),
                "svd_rank_config": int(self.config.svd_rank),
                "request_ids": [r.request_id for r in requests],
                "agent_ids": [r.agent_id for r in requests],
                "agent_roles": [getattr(r.role, "value", str(r.role)) for r in requests],
                "candidate_memory_ids": [m.memory_id for m in pool.candidates],
            }
            with open(json_path, "w", encoding="utf-8") as f:
                _json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - analysis hook is best-effort
            logger.warning("dump_svd_artifacts failed: %s", exc)

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
            "query_matrix_svd": "a5",
            "svd": "a5",
            "query_matrix_identity": "a5_noproj",
            "a5_identity": "a5_noproj",
            "query_matrix_norerank": "a5_norerank",
            "shared_mean_norerank": "a4_norerank",
            "query_matrix_svd_block_routing": "a6",
            "coscope": "a6",
            "full": "a6",
            "raw_question_no_step2": "a7",
            "coscope_no_step2": "a7",
            "step2_rewrite": "a8",
            "coscope_with_step2": "a8",
        }
        value = (variant or "a6").lower()
        return aliases.get(value, value)

    def _clone_request_with_query(
        self, request: RetrievalRequest, new_query: str
    ) -> RetrievalRequest:
        """Shallow-clone a request while overriding the query string.

        Used by the A8 variant to inject LLM-rewritten query_intent without
        mutating the caller-visible request object. All other fields (scope,
        policy, memory_types, metadata, etc.) are preserved as-is.
        """
        return RetrievalRequest(
            request_id=request.request_id,
            agent_id=request.agent_id,
            role=request.role,
            query=new_query,
            scope=request.scope,
            memory_types=list(request.memory_types),
            policy=request.policy,
            state=request.state,
            priority=request.priority,
            metadata=dict(request.metadata or {}),
        )

    def _embedding_for(self, memory) -> np.ndarray:
        """Return the embedding for a memory, encoding its content if needed."""
        if memory.embedding is not None:
            return np.asarray(memory.embedding, dtype="float32")
        return self.embedding_provider.embed_query(memory.content)

    def _l2_normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def get_stats(self) -> Dict[str, Any]:
        """Return pipeline counters and component stats."""
        stats = dict(self._stats)
        stats["projection"] = self.projection.get_stats()
        return stats

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
