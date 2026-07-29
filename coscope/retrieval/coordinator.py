"""Scope-first shared retrieval coordinator."""

from __future__ import annotations

from dataclasses import dataclass

from coscope.adapters.embedding import EmbeddingAdapter
from coscope.memory.store import RuntimeMemoryStore
from coscope.retrieval.cache import SharedRetrievalCache
from coscope.retrieval.fallback import PrivateFallback
from coscope.retrieval.grouping import RequestGrouper
from coscope.retrieval.request import (
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalResult,
)
from coscope.retrieval.reranker import AgentReranker
from coscope.retrieval.shared_retriever import SharedRetriever


@dataclass
class RetrievalStats:
    requests: int = 0
    groups: int = 0
    shared_store_queries: int = 0
    private_store_queries: int = 0
    fallback_triggers: int = 0
    shared_cache_hits: int = 0


class RetrievalCoordinator:
    def __init__(
        self,
        store: RuntimeMemoryStore,
        embedder: EmbeddingAdapter,
        *,
        grouper: RequestGrouper | None = None,
        cache: SharedRetrievalCache | None = None,
        shared_candidate_k: int = 50,
    ):
        self.store = store
        self.embedder = embedder
        self.grouper = grouper or RequestGrouper()
        self.cache = cache or SharedRetrievalCache()
        self.shared_retriever = SharedRetriever(store, shared_candidate_k)
        self.reranker = AgentReranker(embedder)
        self.fallback = PrivateFallback(store, embedder)
        self.stats = RetrievalStats()

    def retrieve_batch(
        self, requests: list[RetrievalRequest]
    ) -> dict[str, RetrievalResult]:
        self.stats = RetrievalStats(requests=len(requests))
        self._validate(requests)
        groups = self.grouper.group(requests)
        self.stats.groups = len(groups)
        results: dict[str, RetrievalResult] = {}

        for group in groups:
            if not group.shared_view.scope_ids:
                # A private-only group has no safe shared corpus. Do not issue or
                # count an empty store query; each request proceeds to fallback.
                shared: list[RetrievalCandidate] = []
            else:
                cache_key = self.cache.make_key(
                    group,
                    embedding_model_version=self.embedder.model_version,
                    retrieval_parameters={
                        "candidate_k": self.shared_retriever.candidate_k
                    },
                )
                cached = self.cache.get(cache_key)
                if cached is None:
                    shared = self.shared_retriever.retrieve(group)
                    self.cache.put(cache_key, shared)
                    self.stats.shared_store_queries += 1
                else:
                    shared = cached
                    self.stats.shared_cache_hits += 1
            for request in group.requests:
                # Defense in depth: reuse must never expand an individual view.
                authorized = [
                    candidate
                    for candidate in shared
                    if candidate.memory.scope.scope_id in request.effective_view.scope_ids
                ]
                ranked_shared = self.reranker.rerank(request, authorized)
                fallback_candidates: list[RetrievalCandidate] = []
                triggered = self.fallback.should_run(request, ranked_shared)
                if triggered:
                    extra = request.effective_view.extra_from(group.shared_view)
                    fallback_candidates = self.fallback.retrieve(request, extra)
                    if extra.scope_ids:
                        self.stats.private_store_queries += 1
                    self.stats.fallback_triggers += 1
                final = self.reranker.rerank(
                    request, ranked_shared + fallback_candidates
                )[: request.retrieval_budget]
                results[request.request_id] = RetrievalResult(
                    request_id=request.request_id,
                    candidates=final,
                    shared_candidates=ranked_shared,
                    private_candidates=fallback_candidates,
                    fallback_triggered=triggered,
                    group_id=group.group_id,
                    metadata={
                        "shared_view_scope_count": len(group.shared_view.scope_ids),
                        "minimum_group_similarity": group.similarity_stats["minimum"],
                    },
                )
        return results

    @staticmethod
    def _validate(requests: list[RetrievalRequest]) -> None:
        ids = [request.request_id for request in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("request IDs must be unique")
        for request in requests:
            if not request.public_intent or not request.public_vector:
                raise ValueError("public intent and vector are required")
            if not request.scope_signature:
                raise ValueError("scope signature is required")
