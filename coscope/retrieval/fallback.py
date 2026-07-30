"""Private/local fallback retrieval."""

from __future__ import annotations

from coscope.adapters.embedding import EmbeddingAdapter
from coscope.memory.store import MemoryStore
from coscope.retrieval.request import RetrievalCandidate, RetrievalRequest
from coscope.scope.effective_view import EffectiveView


class PrivateFallback:
    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingAdapter,
        *,
        minimum_score: float = 0.15,
        top_k: int = 20,
    ):
        self.store = store
        self.embedder = embedder
        self.minimum_score = minimum_score
        self.top_k = top_k

    def should_run(
        self,
        request: RetrievalRequest,
        candidates: list[RetrievalCandidate],
    ) -> bool:
        if request.private_intent and request.effective_view.private_scope_ids:
            return True
        if not candidates or candidates[0].score < self.minimum_score:
            return True
        covered: set[str] = set()
        for candidate in candidates:
            raw_facets = candidate.memory.metadata.get("facets", ())
            if isinstance(raw_facets, str):
                covered.add(raw_facets)
            elif isinstance(raw_facets, (list, tuple, set, frozenset)):
                covered.update(str(facet) for facet in raw_facets)
        return any(facet not in covered for facet in request.required_facets)

    def retrieve(
        self,
        request: RetrievalRequest,
        extra_view: EffectiveView,
    ) -> list[RetrievalCandidate]:
        if not extra_view.scope_ids:
            return []
        query = request.private_intent or request.full_query
        hits = self.store.search(
            self.embedder.embed(query),
            extra_view,
            top_k=min(self.top_k, request.retrieval_budget),
            memory_types=request.memory_types or None,
        )
        return [
            RetrievalCandidate(hit.memory, hit.score, "private") for hit in hits
        ]
