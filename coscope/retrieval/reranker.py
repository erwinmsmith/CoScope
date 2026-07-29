"""Per-agent ranking over already authorized candidates."""

from __future__ import annotations

from coscope.adapters.embedding import EmbeddingAdapter
from coscope.memory.store import cosine_similarity
from coscope.retrieval.request import RetrievalCandidate, RetrievalRequest


class AgentReranker:
    def __init__(self, embedder: EmbeddingAdapter):
        self.embedder = embedder

    def rerank(
        self,
        request: RetrievalRequest,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        query_vector = request.rerank_vector
        if query_vector is None:
            query_vector = self.embedder.embed(
                " ".join(
                    part
                    for part in (request.full_query, request.state_summary)
                    if part
                )
            )
            request.rerank_vector = query_vector
        seen: set[str] = set()
        ranked: list[RetrievalCandidate] = []
        for candidate in candidates:
            if candidate.memory.memory_id in seen:
                continue
            seen.add(candidate.memory.memory_id)
            relevance = (
                cosine_similarity(query_vector, candidate.memory.vector)
                if candidate.memory.vector is not None
                else 0.0
            )
            raw_roles = candidate.memory.metadata.get("roles", ())
            if isinstance(raw_roles, str):
                roles = {raw_roles}
            elif isinstance(raw_roles, (list, tuple, set, frozenset)):
                roles = {str(role) for role in raw_roles}
            else:
                roles = set()
            role_fit = 0.1 if not roles or request.role in roles else -0.1
            score = 0.55 * relevance + 0.35 * candidate.score + role_fit
            ranked.append(
                RetrievalCandidate(candidate.memory, score, candidate.source)
            )
        ranked.sort(key=lambda item: (-item.score, item.memory.memory_id))
        return ranked
