"""Shared first-stage retrieval over the safe view intersection."""

from coscope.memory.store import MemoryStore
from coscope.retrieval.grouping import RetrievalGroup
from coscope.retrieval.request import RetrievalCandidate


class SharedRetriever:
    def __init__(self, store: MemoryStore, candidate_k: int = 50):
        self.store = store
        self.candidate_k = candidate_k

    def retrieve(self, group: RetrievalGroup) -> list[RetrievalCandidate]:
        hits = self.store.search(
            group.representative.public_vector,
            group.shared_view,
            top_k=self.candidate_k,
            memory_types=group.representative.memory_types or None,
        )
        return [
            RetrievalCandidate(hit.memory, hit.score, "shared") for hit in hits
        ]
