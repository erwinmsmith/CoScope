"""Retrieval Retriever Module."""

from coscope.retrieval.retriever.base import CandidatePool, CandidateRetriever, RetrievalContext
from coscope.retrieval.retriever.shared import SharedCandidateRetriever
from coscope.retrieval.retriever.similarity import (
    SimilarityStrategy,
    CosineSimilarity,
    DotProductSimilarity,
)

__all__ = [
    "CandidateRetriever",
    "CandidatePool",
    "RetrievalContext",
    "SharedCandidateRetriever",
    "SimilarityStrategy",
    "CosineSimilarity",
    "DotProductSimilarity",
]
