"""Retrieval Retriever Module."""

from coscope.retrieval.retriever.base import CandidateRetriever
from coscope.retrieval.retriever.shared import SharedCandidateRetriever
from coscope.retrieval.retriever.similarity import (
    SimilarityStrategy,
    CosineSimilarity,
    DotProductSimilarity,
)

__all__ = [
    "CandidateRetriever",
    "SharedCandidateRetriever",
    "SimilarityStrategy",
    "CosineSimilarity",
    "DotProductSimilarity",
]
