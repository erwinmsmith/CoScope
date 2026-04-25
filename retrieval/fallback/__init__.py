"""Retrieval Fallback Module."""

from retrieval.fallback.base import FallbackTrigger, FallbackResult
from retrieval.fallback.strategy import (
    CombinedTrigger,
    CountBasedTrigger,
    PrivateFallbackRetriever,
)

__all__ = [
    "FallbackTrigger",
    "FallbackResult",
    "CountBasedTrigger",
    "CombinedTrigger",
    "PrivateFallbackRetriever",
]
