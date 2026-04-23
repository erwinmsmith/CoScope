"""Retrieval Fallback Module."""

from coscope.retrieval.fallback.base import FallbackTrigger, FallbackResult
from coscope.retrieval.fallback.strategy import (
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
