"""Retrieval Fallback Module."""

from coscope.retrieval.fallback.base import FallbackTrigger, FallbackResult
from coscope.retrieval.fallback.strategy import CountBasedTrigger, CombinedTrigger

__all__ = [
    "FallbackTrigger",
    "FallbackResult",
    "CountBasedTrigger",
    "CombinedTrigger",
]
