"""
Feature extractors for reranking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from coscope.core.types import AgentRole, AgentState, MemoryEntry


class FeatureExtractor(ABC):
    """Base class for reranking feature extractors."""

    @abstractmethod
    def extract(
        self,
        query: str,
        candidate: "MemoryEntry",
        role: "AgentRole",
        state: "AgentState",
    ) -> float:
        """Extract a feature score."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Extractor", "").lower()


class SummaryScoreExtractor(FeatureExtractor):
    """Extracts summary/relevance score."""

    def extract(
        self,
        query: str,
        candidate: "MemoryEntry",
        role: "AgentRole",
        state: "AgentState",
    ) -> float:
        return float(candidate.confidence)


class RecencyExtractor(FeatureExtractor):
    """Extracts recency score based on timestamp."""

    def __init__(self, decay_factor: float = 0.1):
        self.decay_factor = decay_factor

    def extract(
        self,
        query: str,
        candidate: "MemoryEntry",
        role: "AgentRole",
        state: "AgentState",
    ) -> float:
        age = (datetime.utcnow() - candidate.created_at).total_seconds()
        age_hours = age / 3600
        return float(np.exp(-self.decay_factor * age_hours))


class AuthorityExtractor(FeatureExtractor):
    """Extracts authority score based on provenance."""

    SOURCE_WEIGHTS = {
        "verified": 1.0,
        "tool_output": 0.9,
        "agent_generated": 0.7,
        "user_input": 0.8,
        "external": 0.6,
        "unknown": 0.5,
    }

    def extract(
        self,
        query: str,
        candidate: "MemoryEntry",
        role: "AgentRole",
        state: "AgentState",
    ) -> float:
        source = candidate.provenance.source.lower()
        for key, weight in self.SOURCE_WEIGHTS.items():
            if key in source:
                return weight
        return self.SOURCE_WEIGHTS["unknown"]


class ProvenanceExtractor(FeatureExtractor):
    """Extracts provenance score."""

    def extract(
        self,
        query: str,
        candidate: "MemoryEntry",
        role: "AgentRole",
        state: "AgentState",
    ) -> float:
        score = 0.0
        if candidate.provenance.parent_event:
            score += 0.3
        if candidate.provenance.agent_id:
            score += 0.3
        if candidate.provenance.timestamp:
            score += 0.2
        if candidate.provenance.version > 1:
            score += 0.2 * min(candidate.provenance.version / 5, 1.0)
        return min(score, 1.0)


class KeywordOverlapExtractor(FeatureExtractor):
    """Extracts keyword overlap between query and content."""

    def extract(
        self,
        query: str,
        candidate: "MemoryEntry",
        role: "AgentRole",
        state: "AgentState",
    ) -> float:
        query_terms = set(query.lower().split())
        content_terms = set(candidate.content.lower().split())
        overlap = len(query_terms & content_terms)
        return overlap / max(len(query_terms), 1)
