"""Retrieval Fusion Module."""

from coscope.retrieval.fusion.base import EvidenceFusion, FusionResult
from coscope.retrieval.fusion.strategies import (
    ReciprocalRankFusion,
    ScoreWeightedFusion,
)

__all__ = [
    "EvidenceFusion",
    "FusionResult",
    "ReciprocalRankFusion",
    "ScoreWeightedFusion",
]
