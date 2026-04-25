"""
Rollout subpackage.

Contains the offline LLM-driven MAS rollout engine that materializes the
system-layer artifact trace for each episode (plans, scratches, conclusions,
audit reports). See `coscope/data/docs/SystemLayer_Rollout_Design_v1.md`.

Public surface is intentionally small; concrete LLM backends are opt-in
imports to avoid pulling heavy deps at package import time.
"""

from core.artifact_types import (
    ArtifactSlot,
    ArtifactTrace,
    SCOPE_LAYER_BY_SLOT,
    MEMORY_TYPE_BY_SLOT,
    VISIBILITY_BY_SLOT,
)
from core.interfaces import LLMClient, LLMResponse
from rollout.rollout_engine import ArtifactRolloutEngine

__all__ = [
    "ArtifactSlot",
    "ArtifactTrace",
    "SCOPE_LAYER_BY_SLOT",
    "MEMORY_TYPE_BY_SLOT",
    "VISIBILITY_BY_SLOT",
    "LLMClient",
    "LLMResponse",
    "ArtifactRolloutEngine",
]
