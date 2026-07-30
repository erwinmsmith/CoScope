"""Artifacts produced by LLMs, tools, and runtime components."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class ArtifactState(str, Enum):
    DRAFT = "draft"
    PROPOSED = "proposed"
    VERIFIED = "verified"
    COMMITTED = "committed"
    QUARANTINED = "quarantined"
    EXPIRED = "expired"
    REVOKED = "revoked"


class ArtifactType(str, Enum):
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    PLAN = "plan"
    CONSTRAINT = "constraint"
    TOOL_OBSERVATION = "tool_observation"
    RETRIEVED_EVIDENCE = "retrieved_evidence"
    REASONING_SUMMARY = "reasoning_summary"
    DECISION = "decision"
    VERIFICATION_RESULT = "verification_result"
    ERROR = "error"


@dataclass
class Artifact:
    content: str
    artifact_type: ArtifactType
    source_type: str
    source_id: str
    owner_agent_id: str
    artifact_id: str = field(default_factory=lambda: f"art_{uuid.uuid4().hex[:16]}")
    state: ArtifactState = ArtifactState.DRAFT
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, object] = field(default_factory=dict)
