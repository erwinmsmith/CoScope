"""Mutable state owned by one runtime run."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TeamEvidenceState:
    committed_evidence_ids: set[str] = field(default_factory=set)
    covered_facets: set[str] = field(default_factory=set)
    unresolved_questions: set[str] = field(default_factory=set)
    conflicting_claims: dict[str, set[str]] = field(default_factory=dict)
    evidence_owners: dict[str, str] = field(default_factory=dict)


@dataclass
class RunState:
    run_id: str
    task_id: str
    created_at: float = field(default_factory=time.time)
    status: str = "active"
    team_evidence: TeamEvidenceState = field(default_factory=TeamEvidenceState)
    metadata: dict[str, object] = field(default_factory=dict)
