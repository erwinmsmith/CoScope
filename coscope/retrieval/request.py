"""Retrieval request and result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from coscope.core.artifact import ArtifactType
from coscope.core.memory import MemoryEntry
from coscope.scope.effective_view import EffectiveView


@dataclass
class RetrievalRequest:
    request_id: str
    run_id: str
    agent_id: str
    reasoning_node_id: str
    full_query: str
    public_intent: str
    private_intent: str | None
    public_vector: tuple[float, ...]
    scope_signature: str
    effective_view: EffectiveView
    rerank_vector: tuple[float, ...] | None = None
    required_facets: tuple[str, ...] = ()
    context_budget: int = 2_000
    retrieval_budget: int = 20
    role: str = "agent"
    state_summary: str = ""
    memory_types: frozenset[str] = field(default_factory=frozenset)
    readable_artifact_types: frozenset[ArtifactType] = field(
        default_factory=frozenset
    )
    channel_budgets: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalCandidate:
    memory: MemoryEntry
    score: float
    source: str


@dataclass
class RetrievalResult:
    request_id: str
    candidates: list[RetrievalCandidate]
    shared_candidates: list[RetrievalCandidate] = field(default_factory=list)
    private_candidates: list[RetrievalCandidate] = field(default_factory=list)
    fallback_triggered: bool = False
    group_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
