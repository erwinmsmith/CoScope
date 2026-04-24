"""
Core type definitions for CoScope.

This module defines the fundamental types used throughout the CoScope system,
providing a common type system for memory, agents, scopes, and retrieval.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# Scope Types
# ============================================================


class ScopeType(str, Enum):
    """Memory scope types as defined in the design document."""

    AGENT_PRIVATE = "agent_private"
    TASK_SHARED = "task_shared"
    SESSION = "session"
    WORKSPACE = "workspace"
    GOVERNED = "governed"


# ============================================================
# Memory Types
# ============================================================


class MemoryType(str, Enum):
    """Types of memory in the system."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    ARTIFACT = "artifact"
    SHARED = "shared"
    WORKING = "working"


# ============================================================
# Agent Roles
# ============================================================


class AgentRole(str, Enum):
    """Agent roles in a multi-agent system."""

    PLANNER = "planner"
    SOLVER = "solver"
    VERIFIER = "verifier"
    CRITIC = "critic"
    MEMORY_MANAGER = "memory_manager"
    CUSTOM = "custom"


# ============================================================
# Visibility & Policy
# ============================================================


class VisibilityLevel(str, Enum):
    """Visibility levels for memory access control."""

    OWNER = "owner"
    TEAM = "team"
    SESSION = "session"
    PUBLIC = "public"
    RESTRICTED = "restricted"


@dataclass
class PolicyConstraints:
    """
    Access control, visibility, and isolation constraints for a retrieval request.
    """

    visibility: List[VisibilityLevel] = field(
        default_factory=lambda: [VisibilityLevel.TEAM]
    )
    max_clearance: int = 3
    excluded_zones: List[str] = field(
        default_factory=lambda: ["quarantine", "audit_hold"]
    )
    audit_required: bool = False

    def is_compatible_with(self, other: PolicyConstraints) -> bool:
        """
        Check if this policy is compatible with another policy.
        Two policies are compatible if they can share retrieval results.
        """
        # Visibility must overlap
        if not set(self.visibility) & set(other.visibility):
            return False

        # Shared retrieval only happens when policy boundaries are equivalent.
        if self.max_clearance != other.max_clearance:
            return False

        if set(self.excluded_zones) != set(other.excluded_zones):
            return False

        if self.audit_required != other.audit_required:
            return False

        return True

    def policy_hash(self) -> int:
        """Compute a hash for policy deduplication."""
        import hashlib

        key_parts = [
            ",".join(sorted(v.value for v in self.visibility)),
            str(self.max_clearance),
            ",".join(sorted(self.excluded_zones)),
        ]
        hash_str = "|".join(key_parts)
        return int(hashlib.md5(hash_str.encode()).hexdigest(), 16)


# ============================================================
# Agent State
# ============================================================


@dataclass
class AgentState:
    """
    Represents the current state of an agent.

    Includes task context, known evidence, tool usage, reasoning branches, etc.
    """

    plan_node: Optional[str] = None
    known_evidence_ids: List[str] = field(default_factory=list)
    tool_context: Dict[str, Any] = field(default_factory=dict)
    reasoning_branch: Optional[str] = None
    task_context: Dict[str, Any] = field(default_factory=dict)
    custom_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_node": self.plan_node,
            "known_evidence_ids": self.known_evidence_ids,
            "tool_context": self.tool_context,
            "reasoning_branch": self.reasoning_branch,
            "task_context": self.task_context,
            "custom_state": self.custom_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentState:
        return cls(
            plan_node=data.get("plan_node"),
            known_evidence_ids=data.get("known_evidence_ids", []),
            tool_context=data.get("tool_context", {}),
            reasoning_branch=data.get("reasoning_branch"),
            task_context=data.get("task_context", {}),
            custom_state=data.get("custom_state", {}),
        )


# ============================================================
# Scope Specification
# ============================================================


@dataclass
class ScopeSpec:
    """
    Specifies which memory scopes a retrieval request can access.
    """

    private_scopes: List[str] = field(default_factory=list)
    shared_scopes: List[str] = field(default_factory=list)
    workspace_scopes: List[str] = field(default_factory=list)
    governed_scopes: List[str] = field(default_factory=list)

    @property
    def primary_scope(self) -> str:
        """The primary scope for shared retrieval decisions."""
        if self.shared_scopes:
            return self.shared_scopes[0]
        elif self.workspace_scopes:
            return self.workspace_scopes[0]
        elif self.governed_scopes:
            return self.governed_scopes[0]
        elif self.private_scopes:
            return self.private_scopes[0]
        return "default"

    @property
    def all_scopes(self) -> List[str]:
        """All scopes this request can access."""
        return (
            self.private_scopes
            + self.shared_scopes
            + self.workspace_scopes
            + self.governed_scopes
        )

    @property
    def is_shared(self) -> bool:
        """Whether this request accesses shared scopes."""
        return bool(self.shared_scopes or self.workspace_scopes or self.governed_scopes)


# ============================================================
# Retrieval Request
# ============================================================


@dataclass
class RetrievalRequest:
    """
    A standardized retrieval request from an agent.

    This is the fundamental unit of retrieval in CoScope, representing
    a single agent's retrieval needs with all necessary context.
    """

    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}")
    agent_id: str = ""
    role: AgentRole = AgentRole.CUSTOM
    query: str = ""
    scope: ScopeSpec = field(default_factory=ScopeSpec)
    memory_types: List[MemoryType] = field(default_factory=lambda: [MemoryType.EPISODIC])
    policy: PolicyConstraints = field(default_factory=PolicyConstraints)
    state: AgentState = field(default_factory=AgentState)
    priority: int = 0  # Higher priority requests processed first
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_scope_key(self) -> ScopeBucketKey:
        """
        Get the bucket key for scope-based routing.
        This determines which shared retrieval bucket this request belongs to.
        """
        return ScopeBucketKey(
            primary_scope=self.scope.primary_scope,
            memory_types=tuple(sorted(self.memory_types)),
            policy_hash=self.policy.policy_hash(),
        )


@dataclass(frozen=True)
class ScopeBucketKey:
    """
    A hashable key for scope-based request bucketing.
    Requests with the same ScopeBucketKey can share first-stage retrieval.
    """

    primary_scope: str
    memory_types: tuple[str, ...]
    policy_hash: int


# ============================================================
# Memory Objects
# ============================================================


@dataclass
class Provenance:
    """Provenance information for a memory entry."""

    source: str = ""
    parent_event: Optional[str] = None
    agent_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEntry:
    """
    A single memory entry in the CoScope system.
    """

    memory_id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    scope_id: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC
    content: str = ""
    embedding: Optional[np.ndarray] = None
    embedding_dim: int = 0
    visibility: List[VisibilityLevel] = field(
        default_factory=lambda: [VisibilityLevel.TEAM]
    )
    confidence: float = 0.0
    salience: float = 0.0
    ttl: Optional[int] = None  # Time-to-live in seconds
    provenance: Provenance = field(default_factory=Provenance)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def is_expired(self) -> bool:
        """Check if this memory entry has expired based on TTL."""
        if self.ttl is None:
            return False
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "scope_id": self.scope_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "visibility": [v.value for v in self.visibility],
            "confidence": self.confidence,
            "salience": self.salience,
            "ttl": self.ttl,
            "provenance": {
                "source": self.provenance.source,
                "parent_event": self.provenance.parent_event,
                "agent_id": self.provenance.agent_id,
                "timestamp": self.provenance.timestamp.isoformat()
                if self.provenance.timestamp
                else None,
                "version": self.provenance.version,
            },
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryEntry:
        provenance_data = data.get("provenance", {})
        provenance = Provenance(
            source=provenance_data.get("source", ""),
            parent_event=provenance_data.get("parent_event"),
            agent_id=provenance_data.get("agent_id"),
            timestamp=datetime.fromisoformat(provenance_data["timestamp"])
            if provenance_data.get("timestamp")
            else None,
            version=provenance_data.get("version", 1),
        )
        visibility = [
            VisibilityLevel(v) if isinstance(v, str) else v
            for v in data.get("visibility", ["team"])
        ]
        return cls(
            memory_id=data.get("memory_id", f"mem_{uuid.uuid4().hex[:12]}"),
            scope_id=data.get("scope_id", ""),
            memory_type=MemoryType(data.get("memory_type", "episodic")),
            content=data.get("content", ""),
            visibility=visibility,
            confidence=data.get("confidence", 0.0),
            salience=data.get("salience", 0.0),
            ttl=data.get("ttl"),
            provenance=provenance,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else datetime.utcnow(),
        )


# ============================================================
# Retrieval Results
# ============================================================


@dataclass
class RetrievedCandidate:
    """
    A candidate memory entry returned from retrieval.
    """

    memory: MemoryEntry
    score: float
    rank: int = 0
    source: str = "shared"  # "shared", "private", "fallback"
    agent_id: Optional[str] = None
    relevance_labels: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """
    The complete retrieval result for a single request.
    """

    request_id: str
    agent_id: str
    role: AgentRole
    candidates: List[RetrievedCandidate]
    shared_candidates: List[RetrievedCandidate] = field(default_factory=list)
    private_candidates: List[RetrievedCandidate] = field(default_factory=list)
    fallback_triggered: bool = False
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Matrix Types for Query Matrix Mechanism
# ============================================================


@dataclass
class QueryMatrix:
    """
    Represents a query matrix Q for a shared retrieval bucket.

    Q ∈ R^(k × n) where:
    - k = query embedding dimension
    - n = number of queries in the bucket
    - Column i = query vector for request i
    """

    request_ids: List[str] = field(default_factory=list)
    agent_ids: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=np.array)
    scope_bucket_key: Optional[ScopeBucketKey] = None

    @property
    def num_queries(self) -> int:
        return len(self.request_ids)

    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[0] if self.embeddings.size else 0


@dataclass
class SharedProjectionResult:
    """
    Result of the shared projection transformation.

    Z = Q^T @ W_final ∈ R^(n × r) where:
    - n = number of queries
    - r = shared subspace dimension
    """

    query_matrix: QueryMatrix
    projected: np.ndarray = field(default_factory=np.array)
    projection_matrix: np.ndarray = field(default_factory=np.array)
    svd_rank: int = 0
    mask_applied: bool = False

    @property
    def num_queries(self) -> int:
        return self.projected.shape[0] if self.projected.size else 0

    @property
    def subspace_dim(self) -> int:
        return self.projected.shape[1] if len(self.projected.shape) > 1 else 0


@dataclass
class SharedCandidatePool:
    """
    The shared candidate pool generated from collaborative retrieval.
    """

    pool_id: str = field(default_factory=lambda: f"pool_{uuid.uuid4().hex[:8]}")
    scope_bucket_key: Optional[ScopeBucketKey] = None
    candidates: List[MemoryEntry] = field(default_factory=list)
    scores: np.ndarray = field(default_factory=np.array)
    assignment_matrix: Optional[np.ndarray] = None  # Which query each candidate belongs to
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Protocols / Interfaces (Structural Typing)
# ============================================================


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding model providers."""

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Encode a list of texts into embeddings."""
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query into an embedding."""
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for memory storage backends."""

    def add(self, memory: MemoryEntry) -> None:
        """Add a memory entry to the store."""
        ...

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        ...

    def search(
        self,
        query_embedding: np.ndarray,
        scope_filter: Optional[List[str]] = None,
        memory_type_filter: Optional[List[MemoryType]] = None,
        policy_filter: Optional[PolicyConstraints] = None,
        top_k: int = 50,
    ) -> List[RetrievedCandidate]:
        """Search for relevant memories."""
        ...

    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        ...

    def list_scopes(self) -> List[str]:
        """List all available scope IDs."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking models."""

    def rerank(
        self,
        query: str,
        candidates: List[MemoryEntry],
        role: AgentRole,
        state: AgentState,
        top_k: int = 20,
    ) -> List[RetrievedCandidate]:
        """Rerank candidates for a specific agent's context."""
        ...


# ============================================================
# Agent Interface
# ============================================================


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    agent_id: str
    role: AgentRole
    name: str = ""
    description: str = ""
    allowed_scopes: List[str] = field(default_factory=list)
    allowed_memory_types: List[MemoryType] = field(default_factory=list)
    policy: PolicyConstraints = field(default_factory=PolicyConstraints)
    system_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    """
    A simple agent representation for CoScope integration.
    For full agent capabilities, use LangChain/LangGraph integration.
    """

    config: AgentConfig
    state: AgentState = field(default_factory=AgentState)

    @property
    def agent_id(self) -> str:
        return self.config.agent_id

    @property
    def role(self) -> AgentRole:
        return self.config.role

    def create_retrieval_request(
        self,
        query: str,
        scope: Optional[ScopeSpec] = None,
        memory_types: Optional[List[MemoryType]] = None,
        state_override: Optional[AgentState] = None,
    ) -> RetrievalRequest:
        """Create a retrieval request from this agent."""
        return RetrievalRequest(
            agent_id=self.agent_id,
            role=self.role,
            query=query,
            scope=scope
            or ScopeSpec(
                private_scopes=[f"agent/{self.agent_id}/private"],
                shared_scopes=self.config.allowed_scopes,
            ),
            memory_types=memory_types or self.config.allowed_memory_types,
            policy=self.config.policy,
            state=state_override or self.state,
        )


# ============================================================
# Type Variables
# ============================================================

T = TypeVar("T")
ConfigT = TypeVar("ConfigT", bound=BaseModel)


# ============================================================
# Policy Constraints Extension
# ============================================================


def policy_constraints_from_dict(data: Dict[str, Any]) -> PolicyConstraints:
    """Create PolicyConstraints from a dictionary."""
    visibility = [
        VisibilityLevel(v) if isinstance(v, str) else v
        for v in data.get("visibility", ["team"])
    ]
    return PolicyConstraints(
        visibility=visibility,
        max_clearance=data.get("max_clearance", 3),
        excluded_zones=data.get("excluded_zones", ["quarantine"]),
        audit_required=data.get("audit_required", False),
    )


# ============================================================
# Pydantic Models for Serialization
# ============================================================


class RetrievalRequestModel(BaseModel):
    """Pydantic model for RetrievalRequest serialization."""

    request_id: str
    agent_id: str
    role: str
    query: str
    scope: Dict[str, Any]
    memory_types: List[str]
    policy: Dict[str, Any]
    state: Dict[str, Any]
    priority: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_request(cls, request: RetrievalRequest) -> RetrievalRequestModel:
        return cls(
            request_id=request.request_id,
            agent_id=request.agent_id,
            role=request.role.value,
            query=request.query,
            scope={
                "private_scopes": request.scope.private_scopes,
                "shared_scopes": request.scope.shared_scopes,
                "workspace_scopes": request.scope.workspace_scopes,
                "governed_scopes": request.scope.governed_scopes,
            },
            memory_types=[t.value for t in request.memory_types],
            policy={
                "visibility": [v.value for v in request.policy.visibility],
                "max_clearance": request.policy.max_clearance,
                "excluded_zones": request.policy.excluded_zones,
                "audit_required": request.policy.audit_required,
            },
            state=request.state.to_dict(),
            priority=request.priority,
            metadata=request.metadata,
        )


class MemoryEntryModel(BaseModel):
    """Pydantic model for MemoryEntry serialization."""

    memory_id: str
    scope_id: str
    memory_type: str
    content: str
    visibility: List[str]
    confidence: float
    salience: float
    ttl: Optional[int] = None
    provenance: Dict[str, Any]
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_entry(cls, entry: MemoryEntry) -> MemoryEntryModel:
        return cls(
            memory_id=entry.memory_id,
            scope_id=entry.scope_id,
            memory_type=entry.memory_type.value,
            content=entry.content,
            visibility=[v.value for v in entry.visibility],
            confidence=entry.confidence,
            salience=entry.salience,
            ttl=entry.ttl,
            provenance={
                "source": entry.provenance.source,
                "parent_event": entry.provenance.parent_event,
                "agent_id": entry.provenance.agent_id,
                "timestamp": entry.provenance.timestamp.isoformat()
                if entry.provenance.timestamp
                else None,
                "version": entry.provenance.version,
            },
            tags=entry.tags,
            metadata=entry.metadata,
            created_at=entry.created_at.isoformat(),
            updated_at=entry.updated_at.isoformat(),
        )


# ============================================================
# Dataset-construction types (GoT graph, Episode, GroundTruth)
# Moved from coscope.data.schema.types — single source of truth.
# ============================================================

from collections import deque


class ReasoningPathType(str, Enum):
    """Reasoning path type. Stage 1 implements GoT only; CoT / ToT are reserved."""

    GOT = "GoT"
    COT = "CoT"
    TOT = "ToT"


class GraphType(str, Enum):
    """GoT graph structural type. Drives subset (S1/S2/S3) expectations."""

    LINEAR = "LINEAR"
    FORK = "FORK"
    MERGE = "MERGE"
    FORK_MERGE = "FORK_MERGE"
    INDEPENDENT = "INDEPENDENT"
    POLICY_ISOLATED = "POLICY_ISOLATED"


class SubsetLabel(str, Enum):
    """
    Experimental subset label.

    S1/S2/S3 are mutually exclusive and determined by rho.
    S4 is orthogonal (policy conflict) represented via Episode.s4_eligible.
    """

    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"


class NodeType(str, Enum):
    """GoT node type."""

    PLANNER = "PLANNER"
    SOLVER = "SOLVER"
    VERIFIER = "VERIFIER"


class EdgeType(str, Enum):
    """GoT edge type."""

    DEPENDS_ON = "DEPENDS_ON"
    AGGREGATES = "AGGREGATES"


class InvalidGraphError(ValueError):
    """Raised when a constructed GoTGraph fails DAG / structural validation."""


@dataclass
class GoTNode:
    """A node in a Graph-of-Thought."""

    node_id: str
    node_type: str
    agent_role: str
    hop_index: Optional[int] = None
    parent_node_ids: List[str] = field(default_factory=list)
    child_node_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "agent_role": self.agent_role,
            "hop_index": self.hop_index,
            "parent_node_ids": list(self.parent_node_ids),
            "child_node_ids": list(self.child_node_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoTNode":
        return cls(
            node_id=data["node_id"],
            node_type=data["node_type"],
            agent_role=data.get("agent_role", data["node_type"].lower()),
            hop_index=data.get("hop_index"),
            parent_node_ids=list(data.get("parent_node_ids", [])),
            child_node_ids=list(data.get("child_node_ids", [])),
        )


@dataclass
class GoTEdge:
    """A directed edge in a Graph-of-Thought."""

    from_node: str
    to_node: str
    edge_type: str = EdgeType.DEPENDS_ON.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "edge_type": self.edge_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoTEdge":
        return cls(
            from_node=data.get("from") or data["from_node"],
            to_node=data.get("to") or data["to_node"],
            edge_type=data.get("edge_type", EdgeType.DEPENDS_ON.value),
        )


@dataclass
class GoTGraph:
    """
    A directed acyclic Graph-of-Thought describing agent reasoning dependencies
    within a single episode.
    """

    nodes: List[GoTNode] = field(default_factory=list)
    edges: List[GoTEdge] = field(default_factory=list)
    graph_type: GraphType = GraphType.LINEAR

    def get_node(self, node_id: str) -> Optional[GoTNode]:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def reasoning_nodes(self) -> List[GoTNode]:
        return [n for n in self.nodes if n.node_type != NodeType.VERIFIER.value]

    def solver_nodes(self) -> List[GoTNode]:
        return [n for n in self.nodes if n.node_type == NodeType.SOLVER.value]

    def planner_node(self) -> Optional[GoTNode]:
        for n in self.nodes:
            if n.node_type == NodeType.PLANNER.value:
                return n
        return None

    def verifier_node(self) -> Optional[GoTNode]:
        for n in self.nodes:
            if n.node_type == NodeType.VERIFIER.value:
                return n
        return None

    def _parent_map(self) -> Dict[str, List[str]]:
        parents: Dict[str, List[str]] = {n.node_id: list(n.parent_node_ids) for n in self.nodes}
        for e in self.edges:
            if e.to_node in parents and e.from_node not in parents[e.to_node]:
                parents[e.to_node].append(e.from_node)
        return parents

    def _child_map(self) -> Dict[str, List[str]]:
        children: Dict[str, List[str]] = {n.node_id: list(n.child_node_ids) for n in self.nodes}
        for e in self.edges:
            if e.from_node in children and e.to_node not in children[e.from_node]:
                children[e.from_node].append(e.to_node)
        return children

    def get_all_ancestors(self, node_id: str) -> List[str]:
        parent_map = self._parent_map()
        if node_id not in parent_map:
            return []
        visited: set = set()
        queue: deque = deque(parent_map[node_id])
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            for p in parent_map.get(cur, []):
                if p not in visited:
                    queue.append(p)
        return list(visited)

    def is_dag(self) -> bool:
        parent_map = self._parent_map()
        child_map = self._child_map()
        indegree: Dict[str, int] = {nid: len(parent_map.get(nid, [])) for nid in parent_map}
        queue: deque = deque([nid for nid, d in indegree.items() if d == 0])
        visited = 0
        while queue:
            cur = queue.popleft()
            visited += 1
            for ch in child_map.get(cur, []):
                indegree[ch] -= 1
                if indegree[ch] == 0:
                    queue.append(ch)
        return visited == len(indegree)

    def validate(self) -> None:
        if not self.is_dag():
            raise InvalidGraphError(f"GoTGraph is not a DAG (graph_type={self.graph_type})")
        parent_map = self._parent_map()
        child_map = self._child_map()
        for n in self.nodes:
            if n.node_type == NodeType.SOLVER.value and not parent_map.get(n.node_id):
                raise InvalidGraphError(f"Solver node '{n.node_id}' has no parents (orphan)")
            if n.node_type == NodeType.VERIFIER.value:
                if parent_map.get(n.node_id) or child_map.get(n.node_id):
                    raise InvalidGraphError(
                        f"Verifier node '{n.node_id}' must be isolated "
                        f"(no in/out edges), got parents="
                        f"{parent_map.get(n.node_id)} children={child_map.get(n.node_id)}"
                    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "graph_type": self.graph_type.value if isinstance(self.graph_type, GraphType) else str(self.graph_type),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoTGraph":
        gt_raw = data.get("graph_type", GraphType.LINEAR.value)
        try:
            gt = GraphType(gt_raw)
        except ValueError:
            gt = GraphType.LINEAR
        return cls(
            nodes=[GoTNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[GoTEdge.from_dict(e) for e in data.get("edges", [])],
            graph_type=gt,
        )


@dataclass
class GroundTruthEvidence:
    """
    A ground-truth memory item required by one or more agents.
    Used by evaluation to check recall.
    """

    memory_id: str
    scope_layer: str
    hop_index: Optional[int]
    required_by_agent_ids: List[str]
    evidence_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "scope_layer": self.scope_layer,
            "hop_index": self.hop_index,
            "required_by_agent_ids": list(self.required_by_agent_ids),
            "evidence_type": self.evidence_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthEvidence":
        return cls(
            memory_id=data["memory_id"],
            scope_layer=data["scope_layer"],
            hop_index=data.get("hop_index"),
            required_by_agent_ids=list(data.get("required_by_agent_ids", [])),
            evidence_type=data.get("evidence_type", "shared_required"),
        )


@dataclass
class Episode:
    """
    A single multi-agent experiment unit.

    agents, memory_entries, and retrieval_requests hold coscope.core.types
    objects directly so an Episode can be fed into the CoScope runtime
    via to_runtime_objects() without format conversion.
    """

    episode_id: str
    dataset: str
    split: str
    original_id: str
    hop_count: int
    graph_type: GraphType
    reasoning_path_type: ReasoningPathType

    rho: float
    rho_subset: SubsetLabel
    policy_conflict: bool
    s4_eligible: bool

    question: str
    answer: str

    got_graph: GoTGraph
    agents: List[Agent] = field(default_factory=list)
    memory_entries: List[MemoryEntry] = field(default_factory=list)
    retrieval_requests: List[RetrievalRequest] = field(default_factory=list)
    ground_truth: List[GroundTruthEvidence] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_runtime_objects(self):
        """Return (agents, memory_entries, retrieval_requests) for runtime use."""
        return self.agents, self.memory_entries, self.retrieval_requests
