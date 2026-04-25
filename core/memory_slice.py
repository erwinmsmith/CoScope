"""
Memory Slice definitions for CoScope.

A memory slice represents a specific view of memory with scope, type,
and policy constraints. This is the fundamental unit for retrieval operations.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np

from core.scope import PolicyConstraints, ScopeDefinition, ScopeRegistry
from core.types import (
    MemoryEntry,
    MemoryType,
    PolicyConstraints as PolicyConstraintsType,
    RetrievedCandidate,
    ScopeSpec,
    ScopeType,
    VisibilityLevel,
)

if TYPE_CHECKING:
    from core.types import EmbeddingProvider


# ============================================================
# Memory Slice
# ============================================================


@dataclass
class MemorySlice:
    """
    Represents a specific slice of memory for retrieval.

    A memory slice is defined by:
    - scope: The scope region to search
    - memory_types: Types of memory to include
    - policy: Access control constraints
    - embedding_provider: For embedding-based retrieval

    This is the retrieval target in CoScope's collaborative retrieval pipeline.
    """

    slice_id: str = field(default_factory=lambda: f"slice_{uuid.uuid4().hex[:8]}")
    scope: ScopeSpec = field(default_factory=ScopeSpec)
    memory_types: List[MemoryType] = field(
        default_factory=lambda: [MemoryType.EPISODIC]
    )
    policy: PolicyConstraintsType = field(default_factory=PolicyConstraintsType)
    embedding_provider: Optional["EmbeddingProvider"] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def matches_request(
        self, scope_id: str, memory_type: MemoryType, policy: PolicyConstraintsType
    ) -> bool:
        """Check if this slice matches a retrieval request's constraints."""
        # Check scope
        if scope_id not in self.scope.all_scopes:
            return False

        # Check memory type
        if memory_type not in self.memory_types:
            return False

        # Check policy compatibility
        if not self._policies_compatible(self.policy, policy):
            return False

        return True

    def _policies_compatible(
        self, p1: PolicyConstraintsType, p2: PolicyConstraintsType
    ) -> bool:
        """Check if two policies are compatible."""
        # Visibility must overlap
        if not set(v.value for v in p1.visibility) & set(v.value for v in p2.visibility):
            return False
        return True

    @classmethod
    def from_scope_and_types(
        cls,
        scope: ScopeSpec,
        memory_types: List[MemoryType],
        policy: Optional[PolicyConstraintsType] = None,
    ) -> "MemorySlice":
        """Create a memory slice from scope and memory types."""
        return cls(
            scope=scope,
            memory_types=memory_types,
            policy=policy or PolicyConstraintsType(),
        )

    def to_filter_spec(self) -> Dict[str, Any]:
        """Convert to a filter specification for storage backends."""
        return {
            "scope_ids": self.scope.all_scopes,
            "memory_types": [t.value for t in self.memory_types],
            "visibility": [v.value for v in self.policy.visibility],
            "max_clearance": self.policy.max_clearance,
            "excluded_zones": self.policy.excluded_zones,
        }


# ============================================================
# Memory Slice Index
# ============================================================


@dataclass
class SliceKey:
    """
    A hashable key for memory slices.
    Used for caching and deduplication of slice operations.
    """

    scope_id: str
    memory_types: Tuple[str, ...]
    visibility: Tuple[str, ...]
    max_clearance: int

    @classmethod
    def from_slice(cls, slice: MemorySlice) -> "SliceKey":
        return cls(
            scope_id=slice.scope.primary_scope,
            memory_types=tuple(sorted(t.value for t in slice.memory_types)),
            visibility=tuple(sorted(v.value for v in slice.policy.visibility)),
            max_clearance=slice.policy.max_clearance,
        )

    def __hash__(self) -> int:
        return hash(
            (self.scope_id, self.memory_types, self.visibility, self.max_clearance)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SliceKey):
            return False
        return (
            self.scope_id == other.scope_id
            and self.memory_types == other.memory_types
            and self.visibility == other.visibility
            and self.max_clearance == other.max_clearance
        )


class MemorySliceIndex:
    """
    Index for managing and looking up memory slices.
    Provides efficient slice retrieval for the retrieval pipeline.
    """

    def __init__(self):
        self._slices: Dict[str, MemorySlice] = {}
        self._scope_index: Dict[str, Set[str]] = {}  # scope_id -> slice_ids
        self._type_index: Dict[MemoryType, Set[str]] = {}  # memory_type -> slice_ids
        self._slice_keys: Dict[SliceKey, str] = {}  # For deduplication

    def add(self, slice: MemorySlice) -> str:
        """Add a memory slice to the index."""
        self._slices[slice.slice_id] = slice

        # Update scope index
        for scope_id in slice.scope.all_scopes:
            if scope_id not in self._scope_index:
                self._scope_index[scope_id] = set()
            self._scope_index[scope_id].add(slice.slice_id)

        # Update type index
        for mem_type in slice.memory_types:
            if mem_type not in self._type_index:
                self._type_index[mem_type] = set()
            self._type_index[mem_type].add(slice.slice_id)

        # Update slice key index
        key = SliceKey.from_slice(slice)
        self._slice_keys[key] = slice.slice_id

        return slice.slice_id

    def get(self, slice_id: str) -> Optional[MemorySlice]:
        """Get a slice by ID."""
        return self._slices.get(slice_id)

    def find_matching(
        self,
        scope_id: str,
        memory_type: Optional[MemoryType] = None,
    ) -> List[MemorySlice]:
        """Find all slices matching the given constraints."""
        slice_ids = self._scope_index.get(scope_id, set())

        if memory_type:
            slice_ids &= self._type_index.get(memory_type, set())

        return [self._slices[sid] for sid in slice_ids if sid in self._slices]

    def get_or_create(
        self,
        scope: ScopeSpec,
        memory_types: List[MemoryType],
        policy: Optional[PolicyConstraintsType] = None,
    ) -> MemorySlice:
        """
        Get an existing slice or create a new one.
        Uses SliceKey for deduplication.
        """
        # Create a temporary slice for key lookup
        temp_slice = MemorySlice(
            scope=scope,
            memory_types=memory_types,
            policy=policy or PolicyConstraintsType(),
        )
        key = SliceKey.from_slice(temp_slice)

        if key in self._slice_keys:
            existing_id = self._slice_keys[key]
            return self._slices[existing_id]

        # Create new slice
        new_slice = MemorySlice(
            slice_id=f"slice_{uuid.uuid4().hex[:8]}",
            scope=scope,
            memory_types=memory_types,
            policy=policy or PolicyConstraintsType(),
        )
        self.add(new_slice)
        return new_slice

    def remove(self, slice_id: str) -> bool:
        """Remove a slice from the index."""
        if slice_id not in self._slices:
            return False

        slice = self._slices[slice_id]

        # Remove from scope index
        for scope_id in slice.scope.all_scopes:
            if scope_id in self._scope_index:
                self._scope_index[scope_id].discard(slice_id)

        # Remove from type index
        for mem_type in slice.memory_types:
            if mem_type in self._type_index:
                self._type_index[mem_type].discard(slice_id)

        # Remove from slice key index
        key = SliceKey.from_slice(slice)
        self._slice_keys.pop(key, None)

        del self._slices[slice_id]
        return True

    def list_all(self) -> List[MemorySlice]:
        """List all slices in the index."""
        return list(self._slices.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the slice index."""
        return {
            "total_slices": len(self._slices),
            "scopes_covered": len(self._scope_index),
            "memory_types_covered": len(self._type_index),
        }


# ============================================================
# Memory Slice Builder (Fluent API)
# ============================================================


class MemorySliceBuilder:
    """
    Builder for creating MemorySlice instances with a fluent API.
    """

    def __init__(self):
        self._slice_id: Optional[str] = None
        self._scope: Optional[ScopeSpec] = None
        self._memory_types: List[MemoryType] = [MemoryType.EPISODIC]
        self._policy: Optional[PolicyConstraintsType] = None
        self._metadata: Dict[str, Any] = {}

    def with_id(self, slice_id: str) -> "MemorySliceBuilder":
        """Set a custom slice ID."""
        self._slice_id = slice_id
        return self

    def with_scope(self, scope: ScopeSpec) -> "MemorySliceBuilder":
        """Set the scope specification."""
        self._scope = scope
        return self

    def with_scope_id(self, scope_id: str) -> "MemorySliceBuilder":
        """Set a simple scope by ID."""
        self._scope = ScopeSpec(shared_scopes=[scope_id])
        return self

    def with_memory_types(
        self, *types: MemoryType
    ) -> "MemorySliceBuilder":
        """Set the memory types to include."""
        self._memory_types = list(types)
        return self

    def add_memory_type(self, mem_type: MemoryType) -> "MemorySliceBuilder":
        """Add a memory type."""
        if mem_type not in self._memory_types:
            self._memory_types.append(mem_type)
        return self

    def with_policy(
        self, policy: PolicyConstraintsType
    ) -> "MemorySliceBuilder":
        """Set the policy constraints."""
        self._policy = policy
        return self

    def with_visibility(
        self, *visibility: VisibilityLevel
    ) -> "MemorySliceBuilder":
        """Set visibility levels."""
        if self._policy is None:
            self._policy = PolicyConstraintsType()
        self._policy.visibility = list(visibility)
        return self

    def with_max_clearance(self, level: int) -> "MemorySliceBuilder":
        """Set maximum clearance level."""
        if self._policy is None:
            self._policy = PolicyConstraintsType()
        self._policy.max_clearance = level
        return self

    def exclude_zones(self, *zones: str) -> "MemorySliceBuilder":
        """Exclude specific zones."""
        if self._policy is None:
            self._policy = PolicyConstraintsType()
        self._policy.excluded_zones = list(zones)
        return self

    def with_metadata(self, **kwargs: Any) -> "MemorySliceBuilder":
        """Add metadata."""
        self._metadata.update(kwargs)
        return self

    def build(self) -> MemorySlice:
        """Build the memory slice."""
        if self._scope is None:
            raise ValueError("Scope must be set before building")

        return MemorySlice(
            slice_id=self._slice_id or f"slice_{uuid.uuid4().hex[:8]}",
            scope=self._scope,
            memory_types=self._memory_types,
            policy=self._policy or PolicyConstraintsType(),
            metadata=self._metadata,
        )
