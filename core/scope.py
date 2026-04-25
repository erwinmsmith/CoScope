"""
Memory Scope definitions for CoScope.

Defines the scope hierarchy and scope management for multi-agent
memory access control.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
)

from core.types import (
    MemoryType,
    PolicyConstraints,
    ScopeBucketKey,
    ScopeSpec,
    ScopeType,
    VisibilityLevel,
)


# ============================================================
# Scope Overlap Detection
# ============================================================


class ScopeOverlapType(str, Enum):
    """Types of scope overlap between requests."""

    IDENTICAL = "identical"  # Same primary scope
    SUBSET = "subset"  # One scope is a subset of another
    INTERSECTING = "intersecting"  # Scopes overlap partially
    DISJOINT = "disjoint"  # No overlap


@dataclass
class ScopeOverlap:
    """
    Represents the overlap between two requests' scopes.
    Used to determine if collaborative retrieval is possible.
    """

    overlap_type: ScopeOverlapType
    shared_scopes: List[str] = field(default_factory=list)
    private_scopes_1: List[str] = field(default_factory=list)
    private_scopes_2: List[str] = field(default_factory=list)
    score: float = 0.0  # Overlap score [0, 1]

    @property
    def can_share_retrieval(self) -> bool:
        """Whether these scopes allow shared first-stage retrieval."""
        return (
            self.overlap_type == ScopeOverlapType.IDENTICAL
            or self.overlap_type == ScopeOverlapType.SUBSET
            or (self.overlap_type == ScopeOverlapType.INTERSECTING and self.score > 0.5)
        )


class ScopeOverlapDetector:
    """
    Detects scope overlap between retrieval requests.
    Implements the scope overlap detection rules from the design doc.
    """

    # Priority order for determining primary shared scope
    SCOPE_PRIORITY = [
        ScopeType.TASK_SHARED,
        ScopeType.SESSION,
        ScopeType.WORKSPACE,
        ScopeType.GOVERNED,
        ScopeType.AGENT_PRIVATE,
    ]

    def detect_overlap(
        self,
        spec1: ScopeSpec,
        spec2: ScopeSpec,
    ) -> ScopeOverlap:
        """
        Detect the overlap between two scope specifications.

        Returns a ScopeOverlap object describing the overlap type and shared scopes.
        """
        # Get shared scopes
        shared_scopes = self._find_shared_scopes(spec1, spec2)

        if not shared_scopes:
            return ScopeOverlap(
                overlap_type=ScopeOverlapType.DISJOINT,
                score=0.0,
            )

        # Determine overlap type
        if self._is_identical(spec1, spec2, shared_scopes):
            overlap_type = ScopeOverlapType.IDENTICAL
            score = 1.0
        elif self._is_subset(spec1, spec2, shared_scopes):
            overlap_type = ScopeOverlapType.SUBSET
            score = self._calculate_subset_score(spec1, spec2, shared_scopes)
        else:
            overlap_type = ScopeOverlapType.INTERSECTING
            score = self._calculate_intersection_score(
                spec1, spec2, shared_scopes
            )

        return ScopeOverlap(
            overlap_type=overlap_type,
            shared_scopes=shared_scopes,
            private_scopes_1=[
                s for s in spec1.all_scopes if s not in shared_scopes
            ],
            private_scopes_2=[
                s for s in spec2.all_scopes if s not in shared_scopes
            ],
            score=score,
        )

    def _find_shared_scopes(
        self, spec1: ScopeSpec, spec2: ScopeSpec
    ) -> List[str]:
        """Find all scopes shared between two scope specs."""
        all1 = set(spec1.all_scopes)
        all2 = set(spec2.all_scopes)
        return list(all1 & all2)

    def _is_identical(
        self, spec1: ScopeSpec, spec2: ScopeSpec, shared: List[str]
    ) -> bool:
        """Check if two scope specs are identical."""
        return set(spec1.all_scopes) == set(spec2.all_scopes)

    def _is_subset(
        self, spec1: ScopeSpec, spec2: ScopeSpec, shared: List[str]
    ) -> bool:
        """Check if spec1 is a subset of spec2 or vice versa."""
        return (
            set(spec1.all_scopes).issubset(set(spec2.all_scopes))
            or set(spec2.all_scopes).issubset(set(spec1.all_scopes))
        )

    def _calculate_subset_score(
        self, spec1: ScopeSpec, spec2: ScopeSpec, shared: List[str]
    ) -> float:
        """Calculate overlap score for subset relationship."""
        smaller = min(len(spec1.all_scopes), len(spec2.all_scopes))
        larger = max(len(spec1.all_scopes), len(spec2.all_scopes))
        return len(shared) / larger if larger > 0 else 0.0

    def _calculate_intersection_score(
        self, spec1: ScopeSpec, spec2: ScopeSpec, shared: List[str]
    ) -> float:
        """Calculate overlap score for intersecting relationship."""
        union_size = len(set(spec1.all_scopes) | set(spec2.all_scopes))
        return len(shared) / union_size if union_size > 0 else 0.0


# ============================================================
# Scope Registry
# ============================================================


@dataclass
class ScopeDefinition:
    """
    Definition of a memory scope in the system.
    """

    scope_id: str
    name: str
    description: str = ""
    scope_type: ScopeType = ScopeType.TASK_SHARED
    parent_scope_id: Optional[str] = None
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    workspace_id: Optional[str] = None
    default_policy: PolicyConstraints = field(
        default_factory=lambda: PolicyConstraints(
            visibility=[VisibilityLevel.TEAM],
            max_clearance=3,
        )
    )
    max_clearance: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def matches_pattern(self, pattern: str) -> bool:
        """Check if this scope matches a pattern (supports wildcards)."""
        if "*" in pattern:
            import fnmatch

            return fnmatch.fnmatch(self.scope_id, pattern)
        return self.scope_id == pattern

    def is_parent_of(self, other: "ScopeDefinition") -> bool:
        """Check if this scope is a parent of another scope."""
        return other.parent_scope_id == self.scope_id

    def is_child_of(self, other: "ScopeDefinition") -> bool:
        """Check if this scope is a child of another scope."""
        return self.parent_scope_id == other.scope_id

    @classmethod
    def from_template(
        cls,
        template_id: str,
        scope_type: ScopeType,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> "ScopeDefinition":
        """Create a scope definition from a template."""
        # Build scope_id from template
        if scope_type == ScopeType.AGENT_PRIVATE and agent_id:
            scope_id = f"agent/{agent_id}/private"
            name = f"Private Memory - {agent_id}"
        elif scope_type == ScopeType.TASK_SHARED and task_id:
            scope_id = f"task/{task_id}/shared"
            name = f"Task Shared - {task_id}"
        elif scope_type == ScopeType.SESSION and session_id:
            scope_id = f"session/{session_id}"
            name = f"Session - {session_id}"
        elif scope_type == ScopeType.WORKSPACE and workspace_id:
            scope_id = f"workspace/{workspace_id}"
            name = f"Workspace - {workspace_id}"
        elif scope_type == ScopeType.GOVERNED:
            scope_id = f"governed/{template_id}"
            name = f"Governed - {template_id}"
        else:
            scope_id = template_id
            name = template_id

        return cls(
            scope_id=scope_id,
            name=name,
            scope_type=scope_type,
            task_id=task_id,
            session_id=session_id,
            workspace_id=workspace_id,
        )


class ScopeRegistry:
    """
    Registry for managing all memory scopes in the system.
    Provides lookup, creation, and validation of scopes.
    """

    def __init__(self):
        self._scopes: Dict[str, ScopeDefinition] = {}
        self._children: Dict[str, Set[str]] = {}
        self._type_index: Dict[ScopeType, Set[str]] = {}

    def register(self, scope: ScopeDefinition) -> None:
        """Register a new scope."""
        self._scopes[scope.scope_id] = scope

        # Update type index
        if scope.scope_type not in self._type_index:
            self._type_index[scope.scope_type] = set()
        self._type_index[scope.scope_type].add(scope.scope_id)

        # Update children index
        if scope.parent_scope_id:
            if scope.parent_scope_id not in self._children:
                self._children[scope.parent_scope_id] = set()
            self._children[scope.parent_scope_id].add(scope.scope_id)

    def get(self, scope_id: str) -> Optional[ScopeDefinition]:
        """Get a scope by ID."""
        return self._scopes.get(scope_id)

    def get_by_type(self, scope_type: ScopeType) -> List[ScopeDefinition]:
        """Get all scopes of a given type."""
        scope_ids = self._type_index.get(scope_type, set())
        return [self._scopes[sid] for sid in scope_ids if sid in self._scopes]

    def get_children(self, scope_id: str) -> List[ScopeDefinition]:
        """Get all direct children of a scope."""
        child_ids = self._children.get(scope_id, set())
        return [self._scopes[sid] for sid in child_ids if sid in self._scopes]

    def get_descendants(self, scope_id: str) -> List[ScopeDefinition]:
        """Get all descendants of a scope (children, grandchildren, etc.)."""
        result = []
        to_visit = list(self.get_children(scope_id))
        while to_visit:
            child = to_visit.pop()
            result.append(child)
            to_visit.extend(self.get_children(child.scope_id))
        return result

    def list_all(self) -> List[ScopeDefinition]:
        """List all registered scopes."""
        return list(self._scopes.values())

    def exists(self, scope_id: str) -> bool:
        """Check if a scope exists."""
        return scope_id in self._scopes

    def unregister(self, scope_id: str) -> bool:
        """Unregister a scope."""
        if scope_id not in self._scopes:
            return False

        scope = self._scopes[scope_id]

        # Remove from type index
        if scope.scope_type in self._type_index:
            self._type_index[scope.scope_type].discard(scope_id)

        # Remove from children index
        for children in self._children.values():
            children.discard(scope_id)
        if scope_id in self._children:
            del self._children[scope_id]

        del self._scopes[scope_id]
        return True

    def find_matching_scopes(self, pattern: str) -> List[ScopeDefinition]:
        """Find all scopes matching a pattern."""
        return [s for s in self._scopes.values() if s.matches_pattern(pattern)]

    def create_from_spec(
        self, spec: ScopeSpec, task_id: Optional[str] = None,
        session_id: Optional[str] = None, workspace_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[ScopeDefinition]:
        """Create and register scopes from a ScopeSpec."""
        created = []

        for scope_id in spec.private_scopes:
            if not self.exists(scope_id):
                scope = ScopeDefinition.from_template(
                    template_id=scope_id,
                    scope_type=ScopeType.AGENT_PRIVATE,
                    agent_id=agent_id or scope_id.split("/")[1] if "/" in scope_id else None,
                )
                self.register(scope)
                created.append(scope)

        for scope_id in spec.shared_scopes:
            if not self.exists(scope_id):
                scope = ScopeDefinition.from_template(
                    template_id=scope_id,
                    scope_type=ScopeType.TASK_SHARED,
                    task_id=task_id,
                )
                self.register(scope)
                created.append(scope)

        for scope_id in spec.workspace_scopes:
            if not self.exists(scope_id):
                scope = ScopeDefinition.from_template(
                    template_id=scope_id,
                    scope_type=ScopeType.WORKSPACE,
                    workspace_id=workspace_id,
                )
                self.register(scope)
                created.append(scope)

        for scope_id in spec.governed_scopes:
            if not self.exists(scope_id):
                scope = ScopeDefinition.from_template(
                    template_id=scope_id,
                    scope_type=ScopeType.GOVERNED,
                )
                self.register(scope)
                created.append(scope)

        return created


# ============================================================
# Policy Compatibility Checker
# ============================================================


class PolicyCompatibilityChecker:
    """
    Checks if policy constraints are compatible for shared retrieval.
    Implements the policy compatibility rules from the design doc.
    """

    def are_compatible(
        self, policy1: PolicyConstraints, policy2: PolicyConstraints
    ) -> bool:
        """
        Check if two policies are compatible for shared retrieval.

        Returns True if:
        1. Visibility levels overlap
        2. Clearance levels are compatible
        3. Excluded zones don't conflict
        """
        # Visibility must overlap
        if not set(policy1.visibility) & set(policy2.visibility):
            return False

        if policy1.max_clearance != policy2.max_clearance:
            return False

        if set(policy1.excluded_zones) != set(policy2.excluded_zones):
            return False

        if policy1.audit_required != policy2.audit_required:
            return False

        return True

    def merge_policies(
        self, policies: List[PolicyConstraints]
    ) -> PolicyConstraints:
        """
        Merge multiple policies into a common policy for shared retrieval.
        The merged policy is the most restrictive common denominator.
        """
        if not policies:
            return PolicyConstraints()

        if len(policies) == 1:
            return policies[0]

        # Intersect visibility levels
        visibility = set(policies[0].visibility)
        for p in policies[1:]:
            visibility &= set(p.visibility)

        # Take minimum clearance (most restrictive)
        max_clearance = min(p.max_clearance for p in policies)

        # Union of excluded zones
        excluded = set()
        for p in policies:
            excluded |= set(p.excluded_zones)

        # Any audit required means audit required
        audit_required = any(p.audit_required for p in policies)

        return PolicyConstraints(
            visibility=list(visibility) if visibility else [VisibilityLevel.TEAM],
            max_clearance=max_clearance,
            excluded_zones=list(excluded),
            audit_required=audit_required,
        )


# ============================================================
# Global Scope Registry Instance
# ============================================================


_global_registry: Optional[ScopeRegistry] = None


def get_scope_registry() -> ScopeRegistry:
    """Get the global scope registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ScopeRegistry()
    return _global_registry


def reset_scope_registry() -> None:
    """Reset the global scope registry (useful for testing)."""
    global _global_registry
    _global_registry = None
