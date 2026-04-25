"""
Memory CRUD types and filters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.types import MemoryType, VisibilityLevel

logger = logging.getLogger(__name__)


@dataclass
class MemoryFilter:
    """Filter criteria for memory queries."""

    scope_ids: Optional[List[str]] = None
    memory_types: Optional[List["MemoryType"]] = None
    visibility: Optional[List["VisibilityLevel"]] = None
    tags: Optional[List[str]] = None
    min_confidence: Optional[float] = None
    max_age_seconds: Optional[int] = None
    source: Optional[str] = None
    agent_id: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def matches(self, memory: Any) -> bool:
        """Check if a memory matches this filter."""
        # Check scope
        if self.scope_ids and memory.scope_id not in self.scope_ids:
            return False

        # Check memory type
        if self.memory_types and memory.memory_type not in self.memory_types:
            return False

        # Check visibility
        if self.visibility:
            if not any(v in memory.visibility for v in self.visibility):
                return False

        # Check tags
        if self.tags:
            if not any(tag in memory.tags for tag in self.tags):
                return False

        # Check confidence
        if self.min_confidence is not None:
            if memory.confidence < self.min_confidence:
                return False

        # Check age
        if self.max_age_seconds is not None:
            age = (datetime.utcnow() - memory.created_at).total_seconds()
            if age > self.max_age_seconds:
                return False

        # Check source
        if self.source:
            if self.source not in memory.provenance.source:
                return False

        # Check agent
        if self.agent_id:
            if memory.provenance.agent_id != self.agent_id:
                return False

        # Check custom fields
        for key, value in self.custom.items():
            if memory.metadata.get(key) != value:
                return False

        return True


@dataclass
class MemoryQuery:
    """Query specification for memory operations."""

    text: str = ""
    filter: Optional[MemoryFilter] = None
    limit: int = 20
    offset: int = 0
    sort_by: str = "score"  # score, created_at, confidence, relevance
    sort_order: str = "desc"  # asc, desc

    @classmethod
    def simple(
        cls,
        text: str,
        scope: Optional[str] = None,
        memory_type: Optional["MemoryType"] = None,
        limit: int = 20,
    ) -> "MemoryQuery":
        """Create a simple query."""
        filters = []
        if scope:
            filters.append(scope)
        if memory_type:
            filters.append(memory_type)

        return cls(
            text=text,
            filter=MemoryFilter(
                scope_ids=filters if filters else None,
            ),
            limit=limit,
        )
