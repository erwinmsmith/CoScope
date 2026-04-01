"""
CoScope - Collaborative Memory Retrieval Framework for Multi-Agent Systems.

CoScope provides efficient collaborative memory retrieval for multi-agent
systems, enabling shared first-stage retrieval with personalized reranking
and private fallback mechanisms.

Example usage:
    from coscope import CoScope

    coscope = CoScope()
    coscope.create_agent(agent_id="planner_1", role="planner")
    coscope.add_memory(content="...", scope_id="task/shared")
    results = coscope.retrieve([request])
"""

__version__ = "0.1.0"

from coscope.engine import CoScope, create_coscope
from coscope.core.types import (
    Agent,
    AgentConfig,
    AgentRole,
    AgentState,
    MemoryEntry,
    MemoryType,
    MemoryStore,
    PolicyConstraints,
    RetrievalRequest,
    RetrievalResult,
    RetrievedCandidate,
    ScopeSpec,
    ScopeType,
    VisibilityLevel,
)
from coscope.memory.store import MemoryManager, InMemoryMemoryStore
from coscope.config.settings import CoScopeConfig, get_config

__all__ = [
    # Version
    "__version__",
    # Engine
    "CoScope",
    "create_coscope",
    # Core types
    "Agent",
    "AgentConfig",
    "AgentRole",
    "AgentState",
    "MemoryEntry",
    "MemoryType",
    "MemoryStore",
    "PolicyConstraints",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievedCandidate",
    "ScopeSpec",
    "ScopeType",
    "VisibilityLevel",
    # Memory
    "MemoryManager",
    "InMemoryMemoryStore",
    # Config
    "CoScopeConfig",
    "get_config",
]
