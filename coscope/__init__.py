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
from coscope.construction.episode_builder import EpisodeBuilder

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
from coscope.evaluation import (
    EvaluationReport,
    SyntheticCase,
    SyntheticVariantSummary,
    VariantRun,
    build_synthetic_suite,
    evaluate_synthetic_suite,
    evaluate_retrieval,
    evaluate_variants,
    format_synthetic_case_tables,
    format_synthetic_table,
    format_variant_table,
)

__all__ = [
    # Version
    "__version__",
    # Engine
    "CoScope",
    "create_coscope",
    "build_episode",
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
    # Evaluation
    "EvaluationReport",
    "SyntheticCase",
    "SyntheticVariantSummary",
    "VariantRun",
    "build_synthetic_suite",
    "evaluate_synthetic_suite",
    "evaluate_retrieval",
    "evaluate_variants",
    "format_synthetic_case_tables",
    "format_synthetic_table",
    "format_variant_table",
]


def build_episode(
    raw_item,
    *,
    dataset: str,
    split: str,
    target_graph_type,
    seed: int = 42,
    reasoning_path_type: str = "got",
):
    """
    Convenience wrapper used by examples / smoke tests.

    Keeps a tiny public API for one-off structural validation without forcing
    callers to instantiate EpisodeBuilder directly.
    """
    builder = EpisodeBuilder(reasoning_path_type=reasoning_path_type)
    return builder.build_episode(
        raw_item=raw_item,
        dataset=dataset,
        split=split,
        target_graph_type=target_graph_type,
        seed=seed,
    )
