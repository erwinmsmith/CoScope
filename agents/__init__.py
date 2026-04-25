"""CoScope Agent Integrations."""

from agents.planner_builder import PlannerAgentBuilder
from agents.solver_builder import SolverAgentBuilder
from agents.verifier_builder import VerifierAgentBuilder
from agents.langchain_agent import (
    CoScopeRetrievalTool,
    CoScopeLangChainTool,
    CoScopeEmbeddingsAdapter,
    create_langchain_agent,
    is_langchain_available,
)
from agents.langgraph_agent import (
    CoScopeGraphNodes,
    CoScopeGraphBuilder,
    MultiAgentCoordinator,
    create_default_checkpointer,
    is_langgraph_available,
)

__all__ = [
    # builders
    "PlannerAgentBuilder",
    "SolverAgentBuilder",
    "VerifierAgentBuilder",
    # LangChain
    "CoScopeRetrievalTool",
    "CoScopeLangChainTool",
    "CoScopeEmbeddingsAdapter",
    "create_langchain_agent",
    "is_langchain_available",
    # LangGraph
    "CoScopeGraphNodes",
    "CoScopeGraphBuilder",
    "MultiAgentCoordinator",
    "create_default_checkpointer",
    "is_langgraph_available",
]
