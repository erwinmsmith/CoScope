"""CoScope Agent Integrations."""

from coscope.agents.planner_builder import PlannerAgentBuilder
from coscope.agents.solver_builder import SolverAgentBuilder
from coscope.agents.verifier_builder import VerifierAgentBuilder
from coscope.agents.langchain_agent import (
    CoScopeRetrievalTool,
    CoScopeLangChainTool,
    CoScopeEmbeddingsAdapter,
    create_langchain_agent,
    is_langchain_available,
)
from coscope.agents.langgraph_agent import (
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
