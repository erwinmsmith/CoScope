"""CoScope Agent Integrations."""

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
