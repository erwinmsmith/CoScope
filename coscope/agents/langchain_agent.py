"""
LangChain Agent Integration for CoScope.

Provides LangChain-compatible agent and tool interfaces.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TYPE_CHECKING,
)

from coscope.core.scope import PolicyConstraints, ScopeSpec
from coscope.core.types import (
    Agent,
    AgentConfig,
    AgentRole,
    AgentState,
    MemoryEntry,
    MemoryStore,
    MemoryType,
    RetrievedCandidate,
    RetrievalRequest,
    RetrievalResult,
    ScopeSpec,
    VisibilityLevel,
)

if TYPE_CHECKING:
    from coscope.retrieval.pipeline import RetrievalPipeline
    from coscope.memory.store import MemoryManager

logger = logging.getLogger(__name__)

# ============================================================
# LangChain Integration (Optional Imports)
# ============================================================

_langchain_available = False
_langgraph_available = False

try:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.embeddings import Embeddings
    from langchain_core.tools import BaseTool
    from langchain_core.callbacks import CallbackManagerForToolRun
    from langchain_core.runnables import RunnableConfig
    _langchain_available = True
except ImportError:
    logger.debug("LangChain not available. Install with: pip install langchain")


# ============================================================
# CoScope Tool for LangChain
# ============================================================


@dataclass
class ToolInput:
    """Input schema for the CoScope retrieval tool."""
    query: str
    scope_ids: Optional[List[str]] = None
    memory_types: Optional[List[str]] = None
    top_k: int = 20


@dataclass
class ToolOutput:
    """Output schema for the CoScope retrieval tool."""
    results: List[Dict[str, Any]]
    num_results: int
    metadata: Dict[str, Any]


class CoScopeRetrievalTool:
    """
    A retrieval tool that uses CoScope's collaborative memory retrieval.

    This tool can be used as a LangChain tool or standalone.
    """

    name: str = "coscope_memory_retrieval"
    description: str = """
    Retrieves relevant memories from the collaborative memory system.
    Use this when you need to recall past events, facts, or artifacts
    that may be relevant to the current task.

    Input:
    - query: The search query (required)
    - scope_ids: Optional list of scope IDs to search in
    - memory_types: Optional list of memory types (episodic, semantic, artifact, shared, working)
    - top_k: Number of results to return (default 20)
    """

    def __init__(
        self,
        pipeline: "RetrievalPipeline",
        memory_manager: "MemoryManager",
        agent_config: AgentConfig,
    ):
        self.pipeline = pipeline
        self.memory_manager = memory_manager
        self.agent_config = agent_config

    def invoke(
        self,
        query: str,
        scope_ids: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> ToolOutput:
        """
        Invoke the retrieval tool.

        Args:
            query: Search query
            scope_ids: Optional scope filters
            memory_types: Optional memory type filters
            top_k: Number of results

        Returns:
            ToolOutput with results
        """
        # Create a retrieval request
        scope_spec = ScopeSpec(
            shared_scopes=scope_ids or self.agent_config.allowed_scopes,
        )

        memory_type_list = [
            MemoryType(t) for t in (memory_types or self.agent_config.allowed_memory_types)
        ]

        request = RetrievalRequest(
            agent_id=self.agent_config.agent_id,
            role=self.agent_config.role,
            query=query,
            scope=scope_spec,
            memory_types=memory_type_list,
            policy=self.agent_config.policy,
            state=AgentState(),
        )

        # Execute retrieval
        results = self.pipeline.retrieve([request])

        if not results:
            return ToolOutput(
                results=[],
                num_results=0,
                metadata={},
            )

        result = results[0]

        # Format output
        formatted_results = []
        for candidate in result.candidates[:top_k]:
            formatted_results.append({
                "memory_id": candidate.memory.memory_id,
                "content": candidate.memory.content,
                "scope": candidate.memory.scope_id,
                "type": candidate.memory.memory_type.value,
                "score": candidate.score,
                "source": candidate.source,
                "provenance": {
                    "source": candidate.memory.provenance.source,
                    "agent_id": candidate.memory.provenance.agent_id,
                },
            })

        return ToolOutput(
            results=formatted_results,
            num_results=len(formatted_results),
            metadata={
                "request_id": result.request_id,
                "agent_id": result.agent_id,
                "role": result.role.value,
                "fallback_triggered": result.fallback_triggered,
            },
        )

    def __call__(
        self,
        query: str,
        scope_ids: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> ToolOutput:
        """Convenience method for tool invocation."""
        return self.invoke(
            query=query,
            scope_ids=scope_ids,
            memory_types=memory_types,
            top_k=top_k,
        )


# ============================================================
# LangChain Tool Adapter
# ============================================================


class CoScopeLangChainTool:
    """
    LangChain-compatible wrapper for CoScope retrieval.

    This class adapts CoScopeRetrievalTool to the LangChain tool interface.
    """

    def __init__(
        self,
        pipeline: "RetrievalPipeline",
        memory_manager: "MemoryManager",
        agent_config: AgentConfig,
    ):
        self._tool = CoScopeRetrievalTool(
            pipeline=pipeline,
            memory_manager=memory_manager,
            agent_config=agent_config,
        )
        self.name = self._tool.name
        self.description = self._tool.description

    def run(
        self,
        tool_input: str,
        run_manager: Optional["CallbackManagerForToolRun"] = None,
    ) -> str:
        """
        Run the tool with string input (LangChain interface).

        Args:
            tool_input: JSON string with tool arguments
            run_manager: Optional callback manager

        Returns:
            Formatted result string
        """
        import json

        try:
            args = json.loads(tool_input)
        except json.JSONDecodeError:
            # Assume it's just a query string
            args = {"query": tool_input}

        result = self._tool.invoke(**args)

        # Format as readable string
        lines = [f"Found {result.num_results} relevant memories:\n"]

        for i, item in enumerate(result.results, 1):
            lines.append(f"{i}. [{item['type']}] Score: {item['score']:.3f}")
            lines.append(f"   Scope: {item['scope']}")
            lines.append(f"   Content: {item['content'][:200]}...")
            lines.append("")

        return "\n".join(lines)


# ============================================================
# CoScope Embeddings Adapter
# ============================================================


class CoScopeEmbeddingsAdapter:
    """
    Adapter that makes CoScope's embedding provider compatible with LangChain.

    This allows CoScope to be used with LangChain's embedding-dependent features.
    """

    def __init__(self, embedding_provider):
        self._provider = embedding_provider

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query (LangChain interface)."""
        embedding = self._provider.embed_query(text)
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents (LangChain interface)."""
        embeddings = self._provider.embed_texts(texts)
        return [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings]

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed query."""
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed documents."""
        return self.embed_documents(texts)


# ============================================================
# Agent Factory
# ============================================================


def create_langchain_agent(
    pipeline: "RetrievalPipeline",
    memory_manager: "MemoryManager",
    agent_config: AgentConfig,
) -> Any:
    """
    Create a LangChain-compatible agent with CoScope retrieval.

    Args:
        pipeline: CoScope retrieval pipeline
        memory_manager: Memory manager
        agent_config: Agent configuration

    Returns:
        A LangChain agent with CoScope as a tool
    """
    if not _langchain_available:
        raise ImportError(
            "LangChain is not installed. Install with: pip install langchain"
        )

    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from coscope.config.settings import get_config

    # Create the retrieval tool
    retrieval_tool = CoScopeRetrievalTool(
        pipeline=pipeline,
        memory_manager=memory_manager,
        agent_config=agent_config,
    )

    # Create LangChain tool
    lc_tool = retrieval_tool

    # Create prompt
    prompt = PromptTemplate.from_template("""
    You are a {role} agent named {agent_id}.

    Role description: {description}

    You have access to the following tools:
    - coscope_memory_retrieval: Search the collaborative memory system

    When you need information from memory, use the coscope_memory_retrieval tool.

    Current task: {input}

    Think step by step and use tools as needed.
    """)

    # Create LLM using configured provider (SiliconFlow or OpenAI, both OpenAI-compatible)
    cfg = get_config()
    llm_cfg = cfg.llm
    if llm_cfg.provider == "siliconflow":
        provider_settings = llm_cfg.siliconflow
        api_key = (
            os.getenv("COSCOPE_SILICONFLOW_API_KEY")
            or os.getenv("COSCOPE_OPENAI_API_KEY")
        )
    else:
        provider_settings = llm_cfg.openai
        api_key = (
            os.getenv("COSCOPE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
    llm = ChatOpenAI(
        model=provider_settings["model"],
        base_url=provider_settings["api_base"],
        api_key=api_key,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
        timeout=llm_cfg.timeout,
    )

    # Create agent
    agent = create_react_agent(
        llm=llm,
        tools=[lc_tool],
        prompt=prompt,
    )

    # Create executor
    executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=[lc_tool],
        verbose=True,
    )

    return executor


# ============================================================
# Utility Functions
# ============================================================


def is_langchain_available() -> bool:
    """Check if LangChain is available."""
    return _langchain_available


def get_langchain_version() -> Optional[str]:
    """Get the installed LangChain version."""
    if not _langchain_available:
        return None
    try:
        import langchain
        return langchain.__version__
    except Exception:
        return "unknown"
