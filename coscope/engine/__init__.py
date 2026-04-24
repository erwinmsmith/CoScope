"""
CoScope Engine - Main Entry Point.

The CoScope engine provides a unified interface for collaborative memory
retrieval in multi-agent systems.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TYPE_CHECKING,
)

from coscope.config.settings import CoScopeConfig, get_config, ConfigLoader
from coscope.core.scope import PolicyConstraints, ScopeRegistry, ScopeSpec
from coscope.core.types import (
    Agent,
    AgentConfig,
    AgentRole,
    AgentState,
    EmbeddingProvider,
    MemoryEntry,
    MemoryStore,
    MemoryType,
    PolicyConstraints,
    RetrievalRequest,
    RetrievalResult,
    ScopeSpec,
    VisibilityLevel,
)
from coscope.memory.store import (
    InMemoryMemoryStore,
    MemoryManager,
    create_memory_store,
)
from coscope.retrieval.pipeline import (
    PipelineConfig,
    RetrievalPipeline,
    create_pipeline,
)

if TYPE_CHECKING:
    from coscope.agents.langchain_agent import CoScopeRetrievalTool

logger = logging.getLogger(__name__)


# ============================================================
# Embedding Provider Implementations
# ============================================================


class RandomEmbeddingProvider:
    """
    Simple random embedding provider for testing and development.

    In production, use a proper embedding model.
    """

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def embed_query(self, query: str) -> "np.ndarray":
        """Generate a random embedding for a query."""
        import numpy as np
        import hashlib

        # Deterministic based on query hash
        digest = hashlib.md5(query.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.dimension).astype("float32")
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def embed_texts(self, texts: List[str]) -> List["np.ndarray"]:
        """Generate random embeddings for multiple texts."""
        return [self.embed_query(text) for text in texts]


class OpenAIEmbeddingProvider:
    """
    OpenAI embedding provider.

    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.model = model
        self.api_key = (
            api_key
            or os.getenv("COSCOPE_OPENAI_API_KEY")
            or os.getenv("COSCOPE_SILICONFLOW_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.api_base = (
            api_base
            or os.getenv("COSCOPE_OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        )
        self.dimension = dimension or self._get_dimension()

        # Lazy load OpenAI
        self._client = None

    def _get_dimension(self) -> int:
        """Get embedding dimension for the model."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model, 1536)

    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        return self._client

    def embed_query(self, query: str) -> "np.ndarray":
        """Embed a query using OpenAI."""
        import numpy as np

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=query,
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype="float32")
        except ImportError:
            # Fallback to random embedding
            return RandomEmbeddingProvider(dimension=self.dimension).embed_query(query)

    def embed_texts(self, texts: List[str]) -> List["np.ndarray"]:
        """Embed multiple texts using OpenAI."""
        import numpy as np

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [np.array(item.embedding, dtype="float32") for item in response.data]
        except ImportError:
            # Fallback to random embeddings with matching dimension
            fallback = RandomEmbeddingProvider(dimension=self.dimension)
            return fallback.embed_texts(texts)


# ============================================================
# CoScope Engine
# ============================================================


class CoScope:
    """
    The main CoScope engine.

    This provides a unified interface for:
    - Initializing the retrieval pipeline
    - Managing agents and their configurations
    - Managing memory storage
    - Executing collaborative retrieval

    Example usage:
    ```python
    from coscope import CoScope

    # Initialize CoScope
    coscope = CoScope()

    # Register agents
    coscope.register_agent(agent_config)
    coscope.add_memory(memory_entry)

    # Create retrieval requests
    requests = [
        coscope.create_request(agent_id="planner_1", query="What are the constraints?"),
        coscope.create_request(agent_id="verifier_1", query="What is the evidence?"),
    ]

    # Execute collaborative retrieval
    results = coscope.retrieve(requests)

    # Get result for a specific agent
    planner_results = coscope.get_results("planner_1")
    ```
    """

    def __init__(
        self,
        config: Optional[CoScopeConfig] = None,
        config_path: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        memory_store: Optional[MemoryStore] = None,
    ):
        """
        Initialize the CoScope engine.

        Args:
            config: Optional CoScope configuration
            config_path: Path to config.yaml file
            embedding_provider: Optional embedding provider override
            memory_store: Optional memory store override
        """
        # Load configuration
        self.config = config or get_config(config_path)

        # Set up logging
        self._setup_logging()

        # Initialize embedding provider
        self.embedding_provider = embedding_provider or self._create_embedding_provider()

        # Initialize memory store
        self.memory_store = memory_store or self._create_memory_store()

        # Initialize memory manager
        self.memory_manager = MemoryManager(
            memory_store=self.memory_store,
            embedding_provider=self.embedding_provider,
        )

        # Initialize scope registry
        self.scope_registry = ScopeRegistry()

        # Initialize retrieval pipeline
        self.pipeline = self._create_pipeline()

        # Agent registry
        self._agents: Dict[str, AgentConfig] = {}

        logger.info("CoScope engine initialized")

    def _setup_logging(self) -> None:
        """Set up logging based on configuration."""
        log_config = self.config.logging
        level = getattr(logging, log_config.level.upper(), logging.INFO)

        if log_config.structured:
            formatter = logging.Formatter(
                '{"time":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","msg":"%(message)s"}'
            )
        else:
            formatter = logging.Formatter(log_config.format)

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        root_logger = logging.getLogger("coscope")
        root_logger.setLevel(level)
        if not root_logger.handlers:
            root_logger.addHandler(handler)

    def _create_embedding_provider(self) -> EmbeddingProvider:
        """Create an embedding provider based on configuration."""
        provider_type = self.config.embedding.provider

        if provider_type == "openai":
            try:
                # Use configured query_embedding_dim for consistency with projection
                return OpenAIEmbeddingProvider(
                    model=self.config.embedding.model_name,
                    api_key=os.getenv("COSCOPE_OPENAI_API_KEY"),
                    api_base=self.config.embedding.openai.get("api_base"),
                    dimension=self.config.retrieval.query_embedding_dim,
                )
            except ImportError:
                logger.warning(
                    "OpenAI package not installed, falling back to random embeddings"
                )
                return RandomEmbeddingProvider(
                    dimension=self.config.embedding.openai.get("dimensions", 1536)
                )
        elif provider_type == "random":
            return RandomEmbeddingProvider(
                dimension=self.config.retrieval.query_embedding_dim
            )
        else:
            # Default to random for development
            logger.warning(
                f"Unknown embedding provider '{provider_type}', using random embeddings"
            )
            return RandomEmbeddingProvider(
                dimension=self.config.retrieval.query_embedding_dim
            )

    def _create_memory_store(self) -> MemoryStore:
        """Create a memory store based on configuration."""
        return create_memory_store(
            backend=self.config.storage.backend,
            embedding_provider=self.embedding_provider,
            persist_dir=self.config.storage.persist_dir,
        )

    def _create_pipeline(self) -> RetrievalPipeline:
        """Create the retrieval pipeline."""
        # Build pipeline configuration from CoScope config
        pipeline_config = PipelineConfig(
            query_embedding_dim=self.config.retrieval.query_embedding_dim,
            projection_dim=self.config.retrieval.projection_dim,
            svd_rank=self.config.retrieval.svd_rank,
            variant=self.config.retrieval.variant,
            use_sparse_mask=self.config.retrieval.enable_sparse_mask,
            shared_top_k=self.config.retrieval.shared_top_k,
            rerank_top_k=self.config.retrieval.rerank_top_k,
            enable_rerank=self.config.retrieval.enable_rerank,
            enable_private_fallback=self.config.retrieval.enable_private_fallback,
            fallback_threshold=self.config.retrieval.fallback_threshold,
        )

        return RetrievalPipeline(
            memory_store=self.memory_store,
            embedding_provider=self.embedding_provider,
            scope_registry=self.scope_registry,
            config=pipeline_config,
        )

    # ============================================================
    # Agent Management
    # ============================================================

    def register_agent(self, agent_config: AgentConfig) -> None:
        """
        Register an agent with the CoScope engine.

        Args:
            agent_config: Configuration for the agent
        """
        self._agents[agent_config.agent_id] = agent_config
        logger.info(f"Registered agent: {agent_config.agent_id} ({agent_config.role.value})")

    def get_agent(self, agent_id: str) -> Optional[AgentConfig]:
        """Get agent configuration by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[AgentConfig]:
        """List all registered agents."""
        return list(self._agents.values())

    def create_agent(self, **kwargs) -> Agent:
        """
        Create and register a new agent.

        Args:
            agent_id: Unique agent identifier
            role: Agent role (planner, solver, verifier, critic, memory_manager, custom)
            name: Human-readable name
            allowed_scopes: List of scope IDs the agent can access
            allowed_memory_types: List of memory types the agent can access

        Returns:
            Created Agent instance
        """
        agent_id = kwargs.get("agent_id")
        role_str = kwargs.get("role", "custom")

        try:
            role = AgentRole(role_str)
        except ValueError:
            role = AgentRole.CUSTOM

        # Create agent config
        agent_config = AgentConfig(
            agent_id=agent_id,
            role=role,
            name=kwargs.get("name", agent_id),
            description=kwargs.get("description", ""),
            allowed_scopes=kwargs.get(
                "allowed_scopes",
                self._get_default_scopes_for_role(role),
            ),
            allowed_memory_types=kwargs.get(
                "allowed_memory_types",
                self._get_default_memory_types_for_role(role),
            ),
            policy=kwargs.get(
                "policy",
                PolicyConstraints(
                    visibility=[VisibilityLevel.TEAM],
                    max_clearance=kwargs.get("max_clearance", 3),
                ),
            ),
            system_prompt=kwargs.get("system_prompt", ""),
            metadata=kwargs.get("metadata", {}),
        )

        self.register_agent(agent_config)

        return Agent(
            config=agent_config,
            state=kwargs.get("initial_state", AgentState()),
        )

    def _get_default_scopes_for_role(self, role: AgentRole) -> List[str]:
        """Get default scopes for a role."""
        defaults = {
            AgentRole.PLANNER: ["task/shared", "session/current", "workspace/default"],
            AgentRole.SOLVER: ["task/shared", "session/current", "agent/private"],
            AgentRole.VERIFIER: ["task/shared", "session/current", "governed/restricted"],
            AgentRole.CRITIC: ["task/shared", "session/current"],
            AgentRole.MEMORY_MANAGER: [
                "task/shared",
                "session/current",
                "workspace/default",
                "governed/restricted",
            ],
            AgentRole.CUSTOM: ["task/shared", "session/current"],
        }
        return defaults.get(role, defaults[AgentRole.CUSTOM])

    def _get_default_memory_types_for_role(self, role: AgentRole) -> List[MemoryType]:
        """Get default memory types for a role."""
        defaults = {
            AgentRole.PLANNER: [MemoryType.SEMANTIC, MemoryType.ARTIFACT, MemoryType.SHARED],
            AgentRole.SOLVER: [MemoryType.EPISODIC, MemoryType.ARTIFACT, MemoryType.WORKING],
            AgentRole.VERIFIER: [MemoryType.EPISODIC, MemoryType.SEMANTIC],
            AgentRole.CRITIC: [MemoryType.EPISODIC, MemoryType.SHARED, MemoryType.ARTIFACT],
            AgentRole.MEMORY_MANAGER: [
                MemoryType.EPISODIC,
                MemoryType.SEMANTIC,
                MemoryType.ARTIFACT,
                MemoryType.SHARED,
            ],
            AgentRole.CUSTOM: [MemoryType.EPISODIC, MemoryType.ARTIFACT, MemoryType.SHARED],
        }
        return defaults.get(role, defaults[AgentRole.CUSTOM])

    # ============================================================
    # Memory Management
    # ============================================================

    def add_memory(
        self,
        content: str,
        scope_id: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        **kwargs,
    ) -> MemoryEntry:
        """
        Add a memory entry to the store.

        Args:
            content: Memory content
            scope_id: Scope to store in
            memory_type: Type of memory
            **kwargs: Additional memory properties

        Returns:
            Created MemoryEntry
        """
        return self.memory_manager.store_memory(
            content=content,
            scope_id=scope_id,
            memory_type=memory_type,
            visibility=kwargs.get("visibility", [VisibilityLevel.TEAM]),
            confidence=kwargs.get("confidence", 0.5),
            tags=kwargs.get("tags"),
            ttl=kwargs.get("ttl"),
            provenance=kwargs.get("provenance"),
            metadata=kwargs.get("metadata", {}),
        )

    def add_memories(
        self,
        memories: List[Dict[str, Any]],
    ) -> List[MemoryEntry]:
        """
        Add multiple memory entries.

        Args:
            memories: List of memory dicts

        Returns:
            List of created MemoryEntry
        """
        results = []
        for mem in memories:
            entry = self.add_memory(
                content=mem["content"],
                scope_id=mem["scope_id"],
                memory_type=MemoryType(mem.get("type", "episodic")),
                **mem,
            )
            results.append(entry)
        return results

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        return self.memory_store.get(memory_id)

    def search_memory(
        self,
        query: str,
        scope_filter: Optional[List[str]] = None,
        memory_type_filter: Optional[List[MemoryType]] = None,
        top_k: int = 20,
    ) -> List["RetrievedCandidate"]:
        """
        Search memories directly.

        This bypasses the full pipeline and does a simple search.
        For full collaborative retrieval, use retrieve() instead.
        """
        return self.memory_manager.retrieve(
            query=query,
            scope_filter=scope_filter,
            memory_type_filter=memory_type_filter,
            top_k=top_k,
        )

    # ============================================================
    # Retrieval
    # ============================================================

    def create_request(
        self,
        agent_id: str,
        query: str,
        scope: Optional[ScopeSpec] = None,
        memory_types: Optional[List[MemoryType]] = None,
        state: Optional[AgentState] = None,
        priority: int = 0,
        **kwargs,
    ) -> RetrievalRequest:
        """
        Create a retrieval request for an agent.

        Args:
            agent_id: ID of the requesting agent
            query: Query text
            scope: Optional scope specification
            memory_types: Optional memory types
            state: Optional agent state
            priority: Request priority

        Returns:
            Created RetrievalRequest
        """
        agent_config = self._agents.get(agent_id)
        if not agent_config:
            logger.warning(f"Agent {agent_id} not registered, using defaults")
            agent_config = AgentConfig(
                agent_id=agent_id,
                role=AgentRole.CUSTOM,
            )

        # Build scope from agent config if not provided
        if scope is None:
            scope = ScopeSpec(
                shared_scopes=agent_config.allowed_scopes,
            )

        # Build memory types from agent config if not provided
        if memory_types is None:
            memory_types = agent_config.allowed_memory_types

        return RetrievalRequest(
            agent_id=agent_id,
            role=agent_config.role,
            query=query,
            scope=scope,
            memory_types=memory_types,
            policy=agent_config.policy,
            state=state or AgentState(),
            priority=priority,
            metadata=kwargs,
        )

    def retrieve(
        self,
        requests: List[RetrievalRequest],
        fit_projection: bool = True,
        variant: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Execute collaborative retrieval for multiple requests.

        This is the main entry point for retrieval operations.

        Args:
            requests: List of retrieval requests
            fit_projection: Whether to fit projection on this batch
            variant: Optional experiment variant override (a1, a3, a4, a5)

        Returns:
            List of RetrievalResults
        """
        if not requests:
            return []

        logger.info(f"Executing retrieval for {len(requests)} requests")

        previous_variant = self.pipeline.config.variant
        if variant is not None:
            self.pipeline.config.variant = variant
        try:
            results = self.pipeline.retrieve(requests, fit_projection=fit_projection)
        finally:
            self.pipeline.config.variant = previous_variant

        logger.info(f"Retrieved results for {len(results)} requests")

        return results

    def retrieve_single(
        self,
        agent_id: str,
        query: str,
        **kwargs,
    ) -> RetrievalResult:
        """
        Execute retrieval for a single agent request.

        Args:
            agent_id: ID of the requesting agent
            query: Query text
            **kwargs: Additional request parameters

        Returns:
            Single RetrievalResult
        """
        variant = kwargs.pop("variant", None)
        request = self.create_request(agent_id=agent_id, query=query, **kwargs)
        results = self.retrieve([request], variant=variant)

        return results[0] if results else None

    def get_results(self, agent_id: str) -> List["RetrievedCandidate"]:
        """
        Get the most recent retrieval results for an agent.

        Note: This requires results to be cached by the engine.
        """
        # For now, return empty - would need result caching
        return []

    # ============================================================
    # Scope Management
    # ============================================================

    def create_scope(
        self,
        scope_id: str,
        scope_type: str,
        **kwargs,
    ) -> "ScopeDefinition":
        """Create and register a new scope."""
        from coscope.core.scope import ScopeDefinition, ScopeType

        scope = ScopeDefinition(
            scope_id=scope_id,
            name=kwargs.get("name", scope_id),
            description=kwargs.get("description", ""),
            scope_type=ScopeType(scope_type),
            metadata=kwargs.get("metadata", {}),
        )

        self.scope_registry.register(scope)
        return scope

    # ============================================================
    # LangChain Integration
    # ============================================================

    def get_retrieval_tool(
        self,
        agent_id: str,
    ) -> Optional["CoScopeRetrievalTool"]:
        """
        Get a CoScope retrieval tool for an agent.

        This can be used as a LangChain tool.

        Args:
            agent_id: ID of the agent

        Returns:
            CoScopeRetrievalTool or None if agent not found
        """
        agent_config = self._agents.get(agent_id)
        if not agent_config:
            return None

        from coscope.agents.langchain_agent import CoScopeRetrievalTool

        return CoScopeRetrievalTool(
            pipeline=self.pipeline,
            memory_manager=self.memory_manager,
            agent_config=agent_config,
        )

    # ============================================================
    # Statistics & Debugging
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "num_agents": len(self._agents),
            "num_memories": len(self.memory_store._entries),
            "pipeline_stats": self.pipeline.get_stats(),
            "memory_store_stats": self.memory_store.get_stats(),
        }

    def reset(self) -> None:
        """Reset the engine state."""
        self._agents.clear()
        self.pipeline.reset_stats()
        if hasattr(self.memory_store, "clear"):
            self.memory_store.clear()
        logger.info("CoScope engine reset")


# ============================================================
# Factory Functions
# ============================================================


def create_coscope(
    config_path: Optional[str] = None,
    **kwargs,
) -> CoScope:
    """
    Factory function to create a CoScope engine.

    Args:
        config_path: Path to config.yaml file
        **kwargs: Additional configuration overrides

    Returns:
        Initialized CoScope engine
    """
    # Load configuration
    loader = ConfigLoader(config_path)
    config = loader.load()

    # Apply any kwargs overrides
    # (In a full implementation, would apply kwargs to config)

    return CoScope(config=config)
