"""
CoScope Configuration Management.

Loads configuration from:
1. config.yaml (structural pipeline configuration)
2. Environment variables (secrets, runtime overrides)
3. Programmatic overrides (for testing/dynamic config)

Environment variables always take precedence over config.yaml.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Any, Literal, Optional, Union
from dataclasses import dataclass, field
from functools import lru_cache
from dotenv import load_dotenv

import yaml
from pydantic import BaseModel, Field

# Load .env file if present
_load_env_done = False


def _ensure_env_loaded():
    global _load_env_done
    if not _load_env_done:
        load_dotenv()
        _load_env_done = True


# ============================================================
# Pydantic Models for Config Sections
# ============================================================


class RetrievalConfig(BaseModel):
    query_embedding_dim: int = 1024
    projection_dim: int = 256
    svd_rank: int = 64
    shared_top_k: int = 50
    rerank_top_k: int = 20
    max_batch_size: int = 100
    request_timeout: int = 30
    enable_sparse_mask: bool = True
    enable_private_fallback: bool = True
    fallback_threshold: int = 5
    enable_rerank: bool = True


class ScopePolicyConfig(BaseModel):
    visibility: list[str] = Field(default_factory=lambda: ["team"])
    max_clearance: int = 3
    exclude_quarantined: bool = True
    audit_required: bool = False


class ScopeDefinition(BaseModel):
    id: str
    name: str
    description: str = ""
    scope_type: str
    default_policy: ScopePolicyConfig = Field(default_factory=ScopePolicyConfig)


class ScopeHierarchyConfig(BaseModel):
    agent_private: list[ScopeDefinition] = Field(default_factory=list)
    task_shared: list[ScopeDefinition] = Field(default_factory=list)
    session: list[ScopeDefinition] = Field(default_factory=list)
    workspace: list[ScopeDefinition] = Field(default_factory=list)
    governed: list[ScopeDefinition] = Field(default_factory=list)


class PolicyConfig(BaseModel):
    visibility_levels: list[str] = Field(
        default_factory=lambda: ["owner", "team", "session", "public", "restricted"]
    )
    clearance_levels: list[dict] = Field(default_factory=list)
    excluded_zones: list[str] = Field(
        default_factory=lambda: ["quarantine", "audit_hold", "pending_verify"]
    )


class MemoryTypeDefinition(BaseModel):
    name: str
    description: str = ""
    default_relevance_window: int = 86400
    retention_policy: str = "adaptive"


class MemoryTypesConfig(BaseModel):
    episodic: MemoryTypeDefinition = Field(default_factory=lambda: MemoryTypeDefinition(
        name="Episodic Memory", description="Event and experience records",
        default_relevance_window=86400, retention_policy="adaptive"
    ))
    semantic: MemoryTypeDefinition = Field(default_factory=lambda: MemoryTypeDefinition(
        name="Semantic Memory", description="Knowledge and fact records",
        default_relevance_window=604800, retention_policy="persistent"
    ))
    artifact: MemoryTypeDefinition = Field(default_factory=lambda: MemoryTypeDefinition(
        name="Artifact Memory", description="Generated artifacts and documents",
        default_relevance_window=259200, retention_policy="versioned"
    ))
    shared: MemoryTypeDefinition = Field(default_factory=lambda: MemoryTypeDefinition(
        name="Shared Memory", description="Collaboratively generated shared records",
        default_relevance_window=172800, retention_policy="collaborative"
    ))
    working: MemoryTypeDefinition = Field(default_factory=lambda: MemoryTypeDefinition(
        name="Working Memory", description="Temporary scratch space for active tasks",
        default_relevance_window=3600, retention_policy="ephemeral"
    ))


class RerankWeights(BaseModel):
    summary_score: float = 0.2
    constraint_relevance: float = 0.2
    factual_relevance: float = 0.2
    precision: float = 0.2
    recency: float = 0.2
    authority: float = 0.2
    provenance_score: float = 0.2
    source_reliability: float = 0.2
    timestamp_relevance: float = 0.2
    conflict_score: float = 0.2
    edge_case_relevance: float = 0.2
    failure_case_relevance: float = 0.2
    completeness: float = 0.2
    organization_score: float = 0.2
    retention_relevance: float = 0.2


class AgentRoleDefinition(BaseModel):
    name: str
    description: str = ""
    default_scopes: list[str] = Field(default_factory=list)
    default_memory_types: list[str] = Field(default_factory=list)
    rerank_weights: RerankWeights = Field(default_factory=RerankWeights)


class AgentRolesConfig(BaseModel):
    planner: AgentRoleDefinition = Field(default_factory=lambda: AgentRoleDefinition(
        name="Planner Agent", description="Responsible for task planning",
        default_scopes=["task_shared", "session", "workspace"],
        default_memory_types=["semantic", "artifact", "shared"],
        rerank_weights=RerankWeights(
            summary_score=0.4, constraint_relevance=0.3, recency=0.2, authority=0.1
        )
    ))
    solver: AgentRoleDefinition = Field(default_factory=lambda: AgentRoleDefinition(
        name="Solver Agent", description="Responsible for solving sub-tasks",
        default_scopes=["task_shared", "session", "agent_private"],
        default_memory_types=["episodic", "artifact", "working"],
        rerank_weights=RerankWeights(
            factual_relevance=0.5, precision=0.3, recency=0.1, authority=0.1
        )
    ))
    verifier: AgentRoleDefinition = Field(default_factory=lambda: AgentRoleDefinition(
        name="Verifier Agent", description="Responsible for verification",
        default_scopes=["task_shared", "session", "governed"],
        default_memory_types=["episodic", "semantic"],
        rerank_weights=RerankWeights(
            provenance_score=0.4, source_reliability=0.3, timestamp_relevance=0.2, authority=0.1
        )
    ))
    critic: AgentRoleDefinition = Field(default_factory=lambda: AgentRoleDefinition(
        name="Critic Agent", description="Responsible for identifying conflicts",
        default_scopes=["task_shared", "session"],
        default_memory_types=["episodic", "shared", "artifact"],
        rerank_weights=RerankWeights(
            conflict_score=0.4, edge_case_relevance=0.3, failure_case_relevance=0.2, recency=0.1
        )
    ))
    memory_manager: AgentRoleDefinition = Field(default_factory=lambda: AgentRoleDefinition(
        name="Memory Manager Agent", description="Responsible for memory lifecycle",
        default_scopes=["task_shared", "session", "workspace", "governed"],
        default_memory_types=["episodic", "semantic", "artifact", "shared"],
        rerank_weights=RerankWeights(
            completeness=0.4, organization_score=0.3, retention_relevance=0.2, authority=0.1
        )
    ))


class EmbeddingProviderConfig(BaseModel):
    provider: str = "openai"
    model_name: str = "text-embedding-3-small"
    batch_size: int = 32
    device: str = "cpu"
    normalize: bool = True
    timeout: int = 30


class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    model_name: str = "text-embedding-3-small"
    batch_size: int = 32
    device: str = "cpu"
    normalize: bool = True
    timeout: int = 30
    openai: dict = Field(default_factory=lambda: {
        "api_base": "https://api.openai.com/v1",
        "model": "text-embedding-3-small",
        "dimensions": 1536
    })
    huggingface: dict = Field(default_factory=lambda: {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "normalize": True
    })


class StorageConfig(BaseModel):
    backend: str = "inmemory"
    persist_dir: str = "./coscope_data/storage"
    enable_persistence: bool = False
    chromadb: dict = Field(default_factory=lambda: {
        "collection_name": "coscope_memories",
        "distance_metric": "cosine"
    })
    faiss: dict = Field(default_factory=lambda: {
        "index_type": "IDMap2,SQ8",
        "nlist": 100
    })
    qdrant: dict = Field(default_factory=lambda: {
        "collection_name": "coscope_memories",
        "vector_size": 1536,
        "distance": "Cosine"
    })


class LangGraphConfig(BaseModel):
    checkpoint_enabled: bool = True
    checkpoint_dir: str = "./coscope_data/checkpoints"
    interrupt_before_nodes: list[str] = Field(default_factory=list)
    interrupt_after_nodes: list[str] = Field(default_factory=list)


class AgentFrameworkConfig(BaseModel):
    framework: str = "langchain"
    enable_langgraph: bool = False
    langgraph_config: LangGraphConfig = Field(default_factory=LangGraphConfig)


class EvaluationBaseline(BaseModel):
    name: str
    enabled: bool = True


class EvaluationConfig(BaseModel):
    enabled: bool = False
    output_dir: str = "./coscope_data/eval_results"
    baselines: list[EvaluationBaseline] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=lambda: [
        "recall@k", "mrr@k", "evidence_hit_rate", "answer_support_rate",
        "first_stage_retrieval_count", "duplicate_candidate_rate",
        "candidate_reuse_rate", "fallback_trigger_rate", "average_latency"
    ])


class LoggingConfig(BaseModel):
    level: str = "INFO"
    structured: bool = False
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = "./coscope_data/logs"


class AdvancedConfig(BaseModel):
    experimental_features: bool = False
    enable_metrics_collection: bool = True
    metrics_flush_interval: int = 60
    enable_profiling: bool = False
    profiler_output_dir: str = "./coscope_data/profiles"


# ============================================================
# Full Config Model
# ============================================================


class CoScopeConfig(BaseModel):
    """Full CoScope configuration loaded from config.yaml + env vars."""

    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    scope_hierarchy: ScopeHierarchyConfig = Field(default_factory=ScopeHierarchyConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    memory_types: MemoryTypesConfig = Field(default_factory=MemoryTypesConfig)
    agent_roles: AgentRolesConfig = Field(default_factory=AgentRolesConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    agent_framework: AgentFrameworkConfig = Field(default_factory=AgentFrameworkConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)


# ============================================================
# Config Loader
# ============================================================


class ConfigLoader:
    """
    Loads CoScope configuration from multiple sources with precedence:
    1. Environment variables (highest priority)
    2. config.yaml
    3. Defaults (lowest priority)
    """

    def __init__(self, config_path: Optional[str] = None):
        _ensure_env_loaded()
        self.config_path = config_path or self._find_config_file()
        self._raw_yaml: dict[str, Any] = {}
        self._load_yaml()

    def _find_config_file(self) -> Optional[str]:
        """Search for config.yaml in standard locations."""
        search_paths = [
            Path.cwd() / "config.yaml",
            Path.cwd() / "config.yaml.example",
            Path(__file__).parent.parent.parent / "config.yaml",
            Path(__file__).parent.parent.parent / "config.yaml.example",
        ]
        for path in search_paths:
            if path.exists():
                return str(path)
        return None

    def _load_yaml(self) -> None:
        """Load raw YAML configuration."""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, "r") as f:
                self._raw_yaml = yaml.safe_load(f) or {}

    def _apply_env_overrides(self, config: CoScopeConfig) -> CoScopeConfig:
        """
        Apply environment variable overrides.
        Environment variables take precedence over config.yaml.
        """
        # Retrieval overrides
        retrieval_data = self._raw_yaml.get("retrieval", {})
        retrieval_dict = {
            "query_embedding_dim": int(os.getenv(
                "COSCOPE_QUERY_EMBEDDING_DIM",
                str(retrieval_data.get("query_embedding_dim", 1024))
            )),
            "projection_dim": int(os.getenv(
                "COSCOPE_PROJECTION_DIM",
                str(retrieval_data.get("projection_dim", 256))
            )),
            "svd_rank": int(os.getenv(
                "COSCOPE_SVD_RANK",
                str(retrieval_data.get("svd_rank", 64))
            )),
            "shared_top_k": int(os.getenv(
                "COSCOPE_SHARED_TOP_K",
                str(retrieval_data.get("shared_top_k", 50))
            )),
            "rerank_top_k": int(os.getenv(
                "COSCOPE_RERANK_TOP_K",
                str(retrieval_data.get("rerank_top_k", 20))
            )),
            "max_batch_size": int(os.getenv(
                "COSCOPE_MAX_BATCH_SIZE",
                str(retrieval_data.get("max_batch_size", 100))
            )),
            "request_timeout": int(os.getenv(
                "COSCOPE_REQUEST_TIMEOUT",
                str(retrieval_data.get("request_timeout", 30))
            )),
            "enable_sparse_mask": os.getenv(
                "COSCOPE_USE_SPARSE_MASK",
                str(retrieval_data.get("enable_sparse_mask", True))
            ).lower() in ("true", "1", "yes"),
            "enable_private_fallback": os.getenv(
                "COSCOPE_ENABLE_PRIVATE_FALLBACK",
                str(retrieval_data.get("enable_private_fallback", True))
            ).lower() in ("true", "1", "yes"),
            "fallback_threshold": int(os.getenv(
                "COSCOPE_FALLBACK_THRESHOLD",
                str(retrieval_data.get("fallback_threshold", 5))
            )),
            "enable_rerank": os.getenv(
                "COSCOPE_ENABLE_RERANK",
                str(retrieval_data.get("enable_rerank", True))
            ).lower() in ("true", "1", "yes"),
        }
        config.retrieval = RetrievalConfig(**retrieval_dict)

        # Storage overrides
        storage_data = self._raw_yaml.get("storage", {})
        storage_dict = {
            "backend": os.getenv(
                "COSCOPE_MEMORY_BACKEND",
                storage_data.get("backend", "inmemory")
            ),
            "persist_dir": os.getenv(
                "COSCOPE_STORAGE_PERSIST_DIR",
                storage_data.get("persist_dir", "./coscope_data/storage")
            ),
            "enable_persistence": os.getenv(
                "COSCOPE_ENABLE_PERSISTENCE",
                str(storage_data.get("enable_persistence", False))
            ).lower() in ("true", "1", "yes"),
        }
        if "chromadb" in storage_data:
            storage_dict["chromadb"] = storage_data["chromadb"]
        if "faiss" in storage_data:
            storage_dict["faiss"] = storage_data["faiss"]
        if "qdrant" in storage_data:
            storage_dict["qdrant"] = storage_data["qdrant"]
        config.storage = StorageConfig(**storage_dict)

        # Embedding overrides
        embedding_data = self._raw_yaml.get("embedding", {})
        embedding_dict = {
            "provider": os.getenv(
                "COSCOPE_EMBEDDING_PROVIDER",
                embedding_data.get("provider", "openai")
            ),
            "model_name": os.getenv(
                "COSCOPE_EMBEDDING_MODEL",
                embedding_data.get("model_name", "text-embedding-3-small")
            ),
            "batch_size": int(os.getenv(
                "COSCOPE_EMBEDDING_BATCH_SIZE",
                str(embedding_data.get("batch_size", 32))
            )),
            "device": os.getenv(
                "COSCOPE_EMBEDDING_DEVICE",
                embedding_data.get("device", "cpu")
            ),
            "normalize": embedding_data.get("normalize", True),
            "timeout": int(os.getenv(
                "COSCOPE_EMBEDDING_TIMEOUT",
                str(embedding_data.get("timeout", 30))
            )),
        }
        if "openai" in embedding_data:
            embedding_dict["openai"] = embedding_data["openai"]
        if "huggingface" in embedding_data:
            embedding_dict["huggingface"] = embedding_data["huggingface"]
        config.embedding = EmbeddingConfig(**embedding_dict)

        # Agent framework overrides
        framework_data = self._raw_yaml.get("agent_framework", {})
        langgraph_data = framework_data.get("langgraph_config", {})
        config.agent_framework = AgentFrameworkConfig(
            framework=os.getenv(
                "COSCOPE_AGENT_FRAMEWORK",
                framework_data.get("framework", "langchain")
            ),
            enable_langgraph=os.getenv(
                "COSCOPE_ENABLE_LANGGRAPH",
                str(framework_data.get("enable_langgraph", False))
            ).lower() in ("true", "1", "yes"),
            langgraph_config=LangGraphConfig(
                checkpoint_enabled=langgraph_data.get("checkpoint_enabled", True),
                checkpoint_dir=os.getenv(
                    "COSCOPE_LANGGRAPH_CHECKPOINT_DIR",
                    langgraph_data.get("checkpoint_dir", "./coscope_data/checkpoints")
                ),
                interrupt_before_nodes=langgraph_data.get("interrupt_before_nodes", []),
                interrupt_after_nodes=langgraph_data.get("interrupt_after_nodes", []),
            )
        )

        # Evaluation overrides
        eval_data = self._raw_yaml.get("evaluation", {})
        config.evaluation = EvaluationConfig(
            enabled=os.getenv(
                "COSCOPE_EVAL_MODE",
                str(eval_data.get("enabled", False))
            ).lower() in ("true", "1", "yes"),
            output_dir=os.getenv(
                "COSCOPE_EVAL_OUTPUT_DIR",
                eval_data.get("output_dir", "./coscope_data/eval_results")
            ),
            baselines=eval_data.get("baselines", []),
            metrics=eval_data.get("metrics", []),
        )

        # Logging overrides
        log_data = self._raw_yaml.get("logging", {})
        config.logging = LoggingConfig(
            level=os.getenv("COSCOPE_LOG_LEVEL", log_data.get("level", "INFO")),
            structured=os.getenv(
                "COSCOPE_STRUCTURED_LOGGING",
                str(log_data.get("structured", False))
            ).lower() in ("true", "1", "yes"),
            format=log_data.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            log_dir=log_data.get("log_dir", "./coscope_data/logs"),
        )

        # Advanced overrides
        advanced_data = self._raw_yaml.get("advanced", {})
        config.advanced = AdvancedConfig(
            experimental_features=os.getenv(
                "COSCOPE_EXPERIMENTAL_FEATURES",
                str(advanced_data.get("experimental_features", False))
            ).lower() in ("true", "1", "yes"),
            enable_metrics_collection=advanced_data.get("enable_metrics_collection", True),
            metrics_flush_interval=advanced_data.get("metrics_flush_interval", 60),
            enable_profiling=advanced_data.get("enable_profiling", False),
            profiler_output_dir=advanced_data.get("profiler_output_dir", "./coscope_data/profiles"),
        )

        # Parse scope hierarchy from YAML
        if "scope_hierarchy" in self._raw_yaml:
            scope_data = self._raw_yaml["scope_hierarchy"]
            config.scope_hierarchy = ScopeHierarchyConfig(
                agent_private=self._parse_scope_definitions(
                    scope_data.get("agent_private", [])
                ),
                task_shared=self._parse_scope_definitions(
                    scope_data.get("task_shared", [])
                ),
                session=self._parse_scope_definitions(
                    scope_data.get("session", [])
                ),
                workspace=self._parse_scope_definitions(
                    scope_data.get("workspace", [])
                ),
                governed=self._parse_scope_definitions(
                    scope_data.get("governed", [])
                ),
            )

        # Parse policy from YAML
        if "policy" in self._raw_yaml:
            config.policy = PolicyConfig(**self._raw_yaml["policy"])

        # Parse memory types from YAML
        if "memory_types" in self._raw_yaml:
            mt_data = self._raw_yaml["memory_types"]
            config.memory_types = MemoryTypesConfig(
                episodic=MemoryTypeDefinition(**mt_data.get("episodic", {})) if "episodic" in mt_data else MemoryTypeDefinition(name="Episodic Memory"),
                semantic=MemoryTypeDefinition(**mt_data.get("semantic", {})) if "semantic" in mt_data else MemoryTypeDefinition(name="Semantic Memory"),
                artifact=MemoryTypeDefinition(**mt_data.get("artifact", {})) if "artifact" in mt_data else MemoryTypeDefinition(name="Artifact Memory"),
                shared=MemoryTypeDefinition(**mt_data.get("shared", {})) if "shared" in mt_data else MemoryTypeDefinition(name="Shared Memory"),
                working=MemoryTypeDefinition(**mt_data.get("working", {})) if "working" in mt_data else MemoryTypeDefinition(name="Working Memory"),
            )

        # Parse agent roles from YAML
        if "agent_roles" in self._raw_yaml:
            roles_data = self._raw_yaml["agent_roles"]
            config.agent_roles = AgentRolesConfig(**roles_data)

        return config

    def _parse_scope_definitions(
        self, data: list[dict]
    ) -> list[ScopeDefinition]:
        """Parse scope definitions from YAML data."""
        result = []
        for item in data:
            policy = item.get("default_policy", {})
            scope_policy = ScopePolicyConfig(
                visibility=policy.get("visibility", ["team"]),
                max_clearance=policy.get("max_clearance", 3),
                exclude_quarantined=policy.get("exclude_quarantined", True),
                audit_required=policy.get("audit_required", False),
            )
            result.append(ScopeDefinition(
                id=item["id"],
                name=item["name"],
                description=item.get("description", ""),
                scope_type=item["scope_type"],
                default_policy=scope_policy,
            ))
        return result

    def load(self) -> CoScopeConfig:
        """Load the full configuration."""
        # Start with default config
        config = CoScopeConfig()

        # Apply YAML config
        if self._raw_yaml:
            try:
                config = CoScopeConfig(**self._raw_yaml)
            except Exception as e:
                logging.warning(
                    f"Failed to parse config.yaml, using defaults: {e}"
                )

        # Apply environment variable overrides
        config = self._apply_env_overrides(config)

        return config


@lru_cache(maxsize=1)
def get_config(config_path: Optional[str] = None) -> CoScopeConfig:
    """
    Get the global CoScope configuration.
    Loads from config.yaml and environment variables.
    """
    loader = ConfigLoader(config_path)
    return loader.load()


def reload_config(config_path: Optional[str] = None) -> CoScopeConfig:
    """Force reload of configuration."""
    get_config.cache_clear()
    return get_config(config_path)
