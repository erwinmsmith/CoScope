"""
Abstract base classes (ABCs) for all major CoScope subsystems.

Every concrete implementation MUST subclass the corresponding ABC defined here.
This file is the single contract for the CoScope plugin architecture:
  - AbstractLoader        : raw dataset reading
  - AbstractGraphBuilder  : GoT/CoT/ToT graph construction from raw_item
  - AbstractScopeBuilder  : memory scope (workspace/shared/private/restricted)
  - AbstractRolloutEngine : LLM-driven artifact generation within one episode
  - AbstractEpisodeBuilder: single episode assembly (graph+memory+agents+rho)
  - AbstractDatasetPipeline: full batch pipeline orchestration
  - LLMClient             : generation backend Protocol (lightweight, no ABC)
  - Embedder              : embedding backend Protocol
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


# ============================================================
# Forward-declare types to avoid circular imports.
# Concrete type annotations use string literals where needed.
# ============================================================


# ============================================================
# LLM backend (Protocol, not ABC, so backends need not import this file)
# ============================================================


from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class LLMResponse:
    """Unified LLM response container."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    model: str = ""
    latency_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """
    Generation-side LLM interface.

    Implementations must be stateless with respect to requests so that rollout
    is trivially parallelizable across episodes.
    """

    name: str
    max_tokens: int

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        stop: Optional[List[str]] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> LLMResponse:
        ...

    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: Optional[str] = None,
        stop: Optional[List[str]] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[LLMResponse]:
        ...


# ============================================================
# Embedder backend (Protocol)
# ============================================================


@dataclass
class EmbeddingResult:
    """Batch embedding output."""

    vectors: np.ndarray
    dim: int = 0
    model: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Embedder(Protocol):
    """Text embedding interface."""

    name: str
    dim: int

    def embed(self, text: str) -> np.ndarray:
        ...

    def embed_batch(self, texts: List[str]) -> EmbeddingResult:
        ...


# ============================================================
# AbstractLoader
# ============================================================


class AbstractLoader(ABC):
    """
    Raw dataset reader.

    Concrete implementations live in `coscope/utils/loaders/`.
    Each implementation reads one dataset (e.g. MuSiQue, GSM8K) from disk
    and returns a normalised list of raw_item dicts.
    """

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """Return a list of raw_item dicts for the configured split."""

    @abstractmethod
    def load_item(self, index: int) -> Optional[Dict[str, Any]]:
        """Return a single raw_item by index, or None if out-of-bounds."""


# ============================================================
# AbstractGraphBuilder
# ============================================================


class AbstractGraphBuilder(ABC):
    """
    Constructs a GoTGraph (or CoT/ToT equivalent) from a raw_item.

    Concrete implementations live in `coscope/graph/got/`, `coscope/graph/cot/`,
    `coscope/graph/tot/`.
    """

    @abstractmethod
    def build(
        self,
        raw_item: Dict[str, Any],
        *,
        dataset: str,
        target_graph_type: Any,
        seed: int = 42,
    ) -> Any:
        """
        Build and return a graph object (GoTGraph or equivalent).

        Raises InvalidGraphError if the raw_item cannot support the requested
        graph_type.
        """


# ============================================================
# AbstractScopeBuilder
# ============================================================


class AbstractScopeBuilder(ABC):
    """
    Constructs the MemoryEntry list for a single memory scope region.

    Four concrete scope regions:
      - WorkspaceBuilder   (coscope/memory/)
      - TaskSharedBuilder  (coscope/memory/)
      - PrivateBuilder     (coscope/memory/)
      - RestrictedBuilder  (coscope/memory/)
    """

    @abstractmethod
    def build(self, *args: Any, **kwargs: Any) -> List[Any]:
        """Build and return a list of MemoryEntry objects for this scope."""


# ============================================================
# AbstractRolloutEngine
# ============================================================


class AbstractRolloutEngine(ABC):
    """
    Drives LLM-based artifact generation for a single episode.

    Concrete implementation: `coscope/rollout/rollout_engine.py`.

    The engine iterates graph nodes in topological order, calls the LLM
    to fill each ArtifactSlot, writes entries via ArtifactWriter, and
    returns a validated ArtifactTrace.
    """

    @abstractmethod
    def run(
        self,
        *,
        got_graph: Any,
        memory_entries: List[Any],
        episode_id: str,
        dataset: str,
        raw_item: Dict[str, Any],
        seed: int = 42,
    ) -> Any:
        """
        Execute the rollout and return an ArtifactTrace.

        The returned trace is later merged into Episode.memory_entries by
        EpisodeBuilder.
        """


# ============================================================
# AbstractEpisodeBuilder
# ============================================================


class AbstractEpisodeBuilder(ABC):
    """
    Assembles a single Episode from a raw_item and target graph type.

    Orchestrates: graph construction → scope building → agent building →
    rollout → rho calculation → subset labeling → policy validation.

    Concrete implementation: `coscope/construction/episode_builder.py`.
    """

    @abstractmethod
    def build_episode(
        self,
        raw_item: Dict[str, Any],
        dataset: str,
        split: str,
        target_graph_type: Any,
        seed: int = 42,
    ) -> Optional[Any]:
        """
        Build and return a single Episode, or None if the item must be skipped
        (e.g. graph construction fails or policy validation rejects it).
        """


# ============================================================
# AbstractDatasetPipeline
# ============================================================


class AbstractDatasetPipeline(ABC):
    """
    Batch dataset construction pipeline.

    Iterates raw_items for a (dataset, split), builds episodes for each
    requested target_graph_type, and shards output into per-subset JSONL files.

    Concrete implementation: `coscope/construction/dataset_pipeline.py`.
    """

    @abstractmethod
    def run(
        self,
        dataset: str,
        split: str,
        target_graph_types: Sequence[Any],
        *,
        max_workers: int = 1,
        seed: int = 42,
        dataset_config: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        enforce_coverage: bool = True,
    ) -> Dict[str, int]:
        """
        Run the pipeline end-to-end for one (dataset, split) slice.

        Returns a mapping of shard file path -> number of episodes written.
        """
