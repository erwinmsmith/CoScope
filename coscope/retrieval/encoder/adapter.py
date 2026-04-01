"""
Embedding adapters for different provider interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from coscope.core.types import EmbeddingProvider

logger = __import__("logging").getLogger(__name__)


class EmbeddingAdapter(ABC):
    """
    Base class for embedding provider adapters.

    Adapters normalize the interface between different embedding providers
    and the retrieval system.
    """

    @abstractmethod
    def embed_query(self, query: str) -> Any:
        """Embed a single query."""
        ...

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[Any]:
        """Embed multiple texts."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        ...


class DirectAdapter(EmbeddingAdapter):
    """
    Direct pass-through adapter for providers that match the interface.
    """

    def __init__(self, provider: "EmbeddingProvider"):
        self._provider = provider

    def embed_query(self, query: str) -> Any:
        return self._provider.embed_query(query)

    def embed_texts(self, texts: List[str]) -> List[Any]:
        return self._provider.embed_texts(texts)

    @property
    def dimension(self) -> int:
        if hasattr(self._provider, "dimension"):
            return self._provider.dimension
        return 0


class LangChainEmbeddingAdapter(EmbeddingAdapter):
    """
    Adapter for LangChain embedding models.

    This adapter wraps LangChain's embedding interface to work with CoScope.
    """

    def __init__(self, embedding_model: Any):
        """
        Args:
            embedding_model: A LangChain embedding model
        """
        self._model = embedding_model

    def embed_query(self, query: str) -> Any:
        """Embed a single query."""
        embedding = self._model.embed_query(query)
        return self._to_numpy(embedding)

    def embed_texts(self, texts: List[str]) -> List[Any]:
        """Embed multiple texts."""
        embeddings = self._model.embed_documents(texts)
        return [self._to_numpy(e) for e in embeddings]

    @property
    def dimension(self) -> int:
        """Get dimension from first embedding."""
        return 0  # Unknown without querying

    def _to_numpy(self, embedding: Any) -> Any:
        """Convert embedding to numpy array if needed."""
        import numpy as np

        if isinstance(embedding, np.ndarray):
            return embedding
        if isinstance(embedding, list):
            return np.array(embedding)
        return embedding


class BatchAdapter(EmbeddingAdapter):
    """
    Adapter that adds batching support to providers without it.
    """

    def __init__(
        self,
        provider: "EmbeddingProvider",
        batch_size: int = 32,
    ):
        self._provider = provider
        self._batch_size = batch_size

    def embed_query(self, query: str) -> Any:
        return self._provider.embed_query(query)

    def embed_texts(self, texts: List[str]) -> List[Any]:
        """Embed texts in batches."""
        import numpy as np

        all_embeddings = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            batch_embeddings = self._provider.embed_texts(batch)

            # Handle different return formats
            for emb in batch_embeddings:
                if isinstance(emb, np.ndarray):
                    all_embeddings.append(emb)
                elif isinstance(emb, list):
                    all_embeddings.append(np.array(emb))
                else:
                    all_embeddings.append(emb)

        return all_embeddings

    @property
    def dimension(self) -> int:
        if hasattr(self._provider, "dimension"):
            return self._provider.dimension
        return 0


class CachedAdapter(EmbeddingAdapter):
    """
    Caching adapter to avoid re-encoding identical queries.
    """

    def __init__(
        self,
        provider: "EmbeddingProvider",
        cache_size: int = 10000,
    ):
        self._provider = provider
        self._cache: dict = {}
        self._cache_order: List[str] = []
        self._cache_size = cache_size

    def embed_query(self, query: str) -> Any:
        if query in self._cache:
            return self._cache[query]

        embedding = self._provider.embed_query(query)
        self._add_to_cache(query, embedding)
        return embedding

    def embed_texts(self, texts: List[str]) -> List[Any]:
        """Embed texts with caching."""
        embeddings = []
        uncached = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text in self._cache:
                embeddings.append(self._cache[text])
            else:
                embeddings.append(None)
                uncached.append(text)
                uncached_indices.append(i)

        if uncached:
            new_embeddings = self._provider.embed_texts(uncached)
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings[idx] = emb
                self._add_to_cache(texts[idx], emb)

        return [e for e in embeddings if e is not None]

    @property
    def dimension(self) -> int:
        if hasattr(self._provider, "dimension"):
            return self._provider.dimension
        return 0

    def _add_to_cache(self, text: str, embedding: Any) -> None:
        """Add embedding to cache with LRU eviction."""
        if text not in self._cache:
            self._cache_order.append(text)
            while len(self._cache_order) > self._cache_size:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)
        self._cache[text] = embedding

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._cache_order.clear()

    def get_cache_stats(self) -> dict:
        return {
            "size": len(self._cache),
            "max_size": self._cache_size,
        }
