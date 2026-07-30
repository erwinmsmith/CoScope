"""Shared, CPU-only FastEmbed adapter for local BGE inference."""

from __future__ import annotations

import hashlib
import importlib.metadata
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

from coscope.config import EmbeddingSettings
from coscope.core.usage import UsageLedger


class FastEmbedBackend(Protocol):
    dimension: int
    model_version: str
    model_sha256: str

    def embed_many(
        self,
        texts: list[str],
    ) -> tuple[
        list[tuple[float, ...]],
        int,
        float,
        int,
        int,
    ]: ...


class _OnnxBackend:
    """One process-wide ONNX session shared by task-local adapters."""

    def __init__(self, settings: EmbeddingSettings):
        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError(
                "fastembed is required for local embeddings; install the "
                "project with the local embedding dependency"
            ) from error

        self.dimension = settings.dimension
        self.batch_size = settings.batch_size
        self.result_cache_size = settings.result_cache_size
        self._lock = threading.Lock()
        self._result_cache: OrderedDict[
            str,
            tuple[float, ...],
        ] = OrderedDict()
        self._model = TextEmbedding(
            model_name=settings.model,
            cache_dir=settings.cache_dir,
            threads=settings.threads,
            providers=["CPUExecutionProvider"],
            local_files_only=settings.local_files_only,
        )
        actual_dimension = int(self._model.embedding_size)
        if actual_dimension != self.dimension:
            raise RuntimeError(
                f"local embedding dimension mismatch: configured "
                f"{self.dimension}, model provides {actual_dimension}"
            )
        self.model_sha256 = _model_sha256(self._model)
        fastembed_version = importlib.metadata.version("fastembed")
        self.model_version = (
            f"fastembed:{settings.model}:{self.dimension}:"
            f"fastembed-{fastembed_version}:sha256-{self.model_sha256[:16]}"
        )

    def embed_many(
        self,
        texts: list[str],
    ) -> tuple[
        list[tuple[float, ...]],
        int,
        float,
        int,
        int,
    ]:
        started = time.perf_counter()
        with self._lock:
            token_count = int(self._model.token_count(texts))
            missing = list(
                dict.fromkeys(
                    text
                    for text in texts
                    if text not in self._result_cache
                )
            )
            arrays = (
                list(
                    self._model.embed(
                        missing,
                        batch_size=self.batch_size,
                        parallel=None,
                    )
                )
                if missing
                else []
            )
            for text, array in zip(missing, arrays, strict=True):
                self._result_cache[text] = tuple(
                    float(value) for value in array
                )
                self._result_cache.move_to_end(text)
            vectors = []
            for text in texts:
                vector = self._result_cache[text]
                self._result_cache.move_to_end(text)
                vectors.append(vector)
            while len(self._result_cache) > self.result_cache_size:
                self._result_cache.popitem(last=False)
        elapsed = time.perf_counter() - started
        misses = len(missing)
        hits = len(texts) - misses
        return vectors, token_count, elapsed, hits, misses


_BACKENDS: dict[
    tuple[str, int, str, int, int, int, bool],
    FastEmbedBackend,
] = {}
_BACKENDS_LOCK = threading.Lock()


def _shared_backend(settings: EmbeddingSettings) -> FastEmbedBackend:
    cache_dir = str(Path(settings.cache_dir).expanduser().resolve())
    key = (
        settings.model,
        settings.dimension,
        cache_dir,
        settings.threads,
        settings.batch_size,
        settings.result_cache_size,
        settings.local_files_only,
    )
    with _BACKENDS_LOCK:
        backend = _BACKENDS.get(key)
        if backend is None:
            backend = _OnnxBackend(settings)
            _BACKENDS[key] = backend
        return backend


class FastEmbedEmbedding:
    """Task-local metrics wrapper around a shared FastEmbed ONNX backend."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        backend: FastEmbedBackend | None = None,
        usage_ledger: UsageLedger | None = None,
    ):
        settings.validate()
        if settings.provider != "fastembed":
            raise ValueError("FastEmbedEmbedding requires provider=fastembed")
        self.settings = settings
        self.dimension = settings.dimension
        self.usage_ledger = usage_ledger
        self.backend = backend or _shared_backend(settings)
        self.model_version = self.backend.model_version
        self.model_sha256 = self.backend.model_sha256

    def embed(self, text: str) -> tuple[float, ...]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        (
            vectors,
            token_count,
            elapsed,
            cache_hits,
            cache_misses,
        ) = self.backend.embed_many(texts)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"local embedding count mismatch: expected {len(texts)}, "
                f"received {len(vectors)}"
            )
        if any(len(vector) != self.dimension for vector in vectors):
            raise RuntimeError(
                f"local embedding dimension does not match {self.dimension}"
            )
        if self.usage_ledger is not None:
            self.usage_ledger.record(
                "embedding",
                self.model_version,
                {
                    "prompt_tokens": token_count,
                    "total_tokens": token_count,
                },
                metadata={
                    "execution": "local_cpu",
                    "input_count": str(len(texts)),
                    "latency_seconds": f"{elapsed:.9f}",
                    "model_sha256": self.model_sha256,
                    "result_cache_hits": str(cache_hits),
                    "result_cache_misses": str(cache_misses),
                },
            )
        return vectors


def _model_sha256(model: Any) -> str:
    implementation = getattr(model, "model", None)
    model_dir = getattr(implementation, "_model_dir", None)
    description = getattr(implementation, "model_description", None)
    model_file = getattr(description, "model_file", None)
    if model_dir is None or not model_file:
        raise RuntimeError("cannot resolve the downloaded FastEmbed model file")
    path = Path(model_dir) / str(model_file)
    if not path.is_file():
        raise RuntimeError(f"downloaded FastEmbed model file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
