"""Zhipu embedding-3 adapter using its OpenAI-compatible API shape."""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from coscope.adapters.usage import usage_values
from coscope.config import EmbeddingSettings
from coscope.core.usage import UsageLedger


class ZhipuEmbedding:
    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        client: Any | None = None,
        usage_ledger: UsageLedger | None = None,
    ):
        settings.validate()
        self.settings = settings
        self.model_version = f"{settings.model}:{settings.dimension}"
        self.dimension = settings.dimension
        self.usage_ledger = usage_ledger
        self.client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
        )

    def embed(self, text: str) -> tuple[float, ...]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        started = time.perf_counter()
        response = self.client.embeddings.create(
            model=self.settings.model,
            input=texts,
            dimensions=self.settings.dimension,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts):
            raise RuntimeError(
                f"embedding response count mismatch: expected {len(texts)}, "
                f"received {len(ordered)}"
            )
        vectors = [tuple(float(value) for value in item.embedding) for item in ordered]
        if any(len(vector) != self.dimension for vector in vectors):
            raise RuntimeError(
                f"embedding response dimension does not match {self.dimension}"
            )
        if self.usage_ledger is not None:
            self.usage_ledger.record(
                "embedding",
                self.model_version,
                usage_values(getattr(response, "usage", None)),
                metadata={
                    "execution": "remote_api",
                    "input_count": str(len(texts)),
                    "latency_seconds": (
                        f"{time.perf_counter() - started:.9f}"
                    ),
                },
            )
        return vectors
