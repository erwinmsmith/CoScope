"""Embedding adapter contract."""

from __future__ import annotations

from typing import Protocol


class EmbeddingAdapter(Protocol):
    model_version: str

    def embed(self, text: str) -> tuple[float, ...]: ...

    def embed_many(self, texts: list[str]) -> list[tuple[float, ...]]: ...
