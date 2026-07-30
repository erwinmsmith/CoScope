"""Deterministic offline vectorizer for tests and simulated runtime."""

from __future__ import annotations

import hashlib
import math
import re


class DeterministicEmbedding:
    model_version = "deterministic-hash-v1"

    def __init__(self, dimension: int = 128):
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension

    def embed(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self.dimension
        for token in re.findall(r"[\w-]+", text.casefold()):
            digest = hashlib.blake2b(token.encode(), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm:
            values = [value / norm for value in values]
        return tuple(values)

    def embed_many(self, texts: list[str]) -> list[tuple[float, ...]]:
        return [self.embed(text) for text in texts]
