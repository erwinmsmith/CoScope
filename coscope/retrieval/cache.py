"""Scope-bound cache for public shared candidates only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from coscope.retrieval.grouping import RetrievalGroup
from coscope.retrieval.request import RetrievalCandidate


@dataclass(frozen=True)
class SharedCacheKey:
    scope_signature: str
    embedding_model_version: str
    representative_vector_hash: str
    retrieval_parameters_hash: str


class SharedRetrievalCache:
    def __init__(self) -> None:
        self._values: dict[SharedCacheKey, tuple[RetrievalCandidate, ...]] = {}

    def make_key(
        self,
        group: RetrievalGroup,
        *,
        embedding_model_version: str,
        retrieval_parameters: dict[str, object],
    ) -> SharedCacheKey:
        signatures = {request.scope_signature for request in group.requests}
        if len(signatures) != 1:
            raise ValueError("a cached group must have one scope signature")
        vector_raw = json.dumps(group.representative.public_vector, separators=(",", ":"))
        params_raw = json.dumps(
            retrieval_parameters, sort_keys=True, separators=(",", ":"), default=str
        )
        return SharedCacheKey(
            scope_signature=next(iter(signatures)),
            embedding_model_version=embedding_model_version,
            representative_vector_hash=hashlib.sha256(vector_raw.encode()).hexdigest(),
            retrieval_parameters_hash=hashlib.sha256(params_raw.encode()).hexdigest(),
        )

    def get(self, key: SharedCacheKey) -> list[RetrievalCandidate] | None:
        value = self._values.get(key)
        return list(value) if value is not None else None

    def put(
        self, key: SharedCacheKey, candidates: list[RetrievalCandidate]
    ) -> None:
        if any(candidate.source != "shared" for candidate in candidates):
            raise ValueError("private candidates must never enter the shared cache")
        self._values[key] = tuple(candidates)

    def __len__(self) -> int:
        return len(self._values)
